from __future__ import annotations

from dataclasses import dataclass

import torch


Q35_TOKEN_ROLES = ("vlm", "state", "action")
Q35_STREAM_BY_ROLE = {"vlm": 0, "state": 1, "action": 2}


@dataclass(frozen=True)
class Q35TokenLayout:
    """Token spans and block-causal visibility for the action-only policy."""

    lengths: tuple[int, ...]

    @classmethod
    def from_lengths(cls, **lengths: int) -> "Q35TokenLayout":
        unknown = set(lengths) - set(Q35_TOKEN_ROLES)
        if unknown:
            raise ValueError(f"Unknown Q35 token roles: {sorted(unknown)}")
        ordered = tuple(int(lengths.get(role, 0)) for role in Q35_TOKEN_ROLES)
        if any(length < 0 for length in ordered):
            raise ValueError("Token lengths cannot be negative")
        return cls(ordered)

    @classmethod
    def for_step(
        cls,
        step_type: str,
        *,
        expert_type: str = "action",
        **lengths: int,
    ) -> "Q35TokenLayout":
        if step_type != "robot":
            raise ValueError("Q35 supports robot Flow steps only")
        if expert_type != "action":
            raise ValueError("Q35 contains only the action expert")
        layout = cls.from_lengths(**lengths)
        if layout.length("state") != 1:
            raise ValueError("Action-expert robot steps require exactly one state token")
        if layout.length("action") <= 0:
            raise ValueError("Action-expert robot steps require action tokens")
        return layout

    @property
    def roles(self) -> tuple[str, ...]:
        return Q35_TOKEN_ROLES

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)

    def length(self, role: str) -> int:
        return self.lengths[self.roles.index(role)]

    def span(self, role: str) -> slice:
        role_index = self.roles.index(role)
        start = sum(self.lengths[:role_index])
        return slice(start, start + self.lengths[role_index])

    def build_attention_mask(
        self,
        *,
        batch_size: int | None = None,
        valid_mask: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        causal_vlm: bool = False,
    ) -> torch.Tensor:
        """Return the bool `[batch, 1, query, key]` architecture mask."""

        if valid_mask is not None:
            if valid_mask.ndim != 2 or valid_mask.shape[1] != self.total_tokens:
                raise ValueError(
                    f"valid_mask must have shape [batch, {self.total_tokens}], "
                    f"got {tuple(valid_mask.shape)}"
                )
            batch_size = valid_mask.shape[0]
            device = valid_mask.device
            valid_mask = valid_mask.to(dtype=torch.bool)
        elif batch_size is None:
            raise ValueError("batch_size is required when valid_mask is not provided")

        stream_ids = torch.cat(
            [
                torch.full(
                    (length,),
                    Q35_STREAM_BY_ROLE[role],
                    dtype=torch.long,
                    device=device,
                )
                for role, length in zip(self.roles, self.lengths, strict=True)
            ]
        )
        base = stream_ids[:, None] >= stream_ids[None, :]
        if causal_vlm:
            token_ids = torch.arange(self.total_tokens, device=device)
            vlm = self.span("vlm")
            is_vlm = (token_ids >= vlm.start) & (token_ids < vlm.stop)
            vlm_square = is_vlm[:, None] & is_vlm[None, :]
            causal = token_ids[:, None] >= token_ids[None, :]
            base = base & (~vlm_square | causal)

        mask = base[None, None].expand(int(batch_size), 1, -1, -1)
        if valid_mask is not None:
            valid_2d = valid_mask[:, :, None] & valid_mask[:, None, :]
            mask = mask & valid_2d[:, None]
        return mask


__all__ = ["Q35TokenLayout"]
