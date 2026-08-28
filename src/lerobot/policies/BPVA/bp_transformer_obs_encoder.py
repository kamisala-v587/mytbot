from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.abc import MetaPathFinder
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


logger = logging.getLogger(__name__)


class _BlockWandbImport(MetaPathFinder):
    """阻止 timm 导入可选 wandb，避免环境里 wandb/urllib3 版本问题影响模型导入。"""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "wandb" or fullname.startswith("wandb."):
            raise ImportError("wandb is optional for BPVA BPObsEncoder")
        return None


@contextmanager
def _without_optional_wandb():
    finder = _BlockWandbImport()
    import sys

    previous_wandb = {name: module for name, module in sys.modules.items() if name == "wandb" or name.startswith("wandb.")}
    for name in previous_wandb:
        sys.modules.pop(name, None)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        for name, module in previous_wandb.items():
            sys.modules.setdefault(name, module)


def _import_timm():
    """延迟导入 timm。

    timm 顶层会尝试导入可选的 wandb；当前环境的 wandb/urllib3 组合会报 AttributeError。
    BPObsEncoder 不需要 wandb，所以这里在导入 timm 时屏蔽它。
    """
    with _without_optional_wandb():
        import timm
    return timm


DEFAULT_BP_CAMERA_KEYS = [
    f"{OBS_IMAGES}.image0",
    f"{OBS_IMAGES}.image1",
    f"{OBS_IMAGES}.image2",
]


@dataclass
class BPTransformerObsEncoderOutput:
    """Structured output for BPTransformerObsEncoder.

    Attributes:
        chunk_tokens: One token per behavior-prompt chunk, shape `(B, K, output_dim)`.
        modality_tokens: Five modality tokens per chunk, shape `(B, K, 5, token_dim)`.
        mask: Optional valid-chunk mask from input, shape `(B, K)` where True means valid.
    """

    chunk_tokens: torch.Tensor
    modality_tokens: torch.Tensor
    mask: torch.Tensor | None = None


@dataclass
class BPObsEncoderOutput:
    """完整 behavior prompt 的编码结果。"""

    chunk_tokens: torch.Tensor
    mask: torch.Tensor | None = None
    chunk_indices: torch.Tensor | None = None
    modality_tokens: torch.Tensor | None = None


class BPTransformerObsEncoder(nn.Module):
    """Encode behavior-prompt chunks into compact chunk-level tokens.

    This module is the BPVA counterpart of myva's TransformerObsEncoder,
    but it directly consumes behavior_prompt dictionaries and includes action
    chunks in the encoder path.

    Expected input schema:
        behavior_prompt["images"][camera_key]: `(B, K, 3, H, W)` or `(K, 3, H, W)`
        behavior_prompt["state"] or behavior_prompt[OBS_STATE]: `(B, K, state_dim)` or `(K, state_dim)`
        behavior_prompt["action"] or behavior_prompt[ACTION]: `(B, K, action_chunk_size, action_dim)`
            or `(K, action_chunk_size, action_dim)`
        behavior_prompt["mask"] optional: `(B, K)` or `(K,)`, True means valid chunk.

    Output:
        `(B, K, output_dim)` by default, where each token represents one BP
        chunk: three camera observations, robot state, and future action chunk.
    """

    def __init__(
        self,
        image_keys: list[str] | None = None,
        vision_model_name: str = "vit_base_patch16_clip_224.openai",
        pretrained: bool = False,
        vision_checkpoint_path: str | None = None,
        token_dim: int = 768,
        output_dim: int = 2048,
        state_dim: int = 32,
        action_dim: int = 32,
        action_chunk_size: int = 50,
        image_feature_aggregation: str = "cls",
        share_rgb_model: bool = True,
        use_vision_norm: bool = True,
        freeze_vision_encoder: bool = False,
        chunk_fusion_hidden_dim: int | None = None,
        use_action_step_embedding: bool = True,
        use_modality_type_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.image_keys = list(image_keys or DEFAULT_BP_CAMERA_KEYS)
        if len(self.image_keys) != 3:
            raise ValueError(f"BPTransformerObsEncoder expects exactly 3 image keys, got {len(self.image_keys)}")
        if image_feature_aggregation not in {"cls", "mean"}:
            raise ValueError("image_feature_aggregation must be one of {'cls', 'mean'}")

        self.token_dim = int(token_dim)
        self.output_dim = int(output_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_chunk_size = int(action_chunk_size)
        self.image_feature_aggregation = image_feature_aggregation
        self.share_rgb_model = bool(share_rgb_model)
        self.use_vision_norm = bool(use_vision_norm)
        self.use_action_step_embedding = bool(use_action_step_embedding)
        self.use_modality_type_embedding = bool(use_modality_type_embedding)

        local_checkpoint_path: str | None = None
        if vision_checkpoint_path:
            checkpoint_file = Path(vision_checkpoint_path).expanduser()
            if not checkpoint_file.exists():
                raise FileNotFoundError(
                    f"BP vision checkpoint file does not exist: {checkpoint_file}"
                )
            if not checkpoint_file.is_file():
                raise ValueError(
                    "BP vision checkpoint path must point to a checkpoint file "
                    f"(for example model.safetensors), got: {checkpoint_file}"
                )
            local_checkpoint_path = str(checkpoint_file)
            logger.info(
                "Loading BP vision pretrained weights from local checkpoint via timm's "
                "registered checkpoint filter (OpenCLIP visual conversion, network disabled): %s",
                local_checkpoint_path,
            )
        else:
            logger.info(
                "Creating BP vision encoder with timm pretrained=%s (no local checkpoint configured)",
                pretrained,
            )

        create_model_kwargs = {
            "model_name": vision_model_name,
            "pretrained": True if local_checkpoint_path is not None else pretrained,
            "global_pool": "",
            "num_classes": 0,
        }
        if local_checkpoint_path is not None:
            create_model_kwargs["pretrained_cfg_overlay"] = {
                "file": local_checkpoint_path,
                "hf_hub_id": None,
                "url": "",
            }

        timm = _import_timm()
        base_model = timm.create_model(**create_model_kwargs)
        model_data_config = timm.data.resolve_data_config(base_model.pretrained_cfg)
        mean = torch.tensor(model_data_config["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(model_data_config["std"], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

        self.image_module_names = {image_key: f"image_{idx}" for idx, image_key in enumerate(self.image_keys)}
        self.key_model_map = nn.ModuleDict()
        for index, image_key in enumerate(self.image_keys):
            module_name = self.image_module_names[image_key]
            if share_rgb_model or index == 0:
                vision_model = base_model # timm创建vit backbone
            else:
                vision_model = timm.create_model(**create_model_kwargs) # timm创建vit backbone
            self.key_model_map[module_name] = vision_model

        image_feature_dim = self._infer_image_feature_dim()
        self.image_projections = nn.ModuleDict(
            {
                self.image_module_names[image_key]: nn.Identity()
                if image_feature_dim == self.token_dim
                else nn.Linear(image_feature_dim, self.token_dim)
                for image_key in self.image_keys
            }
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.token_dim),
        )
        # 扩展1：action chunk 内部的第 0..T-1 步需要显式区分顺序。
        self.action_step_embedding = (
            nn.Embedding(self.action_chunk_size, self.action_dim)
            if self.use_action_step_embedding
            else None
        )
        self.action_proj = nn.Sequential(
            nn.LayerNorm(self.action_chunk_size * self.action_dim),
            nn.Linear(self.action_chunk_size * self.action_dim, self.token_dim * 2),
            nn.SiLU(),
            nn.Linear(self.token_dim * 2, self.token_dim),
        )

        #扩展2： 0/1/2=image0/1/2，3=state，4=action。
        self.modality_type_embedding = (
            nn.Embedding(5, self.token_dim)
            if self.use_modality_type_embedding
            else None
        )

        fusion_hidden_dim = int(chunk_fusion_hidden_dim or max(self.output_dim, self.token_dim * 2))
        self.chunk_fusion = nn.Sequential(
            nn.LayerNorm(5 * self.token_dim),
            nn.Linear(5 * self.token_dim, fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(fusion_hidden_dim, self.output_dim),
        )

        if freeze_vision_encoder:
            for model in self.key_model_map.values():
                for param in model.parameters():
                    param.requires_grad = False

    def forward(
        self,
        behavior_prompt: dict[str, Any],
        return_dict: bool = False,
    ) -> torch.Tensor | BPTransformerObsEncoderOutput:
        images = behavior_prompt["images"]
        state = self._get_field(behavior_prompt, "state", OBS_STATE)
        action = self._get_field(behavior_prompt, "action", ACTION)
        state, added_batch = self._ensure_batch_state(state)
        action, _ = self._ensure_batch_action(action)
        batch_size, num_chunks = state.shape[:2]

        if action.shape[:2] != (batch_size, num_chunks):
            raise ValueError(f"action shape {tuple(action.shape)} does not match state shape {tuple(state.shape)}")
        if action.shape[-2:] != (self.action_chunk_size, self.action_dim):
            raise ValueError(
                f"Expected action shape (..., {self.action_chunk_size}, {self.action_dim}), got {tuple(action.shape)}"
            )
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {state.shape[-1]}")

        # 1. 每个 chunk 得到 3 个图像 token。共享 backbone 时合并相机维，只执行一次 ViT forward。
        image_validity = self._prepare_image_validity(
            behavior_prompt.get("image_masks"), batch_size, num_chunks, state.device
        )
        image_tokens = self._encode_images(images, batch_size, num_chunks, image_validity) # 编码图像
        # 2. action 先加入 chunk 内 step 位置编码，再将完整动作序列压成一个 token。
        action_for_encoding = action
        if self.action_step_embedding is not None:
            step_ids = torch.arange(self.action_chunk_size, device=action.device)
            step_pos = self.action_step_embedding(step_ids).to(dtype=action.dtype)
            action_for_encoding = action + step_pos.view(1, 1, self.action_chunk_size, self.action_dim)

        # padding action 信息
        action_is_pad = behavior_prompt.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad, _ = self._ensure_batch_action_mask(action_is_pad)
            if action_is_pad.shape != action.shape[:-1]:
                raise ValueError(
                    f"action_is_pad shape {tuple(action_is_pad.shape)} does not match action {tuple(action.shape[:-1])}"
                )
            action_for_encoding = action_for_encoding.masked_fill(
                action_is_pad.to(device=action.device, dtype=torch.bool).unsqueeze(-1),
                0,
            )

        # 3. state 和 action 经投影后各形成一个 token，维度与图像 token 相同。
        state_token = self.state_proj(state).unsqueeze(2)
        state_is_available = behavior_prompt.get("state_is_available")
        if state_is_available is not None:
            state_is_available = self._ensure_batch_mask(state_is_available, added_batch).to(
                device=state_token.device, dtype=torch.bool
            )
            if state_is_available.shape != (batch_size, num_chunks):
                raise ValueError(
                    f"state_is_available shape {tuple(state_is_available.shape)} does not match "
                    f"{(batch_size, num_chunks)}"
                )
            state_token = state_token.masked_fill(~state_is_available.unsqueeze(-1).unsqueeze(-1), 0)
        action_token = self.action_proj(action_for_encoding.flatten(start_dim=2)).unsqueeze(2)
        action_is_available = None
        if action_is_pad is not None:
            action_is_available = ~action_is_pad.all(dim=-1)
            action_token = action_token.masked_fill(~action_is_available.unsqueeze(-1).unsqueeze(-1), 0)

        # 4. 3 个图像 token + 1 个 state token + 1 个 action token：(B, K, 5, 768)。
        modality_tokens = torch.cat([*image_tokens, state_token, action_token], dim=2)
        if self.modality_type_embedding is not None:
            modality_ids = torch.arange(5, device=modality_tokens.device)
            modality_pos = self.modality_type_embedding(modality_ids).to(dtype=modality_tokens.dtype) # 加入类型编码
            modality_tokens = modality_tokens + modality_pos.view(1, 1, 5, self.token_dim)
        modality_tokens[:, :, :3] = modality_tokens[:, :, :3].masked_fill(
            ~image_validity.unsqueeze(-1), 0
        )
        if state_is_available is not None:
            modality_tokens[:, :, 3] = modality_tokens[:, :, 3].masked_fill(
                ~state_is_available.unsqueeze(-1), 0
            )
        if action_is_available is not None:
            modality_tokens[:, :, 4] = modality_tokens[:, :, 4].masked_fill(
                ~action_is_available.unsqueeze(-1), 0
            )
        # 当前代码必定每个chunk这里处理为5个token 后续映射为一个token 输出
        if modality_tokens.shape[2] != 5:
            raise RuntimeError(f"Expected 5 modality tokens per chunk, got {modality_tokens.shape[2]}")

        # 5. 5 * 768 -> 2048：一个输出 token 表示一个完整 BP 行为块。
        chunk_tokens = self.chunk_fusion(modality_tokens.flatten(start_dim=2))

        mask = behavior_prompt.get("mask")
        if mask is not None:
            mask = self._ensure_batch_mask(mask, added_batch).to(device=chunk_tokens.device, dtype=torch.bool)
            if mask.shape != chunk_tokens.shape[:2]:
                raise ValueError(f"mask shape {tuple(mask.shape)} does not match chunk tokens {tuple(chunk_tokens.shape[:2])}")

        if added_batch:
            chunk_tokens = chunk_tokens.squeeze(0)
            modality_tokens = modality_tokens.squeeze(0)
            if mask is not None:
                mask = mask.squeeze(0)

        if return_dict:
            return BPTransformerObsEncoderOutput(chunk_tokens=chunk_tokens, modality_tokens=modality_tokens, mask=mask)
        return chunk_tokens

    @staticmethod
    def _get_field(data: dict[str, Any], short_key: str, canonical_key: str) -> torch.Tensor:
        if short_key in data:
            return data[short_key]
        if canonical_key in data:
            return data[canonical_key]
        raise KeyError(f"behavior_prompt must contain {short_key!r} or {canonical_key!r}")

    @staticmethod
    def _ensure_batch_state(state: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if state.ndim == 2:
            return state.unsqueeze(0), True
        if state.ndim == 3:
            return state, False
        raise ValueError(f"state must be (K, D) or (B, K, D), got {tuple(state.shape)}")

    @staticmethod
    def _ensure_batch_action(action: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if action.ndim == 3:
            return action.unsqueeze(0), True
        if action.ndim == 4:
            return action, False
        raise ValueError(f"action must be (K, T, D) or (B, K, T, D), got {tuple(action.shape)}")

    @staticmethod
    def _ensure_batch_action_mask(mask: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if mask.ndim == 2:
            return mask.unsqueeze(0), True
        if mask.ndim == 3:
            return mask, False
        raise ValueError(f"action_is_pad must be (K, T) or (B, K, T), got {tuple(mask.shape)}")

    @staticmethod
    def _ensure_batch_mask(mask: torch.Tensor, added_batch: bool) -> torch.Tensor:
        if mask.ndim == 1:
            return mask.unsqueeze(0)
        if mask.ndim == 2:
            return mask
        raise ValueError(f"mask must be (K,) or (B, K), got {tuple(mask.shape)}")
    def _prepare_image_batch(
        self,
        image: torch.Tensor,
        image_key: str,
        batch_size: int,
        num_chunks: int,
    ) -> torch.Tensor:
        """校验单路 BP 图像，并展平为 `(B*K, C, H, W)`。"""
        if image.ndim == 4:
            image = image.unsqueeze(0)
        elif image.ndim != 5:
            raise ValueError(
                f"{image_key} image must be (K, C, H, W) or (B, K, C, H, W), got {tuple(image.shape)}"
            )
        if image.shape[:2] != (batch_size, num_chunks):
            raise ValueError(
                f"{image_key} shape {tuple(image.shape)} does not match (B, K)=({batch_size}, {num_chunks})"
            )
        if image.shape[2] != 3:
            raise ValueError(f"{image_key} expected 3 channels, got {image.shape[2]}")
        return image.reshape(batch_size * num_chunks, *image.shape[2:])

    def _normalize_image_batch(self, flat: torch.Tensor, vision_model: nn.Module) -> torch.Tensor:
        flat = flat.to(dtype=next(vision_model.parameters()).dtype)
        if self.use_vision_norm:
            flat = (flat - self.image_mean.to(flat)) / self.image_std.to(flat)
        return flat

    def _prepare_image_validity(
        self,
        image_masks: dict[str, torch.Tensor] | None,
        batch_size: int,
        num_chunks: int,
        device: torch.device,
    ) -> torch.Tensor:
        if image_masks is None:
            return torch.ones(batch_size, num_chunks, len(self.image_keys), dtype=torch.bool, device=device)
        unknown = sorted(set(image_masks) - set(self.image_keys))
        if unknown:
            raise KeyError(f"image_masks contains unsupported camera keys: {unknown}")
        per_camera = []
        for image_key in self.image_keys:
            if image_key not in image_masks:
                raise KeyError(f"image_masks is missing configured camera {image_key!r}")
            mask = torch.as_tensor(image_masks[image_key], dtype=torch.bool, device=device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            elif mask.ndim != 2:
                raise ValueError(
                    f"image_masks[{image_key!r}] must be (K,) or (B, K), got {tuple(mask.shape)}"
                )
            if mask.shape == (1, num_chunks) and batch_size > 1:
                mask = mask.expand(batch_size, -1)
            if mask.shape != (batch_size, num_chunks):
                raise ValueError(
                    f"image_masks[{image_key!r}] shape {tuple(mask.shape)} does not match "
                    f"{(batch_size, num_chunks)}"
                )
            per_camera.append(mask)
        return torch.stack(per_camera, dim=-1)

    def _encode_images(
        self,
        images: dict[str, torch.Tensor],
        batch_size: int,
        num_chunks: int,
        image_validity: torch.Tensor,
    ) -> list[torch.Tensor]:
        """编码三路 BP 图像；共享 backbone 时将三路相机合并为一次 ViT 调用。"""
        flat_by_camera = [
            self._prepare_image_batch(images[image_key], image_key, batch_size, num_chunks)
            for image_key in self.image_keys
        ]
        items_per_camera = batch_size * num_chunks

        if self.share_rgb_model:
            shared_model = self.key_model_map[self.image_module_names[self.image_keys[0]]] # 共享 vit backbone
            merged = self._normalize_image_batch(torch.cat(flat_by_camera, dim=0), shared_model)
            merged_features = self._aggregate_image_features(shared_model(merged))
            features_by_camera = merged_features.split(items_per_camera, dim=0)
        else:
            features_by_camera = [
                self._aggregate_image_features(
                    self.key_model_map[self.image_module_names[image_key]](
                        self._normalize_image_batch(flat, self.key_model_map[self.image_module_names[image_key]])
                    )
                )
                for image_key, flat in zip(self.image_keys, flat_by_camera, strict=True)
            ]

        image_tokens = []
        for image_key, features in zip(self.image_keys, features_by_camera, strict=True):
            module_name = self.image_module_names[image_key]
            token = self.image_projections[module_name](features)
            if token.shape[-1] != self.token_dim:
                raise RuntimeError(f"{image_key} projected dim {token.shape[-1]} != token_dim {self.token_dim}")
            token = token.reshape(batch_size, num_chunks, 1, self.token_dim)
            camera_index = self.image_keys.index(image_key)
            token = token.masked_fill(~image_validity[:, :, camera_index, None, None], 0)
            image_tokens.append(token)
        return image_tokens

    def _aggregate_image_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            if self.image_feature_aggregation == "cls":
                return features[:, 0]
            return features.mean(dim=1)
        if features.ndim == 4:
            return features.flatten(start_dim=2).mean(dim=-1)
        if features.ndim == 2:
            return features
        raise ValueError(f"Unsupported image feature shape {tuple(features.shape)}")

    @torch.no_grad()
    def _infer_image_feature_dim(self) -> int:
        module_name = self.image_module_names[self.image_keys[0]]
        vision_model = self.key_model_map[module_name]
        device = next(vision_model.parameters()).device
        dtype = next(vision_model.parameters()).dtype
        example = torch.zeros(1, 3, 224, 224, device=device, dtype=dtype)
        features = vision_model(example)
        features = self._aggregate_image_features(features)
        return int(features.shape[-1])



class BPObsEncoder(nn.Module):
    """为完整 behavior prompt 加入 chunk 级顺序编码。

    chunk encoder 已向量化地独立编码所有 K 个 chunk；本类接收完整 row/batch
    或 behavior_prompt 字典，并在 `(B, K, 2048)` 上加入 chunk position embedding。
    """

    def __init__(
        self,
        chunk_encoder: BPTransformerObsEncoder | None = None,
        max_num_chunks: int = 4,
        chunk_dim: int | None = None,
        use_chunk_position_embedding: bool = True,
    ) -> None:
        super().__init__()
        if max_num_chunks <= 0:
            raise ValueError("max_num_chunks must be positive")
        if chunk_encoder is None:
            chunk_dim = int(chunk_dim or 2048)
            chunk_encoder = BPTransformerObsEncoder(output_dim=chunk_dim)
        else:
            inferred_chunk_dim = int(chunk_encoder.output_dim)
            if chunk_dim is not None and int(chunk_dim) != inferred_chunk_dim:
                raise ValueError(
                    f"chunk_dim ({chunk_dim}) does not match chunk_encoder.output_dim ({inferred_chunk_dim})"
                )
            chunk_dim = inferred_chunk_dim

        self.chunk_encoder = chunk_encoder
        self.max_num_chunks = int(max_num_chunks)
        self.chunk_dim = int(chunk_dim)
        self.chunk_position_embedding = (
            nn.Embedding(self.max_num_chunks, self.chunk_dim)
            if use_chunk_position_embedding
            else None
        )

    def forward(
        self,
        sample_or_behavior_prompt: dict[str, Any],
        return_dict: bool = False,
    ) -> torch.Tensor | BPObsEncoderOutput:
        behavior_prompt = sample_or_behavior_prompt.get("behavior_prompt", sample_or_behavior_prompt)
        # 实际执行编码
        encoded = self.chunk_encoder(behavior_prompt, return_dict=True)
        chunk_tokens = encoded.chunk_tokens
        mask = encoded.mask

        added_batch = chunk_tokens.ndim == 2
        if added_batch:
            chunk_tokens = chunk_tokens.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        if chunk_tokens.ndim != 3:
            raise ValueError(f"chunk tokens must be (K, D) or (B, K, D), got {tuple(chunk_tokens.shape)}")

        batch_size, num_chunks, chunk_dim = chunk_tokens.shape
        if chunk_dim != self.chunk_dim:
            raise ValueError(f"Expected chunk dim {self.chunk_dim}, got {chunk_dim}")

        chunk_indices = behavior_prompt.get("chunk_indices")
        if chunk_indices is None:
            chunk_indices = torch.arange(num_chunks, device=chunk_tokens.device).view(1, num_chunks)
            chunk_indices = chunk_indices.expand(batch_size, -1)
        else:
            chunk_indices = self._ensure_batch_chunk_indices(chunk_indices).to(
                device=chunk_tokens.device,
                dtype=torch.long,
            )
            if chunk_indices.shape != (batch_size, num_chunks):
                raise ValueError(
                    f"chunk_indices shape {tuple(chunk_indices.shape)} does not match {(batch_size, num_chunks)}"
                )

        if self.chunk_position_embedding is not None:
            if torch.any(chunk_indices < 0) or torch.any(chunk_indices >= self.max_num_chunks):
                raise ValueError(
                    f"chunk_indices must be in [0, {self.max_num_chunks - 1}], got "
                    f"min={int(chunk_indices.min())}, max={int(chunk_indices.max())}"
                )
            chunk_tokens = chunk_tokens + self.chunk_position_embedding(chunk_indices).to(dtype=chunk_tokens.dtype)

        if added_batch:
            chunk_tokens = chunk_tokens.squeeze(0)
            chunk_indices = chunk_indices.squeeze(0)
            if mask is not None:
                mask = mask.squeeze(0)

        if return_dict:
            return BPObsEncoderOutput(
                chunk_tokens=chunk_tokens,
                mask=mask,
                chunk_indices=chunk_indices,
                modality_tokens=encoded.modality_tokens,
            )
        return chunk_tokens

    @staticmethod
    def _ensure_batch_chunk_indices(chunk_indices: torch.Tensor) -> torch.Tensor:
        if chunk_indices.ndim == 1:
            return chunk_indices.unsqueeze(0)
        if chunk_indices.ndim == 2:
            return chunk_indices
        raise ValueError(f"chunk_indices must be (K,) or (B, K), got {tuple(chunk_indices.shape)}")
