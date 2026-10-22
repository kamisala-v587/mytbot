"""Batched forward kinematics of the aloha-agilex arms (link6 origin), straight from the URDF, in torch.

Used to express action errors as end-effector position errors in cm. Only the two serial chains root -> fl_link6 and
root -> fr_link6 are parsed (fixed + revolute joints); no meshes, no extra dependency.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import torch

# copy of my_robotwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf shipped with the code
DEFAULT_URDF = str(Path(__file__).resolve().parents[4] / "assets/urdf/aloha-agilex/arx5_description_isaac.urdf")


def resolve_urdf(value: str) -> str:
    """dataset.heldout_monitor_urdf: "auto" = the shipped copy, "" = none."""
    return DEFAULT_URDF if value == "auto" else value
# arm -> (end link, slice of its 6 joints in the 14-dim aloha state / action)
ARMS = {"L": ("fl_link6", slice(0, 6)), "R": ("fr_link6", slice(7, 13))}


def _rpy(r: float, p: float, y: float) -> torch.Tensor:
    cr, sr, cp, sp, cy, sy = (torch.cos(torch.tensor(v, dtype=torch.float64)) if i % 2 == 0 else
                              torch.sin(torch.tensor(v, dtype=torch.float64))
                              for i, v in enumerate((r, r, p, p, y, y)))
    return torch.stack([
        torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
        torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
        torch.stack([-sp, cp * sr, cp * cr]),
    ])


class ArmFK:
    def __init__(self, urdf: str | Path = DEFAULT_URDF) -> None:
        root = ET.parse(str(urdf)).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        self.chains = {}
        for arm, (leaf, _) in ARMS.items():
            chain, link = [], leaf
            while link in by_child:
                j = by_child[link]
                chain.append(j)
                link = j.find("parent").get("link")
            steps = []
            for j in reversed(chain):
                o = j.find("origin")
                xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
                rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
                t = torch.eye(4, dtype=torch.float64)
                t[:3, :3], t[:3, 3] = _rpy(*rpy), torch.tensor(xyz, dtype=torch.float64)
                a = j.find("axis")
                axis = torch.tensor([float(v) for v in (a.get("xyz") if a is not None else "1 0 0").split()], dtype=torch.float64)
                steps.append((t, axis / axis.norm(), j.get("type") in ("revolute", "continuous")))
            if sum(s[2] for s in steps) != 6:
                raise ValueError(f"{arm}: expected 6 revolute joints to {leaf}, got {sum(s[2] for s in steps)}")
            self.chains[arm] = steps

    @staticmethod
    def _axis_rot(axis: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Rodrigues: [N] angles about a fixed unit axis -> [N, 4, 4]."""
        x, y, z = axis
        c, s = torch.cos(q), torch.sin(q)
        k = 1 - c
        r = torch.stack([
            torch.stack([c + x * x * k, x * y * k - z * s, x * z * k + y * s], -1),
            torch.stack([y * x * k + z * s, c + y * y * k, y * z * k - x * s], -1),
            torch.stack([z * x * k - y * s, z * y * k + x * s, c + z * z * k], -1),
        ], -2)
        out = torch.eye(4, dtype=q.dtype, device=q.device).repeat(len(q), 1, 1)
        out[:, :3, :3] = r
        return out

    def positions(self, q14: torch.Tensor) -> torch.Tensor:
        """[..., >=14] joint values (aloha order) -> [..., 2, 3] link6 positions in metres (left, right)."""
        shape = q14.shape[:-1]
        q = q14.reshape(-1, q14.shape[-1]).double()
        out = []
        for arm, (_, sl) in ARMS.items():
            t = torch.eye(4, dtype=torch.float64, device=q.device).repeat(len(q), 1, 1)
            qa, n = q[:, sl], 0
            for origin, axis, revolute in self.chains[arm]:
                t = t @ origin.to(q.device)
                if revolute:
                    t = t @ self._axis_rot(axis.to(q.device), qa[:, n])
                    n += 1
            out.append(t[:, :3, 3])
        return torch.stack(out, 1).reshape(*shape, 2, 3)
