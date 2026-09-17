from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch


DevicePreference = Literal["auto", "cpu", "mps", "cuda"]


def set_seed(seed: int) -> None:
    """Set deterministic seeds where PyTorch exposes stable controls."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def select_device(preference: DevicePreference = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch build.")
        return torch.device("mps")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch build.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def tensor3(value: torch.Tensor | list[float] | tuple[float, float, float]) -> torch.Tensor:
    out = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if out.shape != (3,):
        raise ValueError(f"Expected a 3D vector, got shape {tuple(out.shape)}")
    return out


def clamp_delta(delta: torch.Tensor, max_step: float) -> torch.Tensor:
    """Clamp a vector or batch of vectors to a maximum L2 magnitude."""
    norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = torch.clamp(max_step / norm, max=1.0)
    return delta * scale
