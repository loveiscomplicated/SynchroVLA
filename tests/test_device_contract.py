from pathlib import Path

import pytest
import torch

from vla_gnn_recurrent.utils import select_device


def test_select_device_auto_prefers_cuda_then_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device("auto").type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"


def test_select_device_cuda_request_fails_cleanly_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        select_device("cuda")


def test_task_code_does_not_hard_code_backend_specific_device_calls() -> None:
    allowed = {
        Path("vla_gnn_recurrent/utils.py"),
        Path("AGENTS.md"),
    }
    forbidden = (".cuda(", ".mps(", 'torch.device("cuda"', "torch.device('cuda'", 'torch.device("mps"', "torch.device('mps'")
    offenders: list[str] = []
    for path in [*Path(".").glob("*.py"), *Path("vla_gnn_recurrent").rglob("*.py")]:
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern in text:
                offenders.append(f"{path}:{pattern}")

    assert offenders == []
