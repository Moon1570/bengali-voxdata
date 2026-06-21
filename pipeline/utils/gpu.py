"""utils.gpu — device and ONNX Runtime provider selection.

Detects CUDA and falls back to CPU with a warning so the scaffold runs on a
laptop (PLAN.md §2). Used by the InsightFace stages (s04 facetrack, s05 facerec).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("utils.gpu")


def has_cuda() -> bool:
    """Return True if a CUDA device is visible to PyTorch (if installed)."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — torch present but CUDA query failed
        return False


def onnx_providers() -> list[str]:
    """Return the preferred ONNX Runtime providers, best-available first.

    Prefers CUDA, then Apple CoreML, always ending with CPU as a fallback.
    Only providers actually available in this environment are returned.
    """
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except ImportError:
        return ["CPUExecutionProvider"]

    preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider",
                 "CPUExecutionProvider"]
    providers = [p for p in preferred if p in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    if providers[0] == "CPUExecutionProvider":
        logger.warning("no GPU/accelerator provider found; running on CPU")
    return providers


def ctx_id() -> int:
    """Return the InsightFace ctx_id: GPU index 0 if CUDA, else -1 (CPU)."""
    return 0 if has_cuda() else -1
