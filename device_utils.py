"""Cross-platform device selection for the LLM router.

Supported accelerators, in priority order:
  1. cuda   - NVIDIA GPUs, and AMD GPUs via ROCm (Linux, and Windows on
              RX 7000/9000-series via AMD's official wheels). ROCm builds
              expose the GPU through the regular torch.cuda API.
  2. mps    - Apple Silicon Macs.
  3. directml - AMD/Intel GPUs on Windows for which no ROCm build exists
              (requires the optional `torch-directml` package).
  4. cpu    - Fallback.

Pass --device on the command line of any script to override the detection.
"""

import importlib
import os
from types import ModuleType

import torch

# Let PyTorch fall back to CPU for the handful of ops MPS does not implement
# instead of crashing mid-training.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

VALID_DEVICES = ("auto", "cuda", "mps", "directml", "cpu")


def _load_directml() -> ModuleType | None:
    """Import the optional torch-directml package, or None if not installed.

    importlib is used instead of a plain import so type checkers do not flag
    the package as missing on the platforms where it cannot be installed.
    """
    try:
        return importlib.import_module("torch_directml")
    except ImportError:
        return None


def select_device(override: str | None = None) -> torch.device:
    """Pick the best available torch device, or return `override` if given.

    Accepts "auto" (detect), "cuda", "mps", "directml" or "cpu". Returns a
    torch.device — note that for DirectML this is a `privateuseone` device
    object, which is why callers should pass it around as-is rather than
    reconstructing it from a string.
    """
    if override and override != "auto":
        return _resolve_override(override)

    if torch.cuda.is_available():
        print(f"[device] Using CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        print("[device] Using Apple Silicon MPS device (CPU fallback enabled)")
        return torch.device("mps")

    directml = _load_directml()
    if directml is not None and directml.device_count() > 0:
        device = directml.device()
        print(f"[device] Using DirectML device: {directml.device_name(0)}")
        return device

    print("[device] No accelerator found, using CPU")
    return torch.device("cpu")


def _resolve_override(override: str) -> torch.device:
    if override == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "[device] --device cuda requested, but torch.cuda.is_available() "
                "is False. If this is an AMD GPU, install a ROCm build of PyTorch "
                "(see README.md)."
            )
        print(f"[device] Using CUDA device (forced): {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    if override == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit(
                "[device] --device mps requested, but MPS is not available "
                "on this system."
            )
        print("[device] Using Apple Silicon MPS device (forced)")
        return torch.device("mps")

    if override == "directml":
        directml = _load_directml()
        if directml is None:
            raise SystemExit(
                "[device] --device directml requested, but torch-directml is "
                "not installed. Run: pip install torch-directml"
            )
        device = directml.device()
        print(f"[device] Using DirectML device (forced): {directml.device_name(0)}")
        return device

    if override == "cpu":
        print("[device] Using CPU (forced)")
        return torch.device("cpu")

    raise SystemExit(
        f"[device] Unknown device '{override}'. Expected one of: "
        + ", ".join(VALID_DEVICES)
    )
