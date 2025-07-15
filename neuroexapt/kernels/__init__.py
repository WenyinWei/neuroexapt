from __future__ import annotations

"""Kernels subpackage.
Provides Triton accelerated implementations if Triton is installed; otherwise
fallback helpers transparently use PyTorch reference ops.
"""

from typing import Optional

try:
    import triton  # type: ignore
    TRITON_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False

# Re-export helpers so consumers can simply:
#   from neuroexapt.kernels import sepconv_forward_generic, TRITON_AVAILABLE
from .sepconv_triton import sepconv_forward  as sepconv_forward_generic  # noqa: E402,F401 