"""J.A.R.V.I.S — AI Desktop Operating System Assistant package."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "JarvisApplication",
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name == "JarvisApplication":
        from app.application import JarvisApplication
        return JarvisApplication
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
