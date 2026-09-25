"""Whether pymbar can actually be imported here, and why not (any exception, not only ImportError)."""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple


@lru_cache(maxsize=1)
def pymbar_status() -> Tuple[bool, str]:
    try:
        import pymbar  # noqa: PLC0415
        from pymbar import MBAR  # noqa: F401,PLC0415
    except Exception as exc:  # e.g. AttributeError from an incompatible jax/scipy pair
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"pymbar {getattr(pymbar, '__version__', '?')}"


def warn_if_pymbar_unusable(context: str) -> bool:
    ok, msg = pymbar_status()
    if not ok:
        print(f"WARNING [pymbar]: unusable ({msg}); {context} will run degraded", flush=True)
    return ok
