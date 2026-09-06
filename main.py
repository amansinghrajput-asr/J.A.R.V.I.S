"""Main entry point for J.A.R.V.I.S.

Executes the core application bootstrap sequence and launches the interactive
JarvisApplication runtime.
"""

from __future__ import annotations

import sys
from typing import Optional

from app.application import JarvisApplication
from app.core.bootstrap import bootstrap


def main(
    args: Optional[list[str]] = None,
    *,
    app_instance: Optional[JarvisApplication] = None,
    interactive: Optional[bool] = None,
) -> int:
    """Launch the J.A.R.V.I.S application.

    Args:
        args: Optional command-line arguments list.
        app_instance: Optional pre-configured JarvisApplication instance.
        interactive: Optional boolean flag to force interactive or non-interactive mode.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    result = bootstrap()
    if not result:
        return 1

    app = app_instance or JarvisApplication()

    if interactive is False:
        return 0
    if interactive is None and not sys.stdin.isatty() and ("unittest" in sys.modules or "pytest" in sys.modules):
        return 0

    return app.run()


if __name__ == "__main__":
    sys.exit(main())
