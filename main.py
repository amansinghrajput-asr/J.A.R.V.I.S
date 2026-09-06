"""Main entry point for J.A.R.V.I.S.

Executes the core application bootstrap sequence to initialize configuration,
logging, and filesystem requirements.
"""

from __future__ import annotations

import sys

from app.core.bootstrap import bootstrap


def main() -> int:
    """Run the primary application bootstrap sequence.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    result = bootstrap()
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
