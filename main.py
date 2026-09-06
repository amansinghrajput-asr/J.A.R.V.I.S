"""Main entry point for J.A.R.V.I.S.

Executes the core application bootstrap sequence and launches the interactive
JarvisApplication runtime.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from app.application import JarvisApplication
from app.core.bootstrap import bootstrap


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for J.A.R.V.I.S.

    Args:
        args: Optional list of argument strings. Defaults to sys.argv[1:].

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="J.A.R.V.I.S Interactive Personal Assistant",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        default=False,
        help="Run J.A.R.V.I.S in Voice Interaction Mode using VoiceConversationEngine",
    )
    parsed, _ = parser.parse_known_args(args if args is not None else sys.argv[1:])
    return parsed


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
    cli_args = parse_args(args)
    voice_mode = bool(getattr(cli_args, "voice", False))

    result = bootstrap()
    if not result:
        return 1

    app = app_instance or JarvisApplication(voice_mode=voice_mode)
    if voice_mode and hasattr(app, "voice_mode"):
        app.voice_mode = True

    if interactive is False:
        return 0
    if interactive is None and not sys.stdin.isatty() and ("unittest" in sys.modules or "pytest" in sys.modules):
        return 0

    if voice_mode:
        return app.run(voice_mode=True)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
