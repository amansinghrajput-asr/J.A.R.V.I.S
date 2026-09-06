"""Application Bootstrap Subsystem for J.A.R.V.I.S.

Responsible for orchestrating the core startup sequence:
1. Configuration loading and validation.
2. Centralized logging subsystem initialization.
3. Startup banner and system metadata display.
4. Core environment and directory verification.
5. Graceful failure recovery and diagnostics.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

from app.core import logger as jarvis_logger
from app.core.config import Settings, load_settings
from app.core.logger import get_logger

# Professional ASCII Art Banner for J.A.R.V.I.S
BANNER_ART: Final[str] = r"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
"""

BANNER_DIVIDER: Final[str] = "=" * 70


@dataclass(frozen=True)
class BootstrapResult:
    """Encapsulates the outcome of the bootstrap sequence.

    Attributes:
        success: Boolean flag indicating whether bootstrap succeeded.
        settings: Verified Settings instance if successful, else None.
        logger: Configured system Logger instance if successful, else None.
        error_message: Diagnostic error explanation if bootstrap failed, else None.
    """

    success: bool
    settings: Optional[Settings] = None
    logger: Optional[logging.Logger] = None
    error_message: Optional[str] = None

    def __bool__(self) -> bool:
        """Enable direct boolean evaluation of the bootstrap result."""
        return self.success

    def __eq__(self, other: object) -> bool:
        """Allow comparison against booleans or other BootstrapResult instances."""
        if isinstance(other, bool):
            return self.success == other
        return super().__eq__(other)


def get_banner(settings_instance: Settings) -> str:
    """Generate the formatted startup banner string.

    Args:
        settings_instance: Loaded Settings containing metadata to display.

    Returns:
        Multi-line formatted banner string.
    """
    metadata_lines = [
        BANNER_DIVIDER,
        BANNER_ART.strip("\n"),
        "       Just A Rather Very Intelligent System — Desktop OS Co-Pilot",
        BANNER_DIVIDER,
        f" Application : {settings_instance.app_name}",
        f" Version     : {settings_instance.version}",
        f" Environment : {settings_instance.environment}",
        f" Log Level   : {settings_instance.log_level}",
        BANNER_DIVIDER,
    ]
    return "\n".join(metadata_lines)


def _display_banner(settings_instance: Settings) -> None:
    """Safely print the startup banner to standard output.

    Args:
        settings_instance: Loaded Settings containing metadata to display.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    sys.stdout.write(get_banner(settings_instance) + "\n")
    sys.stdout.flush()


def _initialize_logging(settings_instance: Settings) -> logging.Logger:
    """Initialize centralized logging with active configuration.

    Args:
        settings_instance: Loaded Settings containing log level and directory.

    Returns:
        Configured SYSTEM logger instance.
    """
    root_logger = logging.getLogger()
    log_dir = getattr(settings_instance, "log_directory", None)
    log_level = getattr(settings_instance, "log_level", "INFO")
    if not isinstance(log_level, str):
        log_level = "INFO"

    target_path = log_dir if isinstance(log_dir, Path) else None

    if not root_logger.handlers or not jarvis_logger._INITIALIZED:
        jarvis_logger._INITIALIZED = False
        jarvis_logger._setup_root_logging(
            log_dir=target_path,
            log_level_name=log_level,
        )
    return jarvis_logger.get_logger("SYSTEM")



def _verify_configuration(settings_instance: Settings) -> None:
    """Verify that configuration is loaded and valid.

    Args:
        settings_instance: Settings instance to inspect.

    Raises:
        RuntimeError: If configuration is invalid or missing required structure.
    """
    if not isinstance(settings_instance, Settings):
        raise RuntimeError("Configuration is not a valid Settings instance.")

    if not settings_instance.app_name:
        raise RuntimeError("Application Name cannot be empty.")

    if not settings_instance.version:
        raise RuntimeError("Application Version cannot be empty.")

    if not settings_instance.environment:
        raise RuntimeError("Application Environment cannot be empty.")

    try:
        settings_instance.validate(
            require_api_keys=(settings_instance.environment == "production")
        )
    except Exception as exc:
        raise RuntimeError(f"Configuration validation failed: {exc}") from exc


def _verify_logger(system_logger: logging.Logger) -> None:
    """Verify that the logging subsystem is operational and handlers attached.

    Args:
        system_logger: Logger instance to inspect.

    Raises:
        RuntimeError: If the logger or root logger is uninitialized or non-functional.
    """
    if not isinstance(system_logger, logging.Logger):
        raise RuntimeError("Logger instance is invalid.")

    root_logger = logging.getLogger()
    if not root_logger.handlers and not system_logger.handlers:
        raise RuntimeError("Logging subsystem initialized with zero handlers.")


def _verify_directories(settings_instance: Settings) -> None:
    """Ensure required system directories exist and are valid directories.

    Args:
        settings_instance: Settings instance containing configured directory paths.

    Raises:
        RuntimeError: If any required directory cannot be created or accessed.
    """
    try:
        settings_instance.ensure_directories()
    except Exception as exc:
        raise RuntimeError(f"Failed to ensure required directories: {exc}") from exc

    required_dirs: tuple[tuple[str, Path], ...] = (
        ("Data", settings_instance.data_dir),
        ("Models", settings_instance.models_dir),
        ("Logs", settings_instance.logs_dir),
        ("Configs", settings_instance.config_dir),
    )

    for name, dir_path in required_dirs:
        if not dir_path.exists():
            raise RuntimeError(
                f"Required {name} directory does not exist: {dir_path}"
            )
        if not dir_path.is_dir():
            raise RuntimeError(
                f"Required {name} path is not a directory: {dir_path}"
            )


def _display_failure_message(error_message: str) -> None:
    """Display a friendly, formatted diagnostic message upon failure.

    Args:
        error_message: Diagnostic description of the initialization failure.
    """
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    message = (
        "\n"
        "======================================================================\n"
        "             [!] J.A.R.V.I.S Startup Initialization Failed\n"
        "======================================================================\n"
        f" An unrecoverable error occurred while bootstrapping the system:\n\n"
        f"   • Reason: {error_message}\n\n"
        " Suggested Steps:\n"
        "   1. Review your '.env' configuration or environment variables.\n"
        "   2. Verify folder read/write permissions for data, models, logs, and configs.\n"
        "   3. Check the application log files in the 'logs/' directory for details.\n"
        "======================================================================\n"
    )
    sys.stderr.write(message)
    sys.stderr.flush()


def bootstrap(
    settings_instance: Optional[Settings] = None,
    *,
    exit_on_failure: bool = True,
    display_banner: bool = True,
) -> BootstrapResult:
    """Initialize and verify the core application runtime environment.

    Orchestrates the complete bootstrap lifecycle:
    1. Loads and validates configuration settings.
    2. Initializes centralized logging subsystem.
    3. Displays startup banner with Application Name, Version, and Environment.
    4. Verifies loaded configuration, logger readiness, and required directories.
    5. Returns a success state (BootstrapResult).
    6. If initialization fails, logs the failure, prints a user-friendly error,
       and exits gracefully (when exit_on_failure=True).

    Args:
        settings_instance: Optional pre-loaded Settings instance. If None,
            loads settings via `load_settings()`.
        exit_on_failure: Whether to invoke sys.exit(1) on failure. Defaults to True.
        display_banner: Whether to output the startup banner to stdout. Defaults to True.

    Returns:
        BootstrapResult containing initialization state and core instances.

    Example:
        >>> from app.core.bootstrap import bootstrap
        >>> result = bootstrap()
        >>> if result:
        ...     print("System ready.")
    """
    try:
        # 1. Initialize configuration
        active_settings = settings_instance or load_settings()

        # 2. Initialize logging
        system_logger = _initialize_logging(active_settings)

        # 3 & 4. Display professional banner with Name, Version, Environment
        if display_banner:
            _display_banner(active_settings)

        # 5. Verify system integrity
        _verify_configuration(active_settings)
        _verify_logger(system_logger)
        _verify_directories(active_settings)

        system_logger.info(
            "J.A.R.V.I.S bootstrap completed successfully [env=%s, v=%s].",
            active_settings.environment,
            active_settings.version,
        )

        # 6. Return success state
        return BootstrapResult(
            success=True,
            settings=active_settings,
            logger=system_logger,
        )

    except Exception as exc:
        err_msg = str(exc)

        # 7a. Log the error if logger subsystem is accessible
        try:
            err_logger = logging.getLogger("SYSTEM")
            err_logger.critical(
                "Bootstrap initialization failed: %s",
                err_msg,
                exc_info=True,
            )
        except Exception:
            pass

        # 7b. Display friendly diagnostic message
        _display_failure_message(err_msg)

        # 7c. Exit gracefully if requested
        if exit_on_failure:
            sys.exit(1)

        return BootstrapResult(
            success=False,
            error_message=err_msg,
        )
