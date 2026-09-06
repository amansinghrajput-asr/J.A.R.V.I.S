"""J.A.R.V.I.S Configuration Manager.

This module provides a typed, immutable, single source of truth configuration
system loaded from environment variables and `.env` files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

from app.core.constants import (
    CONFIGS_DIR_NAME,
    DATA_DIR_NAME,
    DEFAULT_APP_NAME,
    DEFAULT_DEBUG,
    DEFAULT_ENVIRONMENT,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_TTS_VOICE_EN,
    DEFAULT_TTS_VOICE_HI,
    DEFAULT_VERSION,
    DEFAULT_WAKE_WORD,
    LOGS_DIR_NAME,
    MODELS_DIR_NAME,
    VALID_ENVIRONMENTS,
    VALID_LANGUAGES,
    VALID_LOG_LEVELS,
)


def _load_env_file(dotenv_path: Path) -> None:
    """Load environment variables from a .env file into os.environ.

    Attempts to use `dotenv.load_dotenv` if available. Otherwise, falls back
    to a pure-Python parser ensuring zero hard dependencies for configuration.

    Args:
        dotenv_path: Absolute path to the .env file.
    """
    if not dotenv_path.is_file():
        return

    # 1. Attempt using python-dotenv if installed
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv(dotenv_path=dotenv_path, override=False)
        return
    except ImportError:
        pass

    # 2. Built-in lightweight fallback parser
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Strip matching surrounding quotes if present
                if len(value) >= 2 and (
                    (value.startswith('"') and value.endswith('"'))
                    or (value.startswith("'") and value.endswith("'"))
                ):
                    value = value[1:-1]
                # Only set if not already set in environment
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Avoid crashing during boot on unreadable .env
        pass


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    """Parse a boolean value from an environment variable string.

    Args:
        value: String representation of boolean.
        default: Fallback boolean if value is None or empty.

    Returns:
        Parsed boolean value.
    """
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


@dataclass(frozen=True)
class AppConfig:
    """Application metadata and runtime mode configuration.

    Attributes:
        app_name: The name of the application.
        version: Semantic version string.
        environment: Deployment environment (development, testing, production).
        debug: Flag indicating whether debug mode is active.
    """

    app_name: str
    version: str
    environment: str
    debug: bool


@dataclass(frozen=True)
class AIConfig:
    """AI and LLM provider credentials and model choices.

    Attributes:
        gemini_api_key: API key for Google Gemini (Primary).
        gemini_model: Target Gemini model identifier.
        openrouter_api_key: API key for OpenRouter (Fallback).
        openrouter_model: Target OpenRouter free model identifier.
    """

    gemini_api_key: str
    gemini_model: str
    openrouter_api_key: str
    openrouter_model: str


@dataclass(frozen=True)
class VoiceConfig:
    """Speech, wake word, and audio synthesis configuration.

    Attributes:
        language: Primary language code ('en', 'hi', or 'hinglish').
        wake_word: Wake word trigger phrase (e.g., 'jarvis').
        tts_voice: Neural TTS voice model identifier.
    """

    language: str
    wake_word: str
    tts_voice: str


@dataclass(frozen=True)
class LoggingConfig:
    """Logging subsystem configuration.

    Attributes:
        log_level: Minimum severity level for log recording.
        log_directory: Target directory path for file logs.
    """

    log_level: str
    log_directory: Path


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem path configuration for application directories.

    Attributes:
        base_dir: Root workspace directory.
        data_dir: Directory for persistent database and state storage.
        models_dir: Directory for offline models (weights, wake word models).
        logs_dir: Directory for rotating application logs.
        config_dir: Directory for static configuration files and prompts.
    """

    base_dir: Path
    data_dir: Path
    models_dir: Path
    logs_dir: Path
    config_dir: Path


@dataclass(frozen=True)
class Settings:
    """Master immutable settings container for J.A.R.V.I.S.

    Acts as the single source of truth across all subsystems. Provides
    hierarchical component configs and direct top-level access properties.
    """

    app: AppConfig
    ai: AIConfig
    voice: VoiceConfig
    logging: LoggingConfig
    paths: PathsConfig

    # --------------------------------------------------------------------------
    # Top-Level Direct Property Access
    # --------------------------------------------------------------------------

    @property
    def app_name(self) -> str:
        """Application name."""
        return self.app.app_name

    @property
    def version(self) -> str:
        """Application version."""
        return self.app.version

    @property
    def environment(self) -> str:
        """Application environment."""
        return self.app.environment

    @property
    def debug(self) -> bool:
        """Debug mode flag."""
        return self.app.debug

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app.environment == "development"

    @property
    def gemini_api_key(self) -> str:
        """Google Gemini API key."""
        return self.ai.gemini_api_key

    @property
    def gemini_model(self) -> str:
        """Google Gemini model identifier."""
        return self.ai.gemini_model

    @property
    def openrouter_api_key(self) -> str:
        """OpenRouter API key."""
        return self.ai.openrouter_api_key

    @property
    def openrouter_model(self) -> str:
        """OpenRouter model identifier."""
        return self.ai.openrouter_model

    @property
    def language(self) -> str:
        """Configured primary language."""
        return self.voice.language

    @property
    def wake_word(self) -> str:
        """Wake word phrase."""
        return self.voice.wake_word

    @property
    def tts_voice(self) -> str:
        """Configured TTS voice model."""
        return self.voice.tts_voice

    @property
    def log_level(self) -> str:
        """Logging severity level."""
        return self.logging.log_level

    @property
    def log_directory(self) -> Path:
        """Path to log files directory."""
        return self.logging.log_directory

    @property
    def data_dir(self) -> Path:
        """Path to data directory."""
        return self.paths.data_dir

    @property
    def models_dir(self) -> Path:
        """Path to models directory."""
        return self.paths.models_dir

    @property
    def logs_dir(self) -> Path:
        """Path to logs directory."""
        return self.paths.logs_dir

    @property
    def config_dir(self) -> Path:
        """Path to configs directory."""
        return self.paths.config_dir

    def ensure_directories(self) -> None:
        """Ensure all required system directories exist on the filesystem."""
        for directory in (
            self.paths.data_dir,
            self.paths.models_dir,
            self.paths.logs_dir,
            self.paths.config_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def validate(self, require_api_keys: bool = False) -> None:
        """Validate configuration parameters.

        Args:
            require_api_keys: If True, raises ValueError if neither Gemini nor
                OpenRouter API key is set.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.app.environment not in VALID_ENVIRONMENTS:
            raise ValueError(
                f"Invalid ENVIRONMENT '{self.app.environment}'. "
                f"Must be one of: {sorted(VALID_ENVIRONMENTS)}"
            )

        if self.logging.log_level not in VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid LOG_LEVEL '{self.logging.log_level}'. "
                f"Must be one of: {sorted(VALID_LOG_LEVELS)}"
            )

        if self.voice.language not in VALID_LANGUAGES:
            raise ValueError(
                f"Invalid LANGUAGE '{self.voice.language}'. "
                f"Must be one of: {sorted(VALID_LANGUAGES)}"
            )

        if require_api_keys and not self.gemini_api_key and not self.openrouter_api_key:
            raise ValueError(
                "At least one AI API key (GEMINI_API_KEY or OPENROUTER_API_KEY) "
                "must be configured."
            )


def load_settings(
    base_dir: Optional[Path] = None,
    env_file: Optional[str] = ".env",
    validate_on_load: bool = True,
) -> Settings:
    """Factory function to build an immutable Settings singleton.

    Loads environment variables from `.env` file if present, inspects
    `os.environ`, applies defaults, builds path hierarchies, and validates
    configuration state.

    Args:
        base_dir: Root project directory. Defaults to repository root.
        env_file: Filename of the env file to load. Defaults to '.env'.
        validate_on_load: If True, validates configuration immediately.

    Returns:
        An immutable Settings instance.
    """
    if base_dir is None:
        # Determine base directory from location of this file: app/core/config.py -> J.A.R.V.I.S root
        base_dir = Path(__file__).resolve().parent.parent.parent
    else:
        base_dir = base_dir.resolve()

    if env_file:
        _load_env_file(base_dir / env_file)

    # 1. Application Configuration
    app_name = os.getenv("APP_NAME", DEFAULT_APP_NAME).strip()
    version = os.getenv("VERSION", DEFAULT_VERSION).strip()
    environment = os.getenv("ENVIRONMENT", DEFAULT_ENVIRONMENT).strip().lower()
    debug = _parse_bool(os.getenv("DEBUG"), default=DEFAULT_DEBUG)

    app_config = AppConfig(
        app_name=app_name,
        version=version,
        environment=environment,
        debug=debug,
    )

    # 2. AI Configuration
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_model = os.getenv(
        "OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL
    ).strip()

    ai_config = AIConfig(
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
    )

    # 3. Voice Configuration
    language = os.getenv("LANGUAGE", DEFAULT_LANGUAGE).strip().lower()
    wake_word = os.getenv("WAKE_WORD", DEFAULT_WAKE_WORD).strip().lower()

    # Determine default TTS voice based on language
    default_tts = (
        DEFAULT_TTS_VOICE_HI if language == "hi" else DEFAULT_TTS_VOICE_EN
    )
    tts_voice = os.getenv("TTS_VOICE", default_tts).strip()

    voice_config = VoiceConfig(
        language=language,
        wake_word=wake_word,
        tts_voice=tts_voice,
    )

    # 4. Paths Configuration
    data_dir_env = os.getenv("DATA_DIR")
    data_dir = (
        Path(data_dir_env).resolve()
        if data_dir_env
        else (base_dir / DATA_DIR_NAME)
    )

    models_dir_env = os.getenv("MODELS_DIR")
    models_dir = (
        Path(models_dir_env).resolve()
        if models_dir_env
        else (base_dir / MODELS_DIR_NAME)
    )

    logs_dir_env = os.getenv("LOGS_DIR") or os.getenv("LOG_DIRECTORY")
    logs_dir = (
        Path(logs_dir_env).resolve()
        if logs_dir_env
        else (base_dir / LOGS_DIR_NAME)
    )

    config_dir_env = os.getenv("CONFIG_DIR")
    config_dir = (
        Path(config_dir_env).resolve()
        if config_dir_env
        else (base_dir / CONFIGS_DIR_NAME)
    )

    paths_config = PathsConfig(
        base_dir=base_dir,
        data_dir=data_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        config_dir=config_dir,
    )

    # 5. Logging Configuration
    log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).strip().upper()

    logging_config = LoggingConfig(
        log_level=log_level,
        log_directory=logs_dir,
    )

    settings_instance = Settings(
        app=app_config,
        ai=ai_config,
        voice=voice_config,
        logging=logging_config,
        paths=paths_config,
    )

    if validate_on_load:
        settings_instance.validate(
            require_api_keys=(environment == "production")
        )

    return settings_instance


# ------------------------------------------------------------------------------
# Singleton Instance Export
# ------------------------------------------------------------------------------
settings: Final[Settings] = load_settings()
