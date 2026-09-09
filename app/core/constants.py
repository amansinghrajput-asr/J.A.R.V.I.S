"""Core constants and defaults for J.A.R.V.I.S."""

from typing import Final

# Application Information Defaults
DEFAULT_APP_NAME: Final[str] = "J.A.R.V.I.S"
DEFAULT_VERSION: Final[str] = "0.1.0"
DEFAULT_ENVIRONMENT: Final[str] = "development"
DEFAULT_DEBUG: Final[bool] = False

# Valid Environments
VALID_ENVIRONMENTS: Final[frozenset[str]] = frozenset(
    {"development", "testing", "staging", "production"}
)

# AI Models & Free Tier Defaults
DEFAULT_AI_PROVIDER: Final[str] = "gemini"
DEFAULT_GEMINI_MODEL: Final[str] = "gemini-2.5-flash"
DEFAULT_OLLAMA_HOST: Final[str] = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL: Final[str] = "qwen2.5:3b"
DEFAULT_OPENROUTER_MODEL: Final[str] = "meta-llama/llama-3.3-70b-instruct:free"
DEFAULT_AI_TIMEOUT: Final[float] = 30.0
DEFAULT_AI_MAX_RETRIES: Final[int] = 3
DEFAULT_AI_HISTORY_LIMIT: Final[int] = 20

# Known Supported AI Providers
KNOWN_AI_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"gemini", "ollama", "openrouter", "openai", "claude", "deepseek"}
)

# Voice & Speech Defaults
DEFAULT_LANGUAGE: Final[str] = "en"
DEFAULT_WAKE_WORD: Final[str] = "jarvis"
DEFAULT_WAKE_PHRASES: Final[tuple[str, ...]] = (
    "Hey Jarvis",
    "Activate Jarvis",
    "Jarvis Activate",
    "Jarvish Activate",
    "Hello Jarvis",
)
DEFAULT_TTS_VOICE_EN: Final[str] = "en-GB-RyanNeural"
DEFAULT_TTS_VOICE_HI: Final[str] = "hi-IN-MadhurNeural"
DEFAULT_TTS_RATE: Final[str] = "+0%"
DEFAULT_TTS_VOLUME: Final[str] = "+0%"
DEFAULT_TTS_PITCH: Final[str] = "+0Hz"

# Speech-to-Text (STT) Defaults
DEFAULT_STT_MODEL: Final[str] = "small"
DEFAULT_STT_DEVICE: Final[str] = "cpu"
DEFAULT_STT_COMPUTE_TYPE: Final[str] = "int8"

# Valid Supported Languages
VALID_LANGUAGES: Final[frozenset[str]] = frozenset({"en", "hi", "hinglish"})

# Logging Defaults & Valid Levels
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
VALID_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# Directory Names
DATA_DIR_NAME: Final[str] = "data"
MODELS_DIR_NAME: Final[str] = "models"
LOGS_DIR_NAME: Final[str] = "logs"
CONFIGS_DIR_NAME: Final[str] = "configs"
