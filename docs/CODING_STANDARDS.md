# J.A.R.V.I.S — Engineering & Coding Standards

## 1. Python 3.13 Language & Style Standards

All contributions to the J.A.R.V.I.S codebase must strictly adhere to the following language standards:

### 1.1 Strict Type Annotations
- Every function, method, parameter, and return value must be explicitly typed using standard Python typing (`typing` module or built-in generics).
- Avoid `Any` wherever possible; use `Union`, `Optional`, `TypeVar`, or `Protocol` for generic definitions.
- Example:
  ```python
  from typing import Optional, Dict, Any, List
  from dataclasses import dataclass

  @dataclass(frozen=True)
  class VoiceCommand:
      raw_text: str
      language: str
      confidence: float
      timestamp: float

  async def process_utterance(command: VoiceCommand) -> Optional[Dict[str, Any]]:
      ...
  ```

### 1.2 PEP 8 & Code Formatting
- Code formatting follows standard **Black** (line length: 100 characters) and **isort** for import grouping.
- Imports must be grouped in three distinct sections separated by a single blank line:
  1. Standard library imports (`os`, `sys`, `asyncio`, `logging`)
  2. Third-party library imports (`pyside6`, `google-genai`, `pyautogui`)
  3. Internal application imports (`from app.core.events import EventBus`)

---

## 2. Standard Project Directory Structure

The project follows a strict domain-driven, modular structure under `app/`:

```
J.A.R.V.I.S/
├── .env.example              # Sample environment variables template
├── README.md                 # Public overview and quickstart
├── main.py                   # Main application entry point
├── requirements.txt          # Production and development dependencies
├── app/
│   ├── __init__.py
│   ├── core/                 # Kernel, Event Bus, Config, State Management
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── events.py
│   │   ├── state.py
│   │   └── logger.py
│   ├── perception/           # Audio, Wake Word, STT, TTS, Vision Capture
│   │   ├── __init__.py
│   │   ├── wake_word.py
│   │   ├── stt.py
│   │   ├── tts.py
│   │   └── vision.py
│   ├── brain/                # Cognitive Engine, LLM Providers, Tools, Prompts
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── prompt_engine.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── registry.py
│   │   │   └── system_tools.py
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── gemini.py
│   │       └── openrouter.py
│   ├── automation/           # OS & Desktop Control
│   │   ├── __init__.py
│   │   ├── windows.py
│   │   ├── input.py
│   │   ├── system.py
│   │   └── files.py
│   ├── memory/               # Short-term and Long-term Memory
│   │   ├── __init__.py
│   │   ├── working.py
│   │   └── persistent.py
│   ├── ui/                   # PySide6 HUD, Widgets, Tray, Styles
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── hud_widget.py
│   │   ├── visualizer.py
│   │   ├── tray.py
│   │   └── styles.py
│   ├── plugins/              # Plugin Loader & Extensibility
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── loader.py
│   └── robotics/             # Hardware Abstraction Layer (HAL)
│       ├── __init__.py
│       ├── hal.py
│       └── serial_driver.py
├── configs/                  # System configs, prompts, and tool manifests
├── data/                     # Local SQLite DB, user data, cache
├── docs/                     # Project architecture & engineering documentation
├── logs/                     # Rotating execution logs
└── tests/                    # Pytest unit, integration, and E2E test suites
```

---

## 3. Asynchronous Programming & Concurrency Rules

1. **Never Block the Main UI Thread:**
   - Any operation involving network requests, file I/O, LLM generation, or audio processing must be asynchronous (`async def`) or executed in a background worker thread (`QThread` or `ThreadPoolExecutor`).
2. **Bridge Qt Signals and AsyncIO Gracefully:**
   - Use dedicated worker objects inheriting from `QObject` with Qt Signals (`pyqtSignal` / `Signal`) to communicate between background asyncio event loops and PySide6 UI widgets.
3. **Graceful Task Cancellation:**
   - All async tasks must support graceful cancellation via `asyncio.CancelledError` handling and proper resource cleanup.

---

## 4. Error Handling & Resilience Patterns

### 4.1 Custom Exception Hierarchy
All application exceptions inherit from a base `JarvisException`:

```python
class JarvisException(Exception):
    """Base exception for all J.A.R.V.I.S errors."""
    pass

class PerceptionError(JarvisException):
    """Raised when audio or vision capture fails."""
    pass

class LLMProviderError(JarvisException):
    """Raised when an LLM provider fails."""
    pass

class AutomationError(JarvisException):
    """Raised when OS or desktop automation fails."""
    pass
```

### 4.2 Error Handling Best Practices
- **No Bare Excepts:** Never use `except:` or `except Exception: pass`. Always log the exception with traceback information using `logger.exception()`.
- **Fail-Safe Fallbacks:** When primary cloud calls fail, gracefully fall back to alternative providers or local responses without halting the entire application.

---

## 5. Logging Standards

- Use the centralized logger initialized in `app.core.logger`.
- **Log Levels:**
  - `DEBUG`: Detailed diagnostic output (e.g., raw audio amplitude, token counts).
  - `INFO`: Normal runtime events (e.g., wake word triggered, window focused, response generated).
  - `WARNING`: Non-fatal issues (e.g., Gemini rate limit hit, falling back to OpenRouter).
  - `ERROR`: Recoverable failures (e.g., STT failure on audio frame, tool execution error).
  - `CRITICAL`: Unrecoverable errors requiring immediate shutdown.
- **Log Sanitation:** Ensure that API keys, passwords, and sensitive user secrets are NEVER logged in plain text.

---

## 6. Testing & Quality Standards

- **Unit Testing:** Powered by `pytest` and `pytest-asyncio`.
- **Mocking External APIs:** Cloud APIs (Gemini, OpenRouter) and hardware devices (Microphone, Serial) must be mocked in test suites to enable deterministic, zero-cost offline testing.
- **Test File Naming:** All test files must be located in `tests/` and prefixed with `test_` (e.g., `test_config.py`, `test_gemini_provider.py`).
