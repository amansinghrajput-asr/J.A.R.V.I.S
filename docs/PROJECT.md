# J.A.R.V.I.S — Project Specification & Mission Document

## 1. Executive Summary

**J.A.R.V.I.S (Just A Rather Very Intelligent System)** is a next-generation, open-source AI Desktop Operating System Assistant. Unlike conventional web-based chat interfaces or rudimentary command-line bots, J.A.R.V.I.S is engineered as an autonomous, persistent, system-level co-pilot that seamlessly integrates into the user's desktop environment.

J.A.R.V.I.S combines state-of-the-art multi-modal LLM reasoning, voice-activated perception, desktop and window automation, persistent memory, and an extensible plugin system to deliver a truly proactive assistant experience.

---

## 2. Core Vision & Philosophy

### 2.1 An AI Operating System Assistant, Not a Chatbot
Traditional chatbots operate in a passive, text-in/text-out paradigm within isolated sandboxes. J.A.R.V.I.S breaks out of the chatbox:
- **System-Aware:** Aware of active windows, system telemetry, clipboard contents, screen state, and running processes.
- **Action-Oriented:** Translates high-level natural language intent into concrete, deterministic OS actions (file manipulation, window orchestration, application management, browser navigation).
- **Proactive & Ambient:** Capable of background monitoring, autonomous alerts, and hands-free voice interaction via wake-word detection.

### 2.2 Bilingual Fluency (Hindi & English)
J.A.R.V.I.S is natively designed for dual-language operation:
- Full conversational fluency in **English**, **Hindi (हिंदी)**, and **Hinglish** (code-switched Hindi-English).
- Language detection and auto-switching across STT (Speech-to-Text), LLM reasoning, and TTS (Text-to-Speech) pipelines.
- Culturally and linguistically tailored personality prompts and voice responses.

### 2.3 Strict Zero-Cost & Open-Source Policy
- **No Paid APIs:** Designed from the ground up to operate completely free of ongoing subscription costs.
- **Primary Cloud Brain:** Google Gemini (leveraging free tier capabilities and generous rate limits).
- **Fallback Cloud Brain:** OpenRouter Free Tier models (Llama-3, Mistral, Gemma, etc.) with automatic circuit-breaking.
- **Local On-Device Engine:** OpenWakeWord (wake-word), Whisper / faster-whisper (STT), and Edge-TTS (TTS) ensuring high-performance audio processing with zero recurring fees.

---

## 3. High-Level Capabilities & Feature Scope

| Capability Domain | Description | Technology Anchor |
| :--- | :--- | :--- |
| **Voice Activation** | Ultra-low latency, continuous wake-word listening ("Hey Jarvis" / "Jarvis") | `openwakeword` |
| **Speech-to-Text** | Offline/Local transcription supporting Hindi, English, and mixed speech | `openai-whisper` / `faster-whisper` |
| **Speech Synthesis** | Ultra-natural, neural voice synthesis with multilingual personas | `edge-tts` |
| **Cognitive Core** | Advanced multi-step reasoning, intent parsing, tool selection, and synthesis | Google Gemini 2.0 / 1.5 & OpenRouter Free |
| **Desktop Automation** | Native mouse, keyboard, window positioning, focus management, and key combos | `pyautogui`, `pywin32` |
| **System Telemetry** | CPU, RAM, disk, battery, process management, and health monitoring | `psutil` |
| **Graphical Interface** | Sci-fi HUD inspired, modern, non-intrusive floating GUI and system tray widget | `PySide6` (Qt 6) |
| **Memory Engine** | Multi-tier context buffer (working context + persistent vector/sqlite memory) | `sqlite3` + local embeddings |
| **Vision & Screen** | Multi-modal screen capture analysis, optical inspection, and window context | Gemini Vision / Screen capture |
| **Plugin Framework** | Dynamic module loader for user-created scripts, tools, and external APIs | Native Python Plugin Manager |
| **Hardware / Robotics** | Hardware abstraction layer (HAL) for future microcontrollers & IoT integration | `pyserial`, MQTT, WebSockets |

---

## 4. Architectural Tenets

1. **Modularity & Decoupling:** Every major subsystem (Audio, Brain, Automation, GUI, Memory) operates as an independent module communicating through a centralized asynchronous event bus and typed interfaces.
2. **Resilience & Graceful Degradation:** Network outages, API rate limits, or audio device disconnects must never crash the core kernel. The system smoothly switches between primary and fallback providers.
3. **Non-Blocking Execution:** The GUI and audio capture pipelines run strictly decoupled from long-running LLM queries or automation tasks using `asyncio` and Qt `QThread` workers.
4. **Safety & Permission Boundaries:** High-risk actions (file deletion, shell execution, credential handling) adhere to user-configurable confirmation policies.

---

## 5. Security & Privacy Framework

- **Local Secret Storage:** All API keys and environment configurations are stored strictly in a `.env` file on the local machine and never transmitted externally except to authorized API endpoints over TLS.
- **Microphone & Vision Privacy:** Audio listening buffers are processed in memory and discarded; wake-word models run completely locally. Screen capture is strictly triggered on-demand by user intent.
- **Safe Automation Execution:** Automation scripts validate screen boundaries and window handles before firing input events to prevent unintentional clicks or keystrokes.
