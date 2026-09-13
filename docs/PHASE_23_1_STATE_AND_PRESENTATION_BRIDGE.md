# Phase 23.1 — Unified Assistant State Machine & Presentation Adapter

## Overview

Phase 23.1 introduces a headless, Qt-free, strongly typed, thread-safe presentation architecture providing a reliable backend boundary for the upcoming Phase 23.2 PySide6 desktop HUD.

### Key Architectural Boundaries
1. **Core State Models (`app/core/state.py`)**:
   - `AssistantState`: 10 discrete states (`INITIALIZING`, `IDLE`, `LISTENING`, `TRANSCRIBING`, `THINKING`, `PLANNING`, `EXECUTING`, `SPEAKING`, `AWAITING_CONFIRMATION`, `ERROR`).
   - `PendingConfirmation`: Frozen, deeply immutable confirmation token carrying only presentation-safe primitives (`confirmation_id`, `operation`, `target`, `risk_level`, `description`, `created_at`).
   - `AssistantSnapshot`: Point-in-time, deeply immutable snapshot (`@dataclass(frozen=True)` with immutable tuple transcript and clamped metrics) safe to pass across threads.
   - `PresentationEvent`: UI-agnostic presentation event carrying state transitions or telemetry.

2. **Assistant State Manager (`app/core/state_manager.py`)**:
   - Central authority aggregating system lifecycle events from `EventBus` and `PlannerEventBus`.
   - Thread-safe mutation via `threading.RLock`.
   - Deterministic transition matrix (`ALLOWED_TRANSITIONS`) rejecting invalid transitions safely without crashing backend callers.
   - Safe error recovery (`recover_to_idle()`).
   - Bounded snapshot history (`collections.deque(maxlen=100)`).
   - Subscriber exception isolation protecting backend event emitters.
   - Correlation preservation (`plan_id`, `execution_id`, `task_id`) and safe division task progress calculation (`completed_tasks / total_tasks`).

3. **Presentation Adapter & Bounded Queue (`app/core/presentation.py`)**:
   - `PresentationQueue`: Thread-safe bounded event queue (`maxsize=1000`) with explicit drop-oldest overflow policy. Non-blocking enqueue and non-blocking draining (`drain(max_items=50)`).
   - `PresentationAdapter`: Headless presentation facade connecting backend services to presentation clients.
   - Reuses existing asynchronous execution pipelines without creating competing thread pools:
     - Text commands: `CommandRouter.route_async()`
     - Voice interactions: `VoiceConversationEngine.listen_once_async()`
     - Approvals: `SystemConfirmationManager.resolve_confirmation()` (zero security bypass)
     - Cancellations: `ExecutionController.cancel()`
   - Dependency Injection factory: `create_presentation_adapter()`.

4. **Microphone RMS Telemetry (`app/voice/microphone.py`)**:
   - Optional, backward-compatible `amplitude_callback: Optional[Callable[[float], None]] = None`.
   - Computes normalized RMS amplitude (0.0 - 1.0) on captured PCM frames.
   - Callback failures are isolated with warning/debug logging and never disrupt audio recording.

5. **Future Voice Architecture Requirement**:
   - TTS/voice selection must remain replaceable/extensible.
   - Future voices should be addable through a configurable voice profile / provider abstraction without rewriting J.A.R.V.I.S core logic.

---

## Event Mappings

| Source Bus | Source Event / Class | Assistant State Transition | Context Preserved |
| :--- | :--- | :--- | :--- |
| `EventBus` | `application.ready` | `IDLE` | Status: "System ready" |
| `EventBus` | `voice_engine.recording` | `LISTENING` | Recording duration |
| `EventBus` | `voice_engine.transcribing` | `TRANSCRIBING` | Status: "Transcribing speech..." |
| `EventBus` | `voice_engine.routing` | `THINKING` | Command text, transcript append |
| `EventBus` | `voice_engine.speaking` | `SPEAKING` | Response text, transcript append |
| `EventBus` | `voice_engine.completed` | `IDLE` | Status: "Ready" |
| `EventBus` | `voice_engine.failed` | `ERROR` | Error details |
| `EventBus` | `command.received` | `THINKING` | Command text, transcript append |
| `EventBus` | `command.completed` | `IDLE` | Response text |
| `EventBus` | `command.failed` | `ERROR` | Error details |
| `PlannerEventBus` | `PlanStarted` | `PLANNING` | `plan_id`, `execution_id`, total tasks |
| `PlannerEventBus` | `TaskStarted` | `EXECUTING` | `task_id`, action, target |
| `PlannerEventBus` | `TaskCompleted` | `EXECUTING` (internal update) | Increment `completed_tasks`, update progress |
| `PlannerEventBus` | `SystemSkillStarted` | `EXECUTING` | Operation, target |
| `PlannerEventBus` | `SystemSkillConfirmationRequired` | `AWAITING_CONFIRMATION` | `confirmation_id`, operation, risk level |
| `PlannerEventBus` | `PlanCompleted` | `IDLE` | Retains completion progress, clears plan |
| `PlannerEventBus` | `PlanCancelled` | `IDLE` | Reason recorded |
| `PlannerEventBus` | `PlanFailed` | `ERROR` | Error details |
| `PlannerEventBus` | `SystemSkillFailed` | `ERROR` | Error details |

---

## Verification & Test Results

- State Model Tests (`tests/test_assistant_state.py`): 5 passed
- State Manager Tests (`tests/test_state_manager.py`): 8 passed
- Presentation Adapter Tests (`tests/test_presentation_adapter.py`): 14 passed
- Total Phase 23.1 Focused Suite: **27 passed**
- Full Suite Regression: **1,048 passed, 1 skipped** (Baseline was 1,021 passed, 1 skipped; +27 net tests).

---

## Rollback Approach

If rollback is required:
1. Delete `app/core/state.py`
2. Delete `app/core/state_manager.py`
3. Delete `app/core/presentation.py`
4. Revert `app/voice/microphone.py`
5. Delete test files `tests/test_assistant_state.py`, `tests/test_state_manager.py`, `tests/test_presentation_adapter.py`
6. Delete documentation `docs/PHASE_23_1_STATE_AND_PRESENTATION_BRIDGE.md`
