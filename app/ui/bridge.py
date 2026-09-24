"""Presentation Bridge for PySide6 Desktop HUD in J.A.R.V.I.S. Phase 23.2.

Provides the decoupled QObject boundary connecting PresentationAdapter events
and snapshots to Qt signals on the GUI main thread. Never blocks the Qt event
loop and never accesses raw AI providers or planner internals directly.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import datetime
import logging
import os
import time
from typing import Any, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from app.core.presentation import PresentationAdapter
from app.core.state import AssistantSnapshot, AssistantState, PresentationEvent
from app.memory.persistence import ConversationHistoryPersistence

logger = logging.getLogger("app.ui.bridge")

DEFAULT_POLL_INTERVAL_MS = 33  # ~30 Hz polling rate for smooth telemetry and state updates


class UIBridge(QObject):
    """Qt-native Presentation Bridge delivering AssistantSnapshots and telemetry to GUI.

    Signals:
        snapshot_updated(object): Emits latest AssistantSnapshot on change.
        state_changed(str): Emits new AssistantState value string on transition.
        amplitude_updated(float): Emits normalized microphone amplitude (0.0 - 1.0).
        event_dispatched(object): Emits each discrete PresentationEvent drained from backend.
        command_completed(object): Emits result of asynchronously submitted command.
        command_failed(str): Emits error string if command execution fails.
        telemetry_updated(dict): Emits live host CPU, RAM, GPU, network, and uptime metrics.
        plan_updated(object): Emits current active planner tasks and progress.
        ai_telemetry_updated(dict): Emits live AI provider, model, latency, and health.
        execution_history_updated(list): Emits updated execution history records.
        system_diagnostics_updated(dict): Emits comprehensive read-only system diagnostics.
        conversation_history_loaded(list): Emits loaded recent conversation dialogue records.
        cognitive_stage_changed(str, str): Emits active cognitive stage ('STANDBY', 'ANALYZING', 'ROUTING', 'EXECUTING', 'SYNTHESIZING') and detail.
    """

    snapshot_updated = Signal(object)
    state_changed = Signal(str)
    amplitude_updated = Signal(float)
    event_dispatched = Signal(object)
    command_completed = Signal(object)
    command_failed = Signal(str)
    telemetry_updated = Signal(dict)
    plan_updated = Signal(object)
    ai_telemetry_updated = Signal(dict)
    execution_history_updated = Signal(list)
    system_diagnostics_updated = Signal(dict)
    conversation_history_loaded = Signal(list)
    cognitive_stage_changed = Signal(str, str)

    def __init__(
        self,
        presentation_adapter: PresentationAdapter,
        *,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        memory_manager: Optional[Any] = None,
        persistence: Optional[ConversationHistoryPersistence] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        """Initialize UIBridge with PresentationAdapter and start event polling.

        Args:
            presentation_adapter: Authoritative Phase 23.1 presentation boundary.
            poll_interval_ms: Timer tick rate in milliseconds for draining backend events.
            memory_manager: Optional MemoryManager instance for conversation tracking.
            persistence: Optional ConversationHistoryPersistence instance.
            parent: Optional Qt parent QObject.
        """
        super().__init__(parent)
        self._adapter = presentation_adapter
        self._poll_interval_ms = max(10, poll_interval_ms)

        # Thread pool strictly for dispatching async coroutine submissions from Qt main thread
        self._async_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="UIBridgeAsync")

        # Conversation persistence & memory manager resolution
        self._persistence = persistence if persistence is not None else ConversationHistoryPersistence()
        self._memory_manager: Optional[Any] = memory_manager
        if self._memory_manager is None:
            try:
                from app.core.container import container
                if container.exists("memory_manager"):
                    self._memory_manager = container.resolve("memory_manager")
                elif container.exists("memory"):
                    self._memory_manager = container.resolve("memory")
            except Exception as exc:
                logger.debug("Could not resolve MemoryManager in UIBridge: %s", exc)

        if self._memory_manager is not None:
            try:
                self._persistence.bind_to_manager(self._memory_manager)
            except Exception as exc:
                logger.debug("Could not bind persistence in UIBridge: %s", exc)

        self._current_cognitive_stage: str = "STANDBY"

        # Cache last seen values to avoid redundant signal emissions
        self._last_state: Optional[AssistantState] = None
        self._last_snapshot: Optional[AssistantSnapshot] = None
        self._last_amplitude: float = -1.0

        # Voice concurrency guard
        self._voice_active: bool = False

        # Telemetry sampling state
        self._telemetry_sampling: bool = False
        self._last_net_bytes: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._gpu_cached: dict[str, Any] = {"available": False, "percent": None}
        self._last_gpu_check: float = 0.0
        self._os_info_cached: dict[str, Any] = {}

        # Planner task tracking
        self._active_plan_info: dict[str, Any] = {
            "title": "",
            "progress": 0.0,
            "steps": [],
        }

        # High-frequency event polling timer running safely on Qt main thread
        self._timer = QTimer(self)
        self._timer.setInterval(self._poll_interval_ms)
        self._timer.timeout.connect(self._poll_backend_events)
        self._timer.start()

        # Telemetry sampling timer (~2s cadence)
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.setInterval(2000)
        self._telemetry_timer.timeout.connect(self._trigger_telemetry_sample)
        self._telemetry_timer.start()

        # Execution history tracking
        self._execution_history: list[dict[str, Any]] = []

        # AI Core telemetry cache
        self._ai_telemetry_cached: dict[str, Any] = {
            "active_provider": "gemini",
            "active_model": "gemini-2.5-flash",
            "mode": "CLOUD",
            "reasoning_latency_ms": None,
            "health": "ONLINE",
        }

        # Initial snapshot capture & non-blocking initializations
        self._sync_current_snapshot()
        self._trigger_telemetry_sample()
        self.fetch_ai_telemetry()
        self.fetch_system_diagnostics()
        self.fetch_conversation_history()

    @property
    def adapter(self) -> PresentationAdapter:
        """Return attached PresentationAdapter."""
        return self._adapter

    @property
    def current_snapshot(self) -> AssistantSnapshot:
        """Return most recently fetched AssistantSnapshot."""
        if self._last_snapshot is None:
            self._sync_current_snapshot()
        return self._last_snapshot

    def _sync_current_snapshot(self) -> None:
        """Fetch current snapshot and emit signals if changed."""
        snap = self._adapter.get_snapshot()
        self._process_snapshot(snap)

    def _process_snapshot(self, snap: AssistantSnapshot) -> None:
        """Check state and amplitude diffs and emit corresponding Qt signals."""
        self._last_snapshot = snap

        # State change detection
        if snap.state != self._last_state:
            self._last_state = snap.state
            self.state_changed.emit(snap.state.value)

            # Map operational state transitions to cognitive stage
            if snap.state == AssistantState.IDLE:
                self.set_cognitive_stage("STANDBY", "Ready")
            elif snap.state == AssistantState.SPEAKING:
                self.set_cognitive_stage("SYNTHESIZING", "Speaking")
            elif snap.state == AssistantState.THINKING and self._current_cognitive_stage == "STANDBY":
                self.set_cognitive_stage("ANALYZING", "Processing")
            elif snap.state in (AssistantState.PLANNING, AssistantState.EXECUTING) and self._current_cognitive_stage in ("STANDBY", "ANALYZING"):
                self.set_cognitive_stage("EXECUTING", "Running")
            elif snap.state == AssistantState.ERROR:
                self.set_cognitive_stage("ERROR", "Error")

        # Amplitude telemetry change detection
        if abs(snap.mic_amplitude - self._last_amplitude) > 0.005:
            self._last_amplitude = snap.mic_amplitude
            self.amplitude_updated.emit(snap.mic_amplitude)

        self.snapshot_updated.emit(snap)

    def _poll_backend_events(self) -> None:
        """Timer callback draining bounded presentation queue non-blockingly."""
        try:
            events = self._adapter.drain_events(max_items=50)
            for event in events:
                self.event_dispatched.emit(event)
                # Process planner lifecycle events to update dynamic tasks
                self._process_planner_event(event)
                # Event carries a point-in-time snapshot
                if event.snapshot:
                    self._process_snapshot(event.snapshot)
        except Exception as exc:
            logger.warning("Error draining events in UIBridge: %s", exc)

    def _process_planner_event(self, event: PresentationEvent) -> None:
        """Extract planner lifecycle notifications and maintain active plan state."""
        etype = getattr(event, "event_type", "")
        payload_dict = dict(getattr(event, "payload", ()))

        if etype in ("planner.plan_started", "PlanStarted"):
            title = str(payload_dict.get("query") or event.snapshot.current_command or "Executing Plan")
            count = int(payload_dict.get("task_count", 0))
            self._active_plan_info = {
                "title": title,
                "progress": 0.0,
                "steps": [],
            }
            self.plan_updated.emit(self._active_plan_info)
            self.set_cognitive_stage("ROUTING", "Planning tasks")

        elif etype in ("planner.task_started", "TaskStarted"):
            task_id = str(payload_dict.get("task_id", ""))
            action = str(payload_dict.get("action", "") or "Task")
            steps: list[dict[str, Any]] = self._active_plan_info.setdefault("steps", [])

            # Check if task already exists
            existing = next((s for s in steps if s.get("task_id") == task_id), None)
            if existing:
                existing["status"] = "running"
                existing["is_active"] = True
                existing["is_done"] = False
            else:
                steps.append({
                    "task_id": task_id,
                    "title": action,
                    "status": "running",
                    "is_active": True,
                    "is_done": False,
                    "is_failed": False,
                })
            self.plan_updated.emit(self._active_plan_info)
            self.set_cognitive_stage("EXECUTING", action[:18])

        elif etype in ("planner.task_completed", "TaskCompleted"):
            task_id = str(payload_dict.get("task_id", ""))
            prog = float(payload_dict.get("progress", self._active_plan_info.get("progress", 0.0)))
            self._active_plan_info["progress"] = prog

            steps = self._active_plan_info.get("steps", [])
            existing = next((s for s in steps if s.get("task_id") == task_id), None)
            action_title = existing["title"] if existing else (str(payload_dict.get("action")) or "Task")
            if existing:
                existing["status"] = "completed"
                existing["is_active"] = False
                existing["is_done"] = True
            self.plan_updated.emit(self._active_plan_info)
            self.set_cognitive_stage("SYNTHESIZING", "Task complete")

            # Record in execution history
            self.add_execution_record({
                "task_id": task_id,
                "action": action_title,
                "target": payload_dict.get("target"),
                "status": "completed",
                "duration": float(payload_dict.get("duration", 0.0)),
                "timestamp": float(payload_dict.get("timestamp", time.time())),
            })

        elif etype in ("planner.task_failed", "TaskFailed"):
            task_id = str(payload_dict.get("task_id", ""))
            steps = self._active_plan_info.get("steps", [])
            existing = next((s for s in steps if s.get("task_id") == task_id), None)
            action_title = existing["title"] if existing else (str(payload_dict.get("action")) or "Task")
            if existing:
                existing["status"] = "failed"
                existing["is_active"] = False
                existing["is_failed"] = True
            self.plan_updated.emit(self._active_plan_info)
            self.set_cognitive_stage("ERROR", "Task failed")

            # Record in execution history
            self.add_execution_record({
                "task_id": task_id,
                "action": action_title,
                "target": payload_dict.get("target"),
                "status": "failed",
                "error": str(payload_dict.get("error") or "Task failed"),
                "duration": float(payload_dict.get("duration", 0.0)),
                "timestamp": float(payload_dict.get("timestamp", time.time())),
            })

        elif etype in ("planner.plan_completed", "PlanCompleted"):
            self._active_plan_info["progress"] = 1.0
            for s in self._active_plan_info.get("steps", []):
                s["status"] = "completed"
                s["is_active"] = False
                s["is_done"] = True
            self.plan_updated.emit(self._active_plan_info)

        elif etype in ("planner.plan_failed", "PlanFailed"):
            for s in self._active_plan_info.get("steps", []):
                if s.get("is_active"):
                    s["status"] = "failed"
                    s["is_active"] = False
                    s["is_failed"] = True
            self.plan_updated.emit(self._active_plan_info)

    def update_plan_state(
        self,
        title: str,
        progress: float = 0.0,
        steps: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Manually push plan updates to connected GUI listeners."""
        self._active_plan_info = {
            "title": title,
            "progress": max(0.0, min(1.0, float(progress))),
            "steps": list(steps or []),
        }
        self.plan_updated.emit(self._active_plan_info)

    # --------------------------------------------------------------------------
    # Live System Telemetry Background Sampler
    # --------------------------------------------------------------------------

    def _trigger_telemetry_sample(self) -> None:
        """Trigger non-blocking background telemetry sampling on worker thread pool."""
        if self._telemetry_sampling:
            return

        self._telemetry_sampling = True
        self._async_executor.submit(self._telemetry_worker)

    def _telemetry_worker(self) -> None:
        """Collect host system metrics safely without blocking the Qt event loop."""
        try:
            import platform
            import time

            now = time.time()

            # 1. CPU & RAM (using psutil when available)
            cpu_pct = 0.0
            ram_pct = 0.0
            try:
                import psutil
                cpu_pct = float(psutil.cpu_percent(interval=None))
                ram_pct = float(psutil.virtual_memory().percent)
            except Exception:
                pass

            # 2. Network throughput (bytes diff / delta time)
            net_summary = "↑ 0.0 KB/s  ↓ 0.0 KB/s"
            try:
                import psutil
                net_io = psutil.net_io_counters()
                sent = float(net_io.bytes_sent)
                recv = float(net_io.bytes_recv)
                last_sent, last_recv, last_time = self._last_net_bytes
                if last_time > 0 and now > last_time:
                    dt = now - last_time
                    up_kb = max(0.0, (sent - last_sent) / (1024.0 * dt))
                    down_kb = max(0.0, (recv - last_recv) / (1024.0 * dt))
                    if up_kb >= 1024.0 or down_kb >= 1024.0:
                        net_summary = f"↑ {up_kb / 1024.0:.1f} MB/s  ↓ {down_kb / 1024.0:.1f} MB/s"
                    else:
                        net_summary = f"↑ {up_kb:.1f} KB/s  ↓ {down_kb:.1f} KB/s"
                self._last_net_bytes = (sent, recv, now)
            except Exception:
                pass

            # 3. GPU check (throttled every 8 seconds to avoid overhead)
            if (now - self._last_gpu_check) > 8.0:
                self._last_gpu_check = now
                try:
                    import shutil
                    import subprocess
                    smi = shutil.which("nvidia-smi")
                    if smi:
                        res = subprocess.run(
                            [smi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True,
                            text=True,
                            timeout=1.5,
                            check=False,
                        )
                        if res.returncode == 0 and res.stdout.strip():
                            gpu_val = float(res.stdout.strip().splitlines()[0])
                            self._gpu_cached = {"available": True, "percent": gpu_val}
                        else:
                            self._gpu_cached = {"available": False, "percent": None}
                    else:
                        self._gpu_cached = {"available": False, "percent": None}
                except Exception:
                    self._gpu_cached = {"available": False, "percent": None}

            # 4. System Uptime
            uptime_str = "0d 0h 0m"
            try:
                import psutil
                boot_time = psutil.boot_time()
                uptime_secs = int(max(0.0, now - boot_time))
                days = uptime_secs // 86400
                hours = (uptime_secs % 86400) // 3600
                mins = (uptime_secs % 3600) // 60
                uptime_str = f"{days}d {hours}h {mins}m"
            except Exception:
                pass

            # 5. Cached OS and Device Hostname
            if not self._os_info_cached:
                import socket
                os_name = f"{platform.system()} {platform.release()}".strip() or "Windows"
                host_name = socket.gethostname() or platform.node() or "Localhost"
                self._os_info_cached = {
                    "os_name": os_name,
                    "device_name": host_name,
                }

            metrics = {
                "cpu_percent": cpu_pct,
                "memory_percent": ram_pct,
                "gpu_available": self._gpu_cached.get("available", False),
                "gpu_percent": self._gpu_cached.get("percent"),
                "network_summary": net_summary,
                "uptime": uptime_str,
                "os_name": self._os_info_cached.get("os_name", "Windows"),
                "device_name": self._os_info_cached.get("device_name", "Localhost"),
                "system_status": "● All systems normal",
            }

            self.telemetry_updated.emit(metrics)
        except Exception as exc:
            logger.debug("Telemetry sampling error: %s", exc)
        finally:
            self._telemetry_sampling = False

    # --------------------------------------------------------------------------
    # User Actions (Non-blocking Delegation to PresentationAdapter)
    # --------------------------------------------------------------------------

    def add_execution_record(self, record: dict[str, Any]) -> None:
        """Add execution record to history and emit update signal."""
        self._execution_history.append(record)
        # Cap in-memory history to last 100 records
        if len(self._execution_history) > 100:
            self._execution_history = self._execution_history[-100:]
        self.execution_history_updated.emit(self._execution_history)

    def get_execution_history(self) -> list[dict[str, Any]]:
        """Return a copy of recent execution history."""
        return list(self._execution_history)

    def fetch_ai_telemetry(self) -> None:
        """Query active AI provider and model telemetry safely on worker thread."""
        def _worker() -> None:
            try:
                from app.ai.provider_router import provider_router
                from app.core.config import settings

                active_p = provider_router.active_provider_name
                registered = provider_router.get_registered_providers()
                ai_cfg = getattr(settings, "ai", None)

                if active_p == "ollama":
                    model = getattr(ai_cfg, "ollama_model", None) or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
                    mode = "LOCAL"
                else:
                    model = getattr(ai_cfg, "gemini_model", None) or getattr(ai_cfg, "model", None) or "gemini-2.5-flash"
                    mode = "CLOUD"

                telemetry = {
                    "active_provider": active_p,
                    "available_providers": registered,
                    "active_model": model,
                    "mode": mode,
                    "reasoning_latency_ms": self._ai_telemetry_cached.get("reasoning_latency_ms"),
                    "health": "ONLINE",
                }
                self._ai_telemetry_cached.update(telemetry)
                self.ai_telemetry_updated.emit(telemetry)
            except Exception as exc:
                logger.warning("Error fetching AI telemetry: %s", exc)

        self._async_executor.submit(_worker)

    def set_ai_provider(self, provider_name: str) -> None:
        """Switch active AI provider via existing ProviderRouter asynchronously without blocking Qt."""
        clean = (provider_name or "").strip().lower()
        if not clean:
            return

        def _worker() -> None:
            try:
                from app.ai.provider_router import provider_router
                provider_router.set_active_provider(clean)
                self.fetch_ai_telemetry()
            except Exception as exc:
                logger.error("Failed to set active AI provider to '%s': %s", clean, exc)

        self._async_executor.submit(_worker)

    def fetch_system_diagnostics(self) -> None:
        """Fetch comprehensive system diagnostics asynchronously and emit signal."""
        def _worker() -> None:
            try:
                from app.automation.system import system_monitor
                import platform

                # Gather via safe SystemMonitor APIs
                cpu = system_monitor.get_cpu_metrics()
                mem = system_monitor.get_memory_metrics()
                disk = system_monitor.get_disk_metrics()
                battery = system_monitor.get_battery_metrics()
                top_procs = system_monitor.get_top_processes(limit=8, sort_by="cpu")

                net_rate = self._last_net_bytes[2] if len(self._last_net_bytes) >= 3 else 0.0

                data = {
                    "cpu_percent": cpu.percent,
                    "cpu_cores": f"{cpu.physical_cores}P / {cpu.logical_cores}L",
                    "cpu_freq_mhz": cpu.frequency_mhz,
                    "memory_percent": mem.percent,
                    "ram_used_gb": mem.used_gb,
                    "ram_total_gb": mem.total_gb,
                    "ram_free_gb": mem.free_gb,
                    "disk_percent": disk.percent,
                    "disk_used_gb": disk.used_gb,
                    "disk_total_gb": disk.total_gb,
                    "disk_free_gb": disk.free_gb,
                    "disk_mount": disk.mount_point,
                    "gpu_available": self._gpu_cached.get("available", False),
                    "gpu_percent": self._gpu_cached.get("percent"),
                    "network_rate_kbs": net_rate / 1024.0,
                    "network_sent_mb": self._last_net_bytes[0] / (1024.0 * 1024.0) if self._last_net_bytes[0] > 0 else 0.0,
                    "network_recv_mb": self._last_net_bytes[1] / (1024.0 * 1024.0) if self._last_net_bytes[1] > 0 else 0.0,
                    "battery_percent": battery.percent if battery else None,
                    "battery_plugged": battery.power_plugged if battery else True,
                    "os_platform": self._os_info_cached.get("platform", f"{platform.system()} {platform.release()}"),
                    "device_name": self._os_info_cached.get("device_name", "Localhost"),
                    "uptime": self._os_info_cached.get("uptime", "N/A"),
                    "top_processes": [
                        {
                            "pid": p.pid,
                            "name": p.name,
                            "cpu_percent": p.cpu_percent,
                            "memory_percent": p.memory_percent,
                            "status": p.status,
                        }
                        for p in top_procs
                    ],
                }
                self.system_diagnostics_updated.emit(data)
            except Exception as exc:
                logger.warning("Error fetching system diagnostics: %s", exc)

        self._async_executor.submit(_worker)

    def set_cognitive_stage(self, stage_name: str, detail: Optional[str] = None) -> None:
        """Update active cognitive stage and emit signal."""
        clean_stage = (stage_name or "STANDBY").upper()
        self._current_cognitive_stage = clean_stage
        self.cognitive_stage_changed.emit(clean_stage, str(detail or ""))

    def _record_memory(
        self,
        content: str,
        role: str,
        *,
        source: str = "text",
        is_error: bool = False,
    ) -> None:
        """Record conversation turn safely into MemoryManager and persistence without blocking Qt."""
        clean_content = str(content or "").strip()
        if not clean_content:
            return

        def _worker() -> None:
            try:
                meta = {"is_error": True} if is_error else {}
                tags = ["error"] if is_error else []

                # Add to MemoryManager if available
                if self._memory_manager is not None and hasattr(self._memory_manager, "add"):
                    self._memory_manager.add(
                        content=clean_content,
                        role=role,
                        source=source,
                        metadata=meta,
                        tags=tags,
                    )
                else:
                    # Fallback directly to persistence
                    user_c = clean_content if role == "user" else ""
                    asst_c = clean_content if role != "user" else None
                    self._persistence.append_turn(
                        user_content=user_c,
                        assistant_content=asst_c,
                        source=source,
                        assistant_metadata=meta if role != "user" else None,
                    )
            except Exception as exc:
                logger.debug("Error recording conversation turn to memory: %s", exc)

        self._async_executor.submit(_worker)

    def fetch_conversation_history(self, limit: int = 30) -> None:
        """Asynchronously load recent conversation dialogue history without blocking Qt."""
        bounded_limit = max(1, min(100, int(limit)))

        def _worker() -> None:
            records: list[dict[str, Any]] = []
            try:
                memories = []
                if self._memory_manager is not None and hasattr(self._memory_manager, "get_recent"):
                    memories = self._memory_manager.get_recent(limit=bounded_limit)

                # Fallback to persistence if memory_manager had no records yet
                if not memories:
                    all_persisted = self._persistence.load_history()
                    memories = all_persisted[-bounded_limit:] if len(all_persisted) > bounded_limit else all_persisted

                for mem in memories:
                    role = getattr(mem, "role", "user")
                    sender = "You" if role == "user" else "J.A.R.V.I.S"
                    ts = getattr(mem, "timestamp", None)
                    ts_str = None
                    if ts is not None:
                        try:
                            if isinstance(ts, (int, float)):
                                ts_str = datetime.datetime.fromtimestamp(ts).strftime("%I:%M %p")
                            else:
                                ts_str = str(ts)
                        except Exception:
                            ts_str = None

                    tags = getattr(mem, "tags", []) or []
                    metadata = getattr(mem, "metadata", {}) or {}
                    is_err = "error" in tags or bool(metadata.get("is_error", False))

                    records.append({
                        "sender": sender,
                        "text": getattr(mem, "content", ""),
                        "timestamp": ts_str,
                        "is_error": is_err,
                    })

                self.conversation_history_loaded.emit(records)
            except Exception as exc:
                logger.warning("Failed to fetch conversation history in UIBridge: %s", exc)
                self.conversation_history_loaded.emit([])

        self._async_executor.submit(_worker)

    def submit_command(self, text: str) -> None:
        """Submit a text command asynchronously without blocking the Qt event loop."""
        if not text or not text.strip():
            return

        clean_text = text.strip()
        self.set_cognitive_stage("ANALYZING", "Parsing command")
        self._record_memory(content=clean_text, role="user", source="text")

        def _worker() -> None:
            t0 = time.time()
            loop = None
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.set_cognitive_stage("ROUTING", "Routing command")
                result = loop.run_until_complete(self._adapter.submit_command(clean_text))
                elapsed = time.time() - t0

                if result is None:
                    res_str = "Operation completed successfully."
                elif hasattr(result, "to_user_message") and callable(result.to_user_message):
                    res_str = result.to_user_message()
                elif hasattr(result, "message") and isinstance(result.message, str):
                    res_str = result.message
                else:
                    res_str = str(result)
                self.set_cognitive_stage("SYNTHESIZING", "Formulating response")
                self._record_memory(content=res_str, role="assistant", source="text")

                self.command_completed.emit(result)
                self.add_execution_record({
                    "task_id": f"cmd-{int(time.time() * 1000)}",
                    "action": clean_text,
                    "target": "Direct Command",
                    "status": "completed",
                    "duration": round(elapsed, 2),
                    "timestamp": time.time(),
                })
                # Graceful transition back to standby
                self.set_cognitive_stage("STANDBY", "Ready")
            except Exception as exc:
                elapsed = time.time() - t0
                logger.error("Command failed via UIBridge: %s", exc)
                self.set_cognitive_stage("ERROR", "Command failed")
                self._record_memory(content=f"Error: {exc}", role="assistant", source="text", is_error=True)
                self.command_failed.emit(str(exc))
                self.add_execution_record({
                    "task_id": f"cmd-{int(time.time() * 1000)}",
                    "action": clean_text,
                    "target": "Direct Command",
                    "status": "failed",
                    "error": str(exc),
                    "duration": round(elapsed, 2),
                    "timestamp": time.time(),
                })
                self.set_cognitive_stage("STANDBY", "Error")
            finally:
                if loop is not None:
                    try:
                        loop.close()
                    except Exception as loop_exc:
                        logger.debug("Error closing worker loop in submit_command: %s", loop_exc)
                try:
                    asyncio.set_event_loop(None)
                except Exception:
                    pass

        self._async_executor.submit(_worker)

    def interrupt_speech(self) -> bool:
        """Interrupt active audio playback and speech response."""
        try:
            return self._adapter.interrupt_speech()
        except Exception as exc:
            logger.warning("Error interrupting speech via UIBridge: %s", exc)
            return False

    def start_voice_interaction(self, duration: Optional[float] = None) -> bool:
        """Start a voice interaction cycle asynchronously without blocking Qt.

        Includes concurrency guard against duplicate triggers, and acts as an
        interrupt if the assistant is currently speaking.
        """
        if self._last_state == "SPEAKING":
            logger.info("Interrupting active speech response via voice trigger.")
            self.interrupt_speech()
            return True

        if self._voice_active:
            logger.warning("Voice interaction already in progress. Ignoring duplicate trigger.")
            return False

        self._voice_active = True

        def _worker() -> None:
            loop = None
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                res = loop.run_until_complete(self._adapter.start_voice_interaction(duration=duration))
                self.command_completed.emit(res)
            except Exception as exc:
                logger.error("Voice interaction failed via UIBridge: %s", exc)
                self.command_failed.emit(str(exc))
            finally:
                self._voice_active = False
                if loop is not None:
                    try:
                        loop.close()
                    except Exception as loop_exc:
                        logger.debug("Error closing worker loop in start_voice_interaction: %s", loop_exc)
                try:
                    asyncio.set_event_loop(None)
                except Exception:
                    pass

        self._async_executor.submit(_worker)
        return True

    def resolve_confirmation(
        self,
        confirmation_id: str,
        approved: bool,
        *,
        decided_by: str = "gui_operator",
        reason: str = "",
    ) -> bool:
        """Resolve a pending confirmation synchronously through PresentationAdapter."""
        return self._adapter.resolve_confirmation(
            confirmation_id,
            approved=approved,
            decided_by=decided_by,
            reason=reason,
        )

    def cancel_current_task(self, reason: Optional[str] = None) -> bool:
        """Cancel current task via PresentationAdapter."""
        return self._adapter.cancel_current_task(reason=reason or "Cancelled by user via UI")

    def close(self) -> None:
        """Stop polling timer, telemetry timer, and shutdown background executor."""
        if self._timer.isActive():
            self._timer.stop()
        if self._telemetry_timer.isActive():
            self._telemetry_timer.stop()
        self._async_executor.shutdown(wait=False)
