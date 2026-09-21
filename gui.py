"""J.A.R.V.I.S Desktop PySide6 GUI Entry Point.

Launches the live J.A.R.V.I.S backend subsystems together with the animated
PySide6 Sci-Fi Desktop HUD. Maintains strict boundaries:
- Qt owns the main GUI thread.
- Backend runs without blocking the Qt event loop (run() is NOT called).
- All UI actions flow through UIBridge -> PresentationAdapter -> CommandRouter.
- Clean shutdown and resource release on application exit.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Tuple

from PySide6.QtWidgets import QApplication

from app.application import JarvisApplication
from app.core.bootstrap import bootstrap
from app.core.presentation import PresentationAdapter, create_presentation_adapter
from app.ui.bridge import UIBridge
from app.ui.main_window import JarvisMainWindow

logger = logging.getLogger("GUI")


def parse_gui_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the GUI launcher."""
    parser = argparse.ArgumentParser(
        prog="jarvis-gui",
        description="J.A.R.V.I.S PySide6 Sci-Fi Desktop HUD",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        default=False,
        help="Launch J.A.R.V.I.S HUD in fullscreen mode",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        default=False,
        help="Suppress the startup terminal banner",
    )
    parsed, _ = parser.parse_known_args(args if args is not None else sys.argv[1:])
    return parsed


def setup_gui_components(
    args: Optional[list[str]] = None,
    *,
    app_instance: Optional[JarvisApplication] = None,
) -> Tuple[QApplication, JarvisMainWindow, UIBridge, JarvisApplication, PresentationAdapter]:
    """Initialize core backend and GUI components without entering the Qt event loop.

    Args:
        args: Optional command line arguments.
        app_instance: Optional pre-configured JarvisApplication instance.

    Returns:
        Tuple of (QApplication, JarvisMainWindow, UIBridge, JarvisApplication, PresentationAdapter).
    """
    cli_args = parse_gui_args(args)

    # 1. Bootstrap system configuration and logging
    bootstrap(display_banner=not cli_args.no_banner, exit_on_failure=False)

    # 2. Ensure QApplication is initialized on the main thread
    qapp = QApplication.instance()
    if qapp is None:
        qapp = QApplication(sys.argv)

    # 3. Instantiate live backend (without calling run())
    backend = app_instance or JarvisApplication(
        voice_mode=False,
        auto_discover_skills=True,
        print_ready=False,
    )

    # 4. Register shared system skill foundation in backend container
    from app.skills.system import register_system_foundation
    register_system_foundation(backend.container)
    p_bus = backend.container.resolve("planner_event_bus")

    # Wire shared planner_event_bus into Executor and Planner
    if hasattr(backend, "ai_manager") and backend.ai_manager is not None:
        if hasattr(backend.ai_manager, "executor") and backend.ai_manager.executor is not None:
            backend.ai_manager.executor.planner_event_bus = p_bus
        if hasattr(backend.ai_manager, "planner") and backend.ai_manager.planner is not None:
            backend.ai_manager.planner.event_bus = p_bus
    if backend.container.exists("executor"):
        exec_inst = backend.container.resolve("executor")
        if hasattr(exec_inst, "planner_event_bus"):
            exec_inst.planner_event_bus = p_bus
    if backend.container.exists("plan_executor"):
        exec_inst = backend.container.resolve("plan_executor")
        if hasattr(exec_inst, "planner_event_bus"):
            exec_inst.planner_event_bus = p_bus
    if backend.container.exists("planner"):
        p_inst = backend.container.resolve("planner")
        if hasattr(p_inst, "event_bus"):
            p_inst.event_bus = p_bus

    # 5. Resolve presentation adapter boundary from live container
    adapter = create_presentation_adapter(backend.container)

    # Ensure state manager is connected to the exact shared planner_event_bus
    sm = getattr(adapter, "_state_manager", None)
    if sm is not None:
        if getattr(sm, "_planner_event_bus", None) is not p_bus:
            sm._planner_event_bus = p_bus
            sm._subscribe_planner_event_bus()

    # 6. Create UI bridge connecting presentation adapter and conversation memory to Qt signals
    mem_mgr = None
    if backend.container.exists("memory_manager"):
        mem_mgr = backend.container.resolve("memory_manager")
    elif backend.container.exists("memory"):
        mem_mgr = backend.container.resolve("memory")

    bridge = UIBridge(presentation_adapter=adapter, memory_manager=mem_mgr)

    # 6. Assemble HUD window
    window = JarvisMainWindow(bridge=bridge)

    if getattr(cli_args, "fullscreen", False):
        window.showFullScreen()

    return qapp, window, bridge, backend, adapter


def shutdown_gui(
    bridge: Optional[UIBridge] = None,
    backend: Optional[JarvisApplication] = None,
    adapter: Optional[PresentationAdapter] = None,
) -> None:
    """Clean up GUI bridge and backend resources gracefully on application exit."""
    logger.info("Initiating J.A.R.V.I.S GUI shutdown...")

    # 1. Safely stop UIBridge polling timer and background executor
    if bridge is not None:
        try:
            bridge.close()
        except Exception as exc:
            logger.debug("Error closing UIBridge: %s", exc)

    # 2. Safely stop backend application services (e.g. wake word engine)
    if backend is not None:
        try:
            backend.stop()
        except Exception as exc:
            logger.debug("Error stopping JarvisApplication: %s", exc)

    # 3. Close state manager listeners if registered
    if adapter is not None and hasattr(adapter, "_state_manager"):
        sm = getattr(adapter, "_state_manager", None)
        if sm is not None and hasattr(sm, "close"):
            try:
                sm.close()
            except Exception as exc:
                logger.debug("Error closing AssistantStateManager: %s", exc)

    logger.info("J.A.R.V.I.S GUI shutdown complete.")


def main(args: Optional[list[str]] = None) -> int:
    """Main launcher entry point for the J.A.R.V.I.S desktop GUI."""
    qapp, window, bridge, backend, adapter = setup_gui_components(args)

    window.show()

    exit_code = 0
    try:
        exit_code = qapp.exec()
    finally:
        shutdown_gui(bridge=bridge, backend=backend, adapter=adapter)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
