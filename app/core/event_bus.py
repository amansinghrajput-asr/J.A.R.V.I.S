"""Event Bus Subsystem for J.A.R.V.I.S.

Provides a lightweight, thread-safe asynchronous and synchronous publish-subscribe
event architecture supporting event prioritization, wildcard topic subscriptions,
per-listener error isolation, and future-ready asynchronous event streaming.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Final, Optional, Union

from app.core.logger import get_logger

logger = get_logger("EVENT_BUS")


class EventPriority(IntEnum):
    """Execution priority levels for event listeners.

    Higher numeric values represent higher execution priority.
    """

    LOW = 10
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


@dataclass
class Event:
    """Represents a discrete event within the J.A.R.V.I.S ecosystem.

    Attributes:
        name: Unique topic or identifier for the event (e.g. 'WAKE_WORD_DETECTED').
        payload: Arbitrary contextual data associated with the event.
        timestamp: Unix epoch timestamp in seconds marking event creation.
        source: Component or subsystem originating the event (e.g. 'perception.wake_word').
    """

    name: str
    payload: Any = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and normalize event fields."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Event name must be a non-empty string.")
        self.name = self.name.strip()

        if self.payload is None:
            self.payload = {}

        if self.source is None or not str(self.source).strip():
            self.source = "system"
        else:
            self.source = str(self.source).strip()


@dataclass(frozen=True)
class _Subscription:
    """Internal container storing a listener registration."""

    priority: int
    order: int
    callback: Callable[..., Any]


class EventBus:
    """Thread-safe, priority-aware Event Bus for J.A.R.V.I.S.

    Coordinates decoupled communication between Perception, Cognition, Automation,
    Memory, and UI subsystems with robust error isolation and both synchronous
    and asynchronous publishing mechanisms.
    """

    def __init__(self) -> None:
        """Initialize an empty thread-safe EventBus."""
        self._lock = threading.RLock()
        self._listeners: dict[str, list[_Subscription]] = {}
        self._order_counter: int = 0

    def _validate_event_name(self, event_name: str) -> str:
        """Validate that an event name is a valid non-empty string.

        Args:
            event_name: The event identifier to validate.

        Returns:
            The sanitized event identifier.

        Raises:
            ValueError: If event_name is not a valid string.
        """
        if not isinstance(event_name, str) or not event_name.strip():
            raise ValueError("Event name must be a non-empty string.")
        return event_name.strip()

    def subscribe(
        self,
        event_name: str,
        callback: Optional[Callable[..., Any]] = None,
        priority: int = EventPriority.NORMAL,
    ) -> Any:
        """Register a listener for an event topic with an optional priority.

        Can be called directly with a callback or used as a function decorator:
        ```python
        # Direct subscription
        unsubscribe_fn = event_bus.subscribe('USER_INPUT', on_input, priority=EventPriority.HIGH)

        # Decorator syntax
        @event_bus.subscribe('USER_INPUT')
        def on_input(event: Event):
            ...
        ```

        Args:
            event_name: Target event identifier or '*' for all events.
            callback: Callable executed when the event is published.
            priority: Listener execution priority (higher runs first). Defaults to NORMAL (50).

        Returns:
            If called with callback: an unsubscription callable `() -> bool`.
            If called without callback: a decorator function.

        Raises:
            ValueError: If event_name is empty or not a string.
            TypeError: If callback is provided but not callable.
        """
        clean_name = self._validate_event_name(event_name)

        def _register(cb: Callable[..., Any]) -> Callable[[], bool]:
            if not callable(cb):
                raise TypeError(f"Event listener callback must be callable, got {type(cb).__name__}.")

            with self._lock:
                self._order_counter += 1
                subscription = _Subscription(
                    priority=int(priority),
                    order=self._order_counter,
                    callback=cb,
                )
                if clean_name not in self._listeners:
                    self._listeners[clean_name] = []

                self._listeners[clean_name].append(subscription)
                # Sort: highest priority first (-priority), then registration FIFO (order)
                self._listeners[clean_name].sort(key=lambda s: (-s.priority, s.order))

                logger.debug(
                    f"Subscribed '{getattr(cb, '__name__', repr(cb))}' to '{clean_name}' "
                    f"(priority={priority})"
                )

            def _unsubscribe() -> bool:
                return self.unsubscribe(clean_name, cb)

            return _unsubscribe

        if callback is None:
            # Decorator mode
            def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
                _register(fn)
                return fn

            return decorator

        return _register(callback)

    def unsubscribe(self, event_name: str, callback: Callable[..., Any]) -> bool:
        """Unregister a listener from an event topic.

        Args:
            event_name: The event identifier the listener was subscribed to.
            callback: The callback function to remove.

        Returns:
            True if the listener was found and removed, False otherwise.

        Raises:
            ValueError: If event_name is invalid.
        """
        clean_name = self._validate_event_name(event_name)

        with self._lock:
            if clean_name not in self._listeners:
                return False

            original_count = len(self._listeners[clean_name])
            self._listeners[clean_name] = [
                sub for sub in self._listeners[clean_name] if sub.callback != callback
            ]
            removed = len(self._listeners[clean_name]) < original_count

            if not self._listeners[clean_name]:
                del self._listeners[clean_name]

            if removed:
                logger.debug(
                    f"Unsubscribed '{getattr(callback, '__name__', repr(callback))}' "
                    f"from '{clean_name}'"
                )
            return removed

    def _normalize_event(
        self,
        event: Union[Event, str],
        payload: Optional[Any] = None,
        source: Optional[str] = None,
    ) -> Event:
        """Convert input arguments into a standardized Event instance."""
        if isinstance(event, Event):
            return event
        if isinstance(event, str):
            return Event(name=event, payload=payload, source=source)
        raise TypeError(
            f"Event must be an Event instance or event name string, got {type(event).__name__}."
        )

    def _get_target_listeners(self, event_name: str) -> list[_Subscription]:
        """Snapshot listeners matching the event name and wildcard subscriptions."""
        with self._lock:
            targets: list[_Subscription] = list(self._listeners.get(event_name, []))
            if "*" in self._listeners and event_name != "*":
                targets.extend(self._listeners["*"])
                # Re-sort combined list according to priority and FIFO
                targets.sort(key=lambda s: (-s.priority, s.order))
            return targets

    def _invoke_listener(self, callback: Callable[..., Any], event: Event) -> Any:
        """Invoke a listener callback, adapting to its signature."""
        try:
            sig = inspect.signature(callback)
            pos_params = [
                p
                for p in sig.parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            has_varargs = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
            )
            if len(pos_params) == 0 and not has_varargs:
                return callback()
            return callback(event)
        except (ValueError, TypeError):
            # Built-ins or callables without inspectable signatures
            return callback(event)

    def publish(
        self,
        event: Union[Event, str],
        payload: Optional[Any] = None,
        source: Optional[str] = None,
    ) -> Event:
        """Synchronously publish an event to all subscribed listeners.

        Listeners are executed in priority order. If any listener raises an
        exception, it is caught, logged, and isolated without preventing
        remaining listeners from processing the event.

        Args:
            event: Event instance or event name string to publish.
            payload: Optional payload if event name string is provided.
            source: Optional source identifier if event name string is provided.

        Returns:
            The Event instance that was published.
        """
        evt = self._normalize_event(event, payload=payload, source=source)
        targets = self._get_target_listeners(evt.name)

        logger.debug(f"Publishing event '{evt.name}' to {len(targets)} listener(s)")

        for sub in targets:
            try:
                res = self._invoke_listener(sub.callback, evt)
                # If a coroutine was returned in sync publish, schedule or run it safely
                if inspect.iscoroutine(res):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(res)
                    except RuntimeError:
                        asyncio.run(res)
            except Exception as exc:
                # Error isolation: log exception and continue with next listener
                cb_name = getattr(sub.callback, "__name__", repr(sub.callback))
                logger.exception(
                    f"Error in listener '{cb_name}' handling event '{evt.name}': {exc}"
                )

        return evt

    async def publish_async(
        self,
        event: Union[Event, str],
        payload: Optional[Any] = None,
        source: Optional[str] = None,
    ) -> Event:
        """Asynchronously publish an event to all subscribed listeners.

        Awaits asynchronous listener coroutines while safely invoking synchronous
        listeners. Guarantees per-listener error isolation.

        Args:
            event: Event instance or event name string to publish.
            payload: Optional payload if event name string is provided.
            source: Optional source identifier if event name string is provided.

        Returns:
            The Event instance that was published.
        """
        evt = self._normalize_event(event, payload=payload, source=source)
        targets = self._get_target_listeners(evt.name)

        logger.debug(f"Async publishing event '{evt.name}' to {len(targets)} listener(s)")

        for sub in targets:
            try:
                res = self._invoke_listener(sub.callback, evt)
                if inspect.iscoroutine(res):
                    await res
            except Exception as exc:
                # Error isolation
                cb_name = getattr(sub.callback, "__name__", repr(sub.callback))
                logger.exception(
                    f"Error in async listener '{cb_name}' handling event '{evt.name}': {exc}"
                )

        return evt

    def clear(self, event_name: Optional[str] = None) -> None:
        """Remove subscribed listeners.

        Args:
            event_name: If provided, clears listeners only for this specific event.
                If None, clears all listeners across all events.
        """
        with self._lock:
            if event_name is None:
                self._listeners.clear()
                logger.debug("Cleared all event listeners.")
            else:
                clean_name = self._validate_event_name(event_name)
                self._listeners.pop(clean_name, None)
                logger.debug(f"Cleared listeners for event '{clean_name}'.")

    def listener_count(self, event_name: Optional[str] = None) -> int:
        """Return the number of registered listeners.

        Args:
            event_name: If provided, returns listener count for this specific event.
                If None, returns total listener count across all events.

        Returns:
            The total number of registered listeners.
        """
        with self._lock:
            if event_name is None:
                return sum(len(subs) for subs in self._listeners.values())

            if not isinstance(event_name, str) or not event_name.strip():
                return 0
            clean_name = event_name.strip()
            return len(self._listeners.get(clean_name, []))

    def has_listeners(self, event_name: str) -> bool:
        """Check if any listeners are subscribed to the given event.

        Args:
            event_name: Unique event identifier.

        Returns:
            True if one or more listeners exist (including wildcard '*'), False otherwise.
        """
        if not isinstance(event_name, str) or not event_name.strip():
            return False
        clean_name = event_name.strip()
        with self._lock:
            count = len(self._listeners.get(clean_name, []))
            if clean_name != "*":
                count += len(self._listeners.get("*", []))
            return count > 0

    def registered_events(self) -> list[str]:
        """Return sorted list of all active event names with subscriptions.

        Returns:
            List of event name strings.
        """
        with self._lock:
            return sorted(self._listeners.keys())

    def __contains__(self, event_name: str) -> bool:
        """Support 'in' operator syntax: `event_name in event_bus`."""
        return self.has_listeners(event_name)

    def __len__(self) -> int:
        """Return total count of all registered listeners."""
        return self.listener_count()

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        with self._lock:
            return (
                f"<EventBus(events={len(self._listeners)}, "
                f"total_listeners={self.listener_count()})>"
            )


# Global default EventBus singleton
event_bus: Final[EventBus] = EventBus()


# Module-level convenience functions delegating to the default singleton:
def subscribe(
    event_name: str,
    callback: Optional[Callable[..., Any]] = None,
    priority: int = EventPriority.NORMAL,
) -> Any:
    """Subscribe a listener to an event using the default event_bus singleton.

    Args:
        event_name: Target event identifier or '*' for all events.
        callback: Optional callable executed when the event is published.
        priority: Listener execution priority (higher runs first). Defaults to NORMAL.

    Returns:
        Unsubscription callable if callback given, or decorator function.
    """
    return event_bus.subscribe(event_name, callback, priority=priority)


def unsubscribe(event_name: str, callback: Callable[..., Any]) -> bool:
    """Unsubscribe a listener from an event using the default event_bus singleton.

    Args:
        event_name: The event identifier.
        callback: The callback function to remove.

    Returns:
        True if removed, False otherwise.
    """
    return event_bus.unsubscribe(event_name, callback)


def publish(
    event: Union[Event, str],
    payload: Optional[Any] = None,
    source: Optional[str] = None,
) -> Event:
    """Synchronously publish an event using the default event_bus singleton.

    Args:
        event: Event instance or event name string to publish.
        payload: Optional payload if event name string is provided.
        source: Optional source identifier if event name string is provided.

    Returns:
        The published Event instance.
    """
    return event_bus.publish(event, payload=payload, source=source)


async def publish_async(
    event: Union[Event, str],
    payload: Optional[Any] = None,
    source: Optional[str] = None,
) -> Event:
    """Asynchronously publish an event using the default event_bus singleton.

    Args:
        event: Event instance or event name string to publish.
        payload: Optional payload if event name string is provided.
        source: Optional source identifier if event name string is provided.

    Returns:
        The published Event instance.
    """
    return await event_bus.publish_async(event, payload=payload, source=source)


def clear(event_name: Optional[str] = None) -> None:
    """Clear listeners on the default event_bus singleton.

    Args:
        event_name: If provided, clears listeners for this event.
            If None, clears all listeners.
    """
    event_bus.clear(event_name)


def listener_count(event_name: Optional[str] = None) -> int:
    """Return listener count on the default event_bus singleton.

    Args:
        event_name: If provided, count for this event. If None, total count.

    Returns:
        Total listener count.
    """
    return event_bus.listener_count(event_name)


__all__ = [
    "Event",
    "EventBus",
    "EventPriority",
    "clear",
    "event_bus",
    "listener_count",
    "publish",
    "publish_async",
    "subscribe",
    "unsubscribe",
]
