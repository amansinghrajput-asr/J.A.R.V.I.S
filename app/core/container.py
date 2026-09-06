"""Service Container for J.A.R.V.I.S.

Provides a lightweight, thread-safe dependency injection and service registry
subsystem supporting singleton and factory service lifecycles, circular dependency
detection, and strict registration safeguards.
"""

from __future__ import annotations

import inspect
import threading
from typing import Any, Callable, Final, Optional, TypeVar

from app.core.logger import get_logger

logger = get_logger("CONTAINER")

T = TypeVar("T")


class JarvisException(Exception):
    """Base exception for all J.A.R.V.I.S errors."""


class ContainerError(JarvisException):
    """Base exception for all service container errors."""


class ServiceNotFoundError(ContainerError):
    """Raised when a requested service cannot be found in the container."""


class ServiceAlreadyRegisteredError(ContainerError):
    """Raised when attempting to register a service name that already exists."""


class ServiceResolutionError(ContainerError):
    """Raised when a service factory fails during resolution."""


class CircularDependencyError(ServiceResolutionError):
    """Raised when a circular dependency is detected during resolution."""


class InvalidServiceError(ContainerError):
    """Raised when invalid service parameters are provided."""


class ServiceContainer:
    """Thread-safe dependency injection and service registry container.

    Manages the lifecycle and resolution of singleton instances and factory
    providers across the J.A.R.V.I.S application architecture.
    """

    def __init__(self) -> None:
        """Initialize an empty thread-safe ServiceContainer."""
        self._lock = threading.RLock()
        self._singletons: dict[str, Any] = {}
        self._factories: dict[str, Callable[..., Any]] = {}
        self._local = threading.local()

    @property
    def _resolution_stack(self) -> list[str]:
        """Thread-local resolution stack used for cycle detection."""
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def _validate_name(self, name: str) -> str:
        """Validate that a service name is a non-empty string.

        Args:
            name: The service name to validate.

        Returns:
            The stripped, valid service name.

        Raises:
            InvalidServiceError: If name is not a valid non-empty string.
        """
        if not isinstance(name, str):
            raise InvalidServiceError(
                f"Service name must be a string, got {type(name).__name__}: {name!r}"
            )
        clean_name = name.strip()
        if not clean_name:
            raise InvalidServiceError("Service name must not be empty or whitespace only.")
        return clean_name

    def register_singleton(
        self,
        name: str,
        instance: Any,
        *,
        allow_override: bool = False,
    ) -> None:
        """Register an existing object as a singleton service.

        Args:
            name: Unique identifier for the service.
            instance: The pre-instantiated service object.
            allow_override: If True, allows replacing an existing registration.
                Defaults to False.

        Raises:
            InvalidServiceError: If the service name is invalid.
            ServiceAlreadyRegisteredError: If the service name already exists
                and allow_override is False.
        """
        clean_name = self._validate_name(name)

        with self._lock:
            if not allow_override and (
                clean_name in self._singletons or clean_name in self._factories
            ):
                raise ServiceAlreadyRegisteredError(
                    f"Service '{clean_name}' is already registered in the container."
                )

            # Clear any conflicting factory registration if overriding
            self._factories.pop(clean_name, None)
            self._singletons[clean_name] = instance
            logger.debug(f"Registered singleton service: '{clean_name}'")

    def register_factory(
        self,
        name: str,
        factory: Callable[..., Any],
        *,
        allow_override: bool = False,
    ) -> None:
        """Register a factory callable that creates a service on resolution.

        Args:
            name: Unique identifier for the service.
            factory: Callable producing a new service instance upon each invocation.
            allow_override: If True, allows replacing an existing registration.
                Defaults to False.

        Raises:
            InvalidServiceError: If the name is invalid or factory is not callable.
            ServiceAlreadyRegisteredError: If the service name already exists
                and allow_override is False.
        """
        clean_name = self._validate_name(name)

        if not callable(factory):
            raise InvalidServiceError(
                f"Factory for service '{clean_name}' must be callable, got {type(factory).__name__}."
            )

        with self._lock:
            if not allow_override and (
                clean_name in self._singletons or clean_name in self._factories
            ):
                raise ServiceAlreadyRegisteredError(
                    f"Service '{clean_name}' is already registered in the container."
                )

            # Clear any conflicting singleton registration if overriding
            self._singletons.pop(clean_name, None)
            self._factories[clean_name] = factory
            logger.debug(f"Registered factory service: '{clean_name}'")

    def resolve(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Resolve and return a registered service by name.

        If the service was registered as a singleton, the identical instance
        is returned. If registered as a factory, the factory callable is invoked
        to produce and return an instance.

        Args:
            name: Unique identifier for the service.
            *args: Optional positional arguments forwarded to the factory callable.
            **kwargs: Optional keyword arguments forwarded to the factory callable.

        Returns:
            The resolved service instance.

        Raises:
            InvalidServiceError: If the service name is invalid.
            ServiceNotFoundError: If no service is registered under the given name.
            CircularDependencyError: If a resolution cycle is detected.
            ServiceResolutionError: If factory execution raises an exception.
        """
        clean_name = self._validate_name(name)

        with self._lock:
            if clean_name in self._singletons:
                return self._singletons[clean_name]

            if clean_name not in self._factories:
                raise ServiceNotFoundError(
                    f"Service '{clean_name}' is not registered in the container."
                )

            # Factory resolution with circular dependency detection
            if clean_name in self._resolution_stack:
                cycle = " -> ".join(self._resolution_stack + [clean_name])
                raise CircularDependencyError(
                    f"Circular dependency detected while resolving service '{clean_name}': {cycle}"
                )

            factory = self._factories[clean_name]
            self._resolution_stack.append(clean_name)
            try:
                if args or kwargs:
                    return factory(*args, **kwargs)

                # Introspect factory signature if no explicit args provided
                try:
                    sig = inspect.signature(factory)
                    required_params = [
                        p
                        for p in sig.parameters.values()
                        if p.default == inspect.Parameter.empty
                        and p.kind
                        in (
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                    ]
                    if len(required_params) == 1:
                        return factory(self)
                except (ValueError, TypeError):
                    pass

                return factory()
            except (
                ServiceNotFoundError,
                ServiceAlreadyRegisteredError,
                CircularDependencyError,
                InvalidServiceError,
            ):
                raise
            except Exception as exc:
                raise ServiceResolutionError(
                    f"Failed to resolve factory service '{clean_name}': {exc}"
                ) from exc
            finally:
                self._resolution_stack.pop()

    def exists(self, name: str) -> bool:
        """Check whether a service is registered in the container.

        Args:
            name: Unique identifier for the service.

        Returns:
            True if registered (as singleton or factory), False otherwise.
        """
        if not isinstance(name, str) or not name.strip():
            return False
        clean_name = name.strip()
        with self._lock:
            return clean_name in self._singletons or clean_name in self._factories

    def clear(self) -> None:
        """Remove all registered singleton and factory services."""
        with self._lock:
            self._singletons.clear()
            self._factories.clear()
            if hasattr(self._local, "stack"):
                self._local.stack.clear()
            logger.debug("Service container cleared.")

    def unregister(self, name: str) -> bool:
        """Unregister a service by name.

        Args:
            name: Unique identifier for the service.

        Returns:
            True if the service was found and removed, False otherwise.
        """
        if not isinstance(name, str) or not name.strip():
            return False
        clean_name = name.strip()
        with self._lock:
            removed_singleton = self._singletons.pop(clean_name, None) is not None
            removed_factory = self._factories.pop(clean_name, None) is not None
            return removed_singleton or removed_factory

    def registered_services(self) -> list[str]:
        """Return a sorted list of all registered service names.

        Returns:
            Alphabetically sorted list of registered service identifiers.
        """
        with self._lock:
            names = set(self._singletons.keys()) | set(self._factories.keys())
            return sorted(names)

    def is_singleton(self, name: str) -> bool:
        """Check if a registered service is a singleton.

        Args:
            name: Unique identifier for the service.

        Returns:
            True if registered as singleton, False otherwise.
        """
        if not isinstance(name, str) or not name.strip():
            return False
        clean_name = name.strip()
        with self._lock:
            return clean_name in self._singletons

    def is_factory(self, name: str) -> bool:
        """Check if a registered service is a factory.

        Args:
            name: Unique identifier for the service.

        Returns:
            True if registered as factory, False otherwise.
        """
        if not isinstance(name, str) or not name.strip():
            return False
        clean_name = name.strip()
        with self._lock:
            return clean_name in self._factories

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator syntax: `name in container`."""
        return self.exists(name)

    def __getitem__(self, name: str) -> Any:
        """Support dictionary-style access syntax: `container['service']`."""
        return self.resolve(name)

    def __len__(self) -> int:
        """Return the total number of registered services."""
        with self._lock:
            return len(self._singletons) + len(self._factories)

    def __repr__(self) -> str:
        """Return developer-friendly string representation of the container."""
        with self._lock:
            return (
                f"<ServiceContainer(singletons={len(self._singletons)}, "
                f"factories={len(self._factories)})>"
            )


# Type alias for convenience
Container = ServiceContainer

# Global default ServiceContainer instance
container: Final[ServiceContainer] = ServiceContainer()


# Module-level convenience functions delegating to the default container instance:
def register_singleton(
    name: str,
    instance: Any,
    *,
    allow_override: bool = False,
) -> None:
    """Register an existing object as a singleton in the global container.

    Args:
        name: Unique identifier for the service.
        instance: The pre-instantiated service object.
        allow_override: If True, allows replacing an existing registration.
            Defaults to False.

    Raises:
        InvalidServiceError: If the service name is invalid.
        ServiceAlreadyRegisteredError: If the service name already exists
            and allow_override is False.
    """
    container.register_singleton(name, instance, allow_override=allow_override)


def register_factory(
    name: str,
    factory: Callable[..., Any],
    *,
    allow_override: bool = False,
) -> None:
    """Register a factory callable in the global container.

    Args:
        name: Unique identifier for the service.
        factory: Callable producing a new service instance upon each invocation.
        allow_override: If True, allows replacing an existing registration.
            Defaults to False.

    Raises:
        InvalidServiceError: If the name is invalid or factory is not callable.
        ServiceAlreadyRegisteredError: If the service name already exists
            and allow_override is False.
    """
    container.register_factory(name, factory, allow_override=allow_override)


def resolve(name: str, *args: Any, **kwargs: Any) -> Any:
    """Resolve and return a registered service by name from the global container.

    Args:
        name: Unique identifier for the service.
        *args: Optional positional arguments forwarded to the factory callable.
        **kwargs: Optional keyword arguments forwarded to the factory callable.

    Returns:
        The resolved service instance.

    Raises:
        InvalidServiceError: If the service name is invalid.
        ServiceNotFoundError: If no service is registered under the given name.
        CircularDependencyError: If a resolution cycle is detected.
        ServiceResolutionError: If factory execution raises an exception.
    """
    return container.resolve(name, *args, **kwargs)


def exists(name: str) -> bool:
    """Check whether a service is registered in the global container.

    Args:
        name: Unique identifier for the service.

    Returns:
        True if registered (as singleton or factory), False otherwise.
    """
    return container.exists(name)


def clear() -> None:
    """Remove all registered services from the global container."""
    container.clear()


__all__ = [
    "CircularDependencyError",
    "Container",
    "ContainerError",
    "InvalidServiceError",
    "JarvisException",
    "ServiceAlreadyRegisteredError",
    "ServiceContainer",
    "ServiceNotFoundError",
    "ServiceResolutionError",
    "clear",
    "container",
    "exists",
    "register_factory",
    "register_singleton",
    "resolve",
]
