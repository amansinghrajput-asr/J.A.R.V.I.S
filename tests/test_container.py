"""Unit tests for the J.A.R.V.I.S Service Container Subsystem."""

from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from app.core.container import (
    CircularDependencyError,
    Container,
    ContainerError,
    InvalidServiceError,
    JarvisException,
    ServiceAlreadyRegisteredError,
    ServiceContainer,
    ServiceNotFoundError,
    ServiceResolutionError,
    clear,
    container,
    exists,
    register_factory,
    register_singleton,
    resolve,
)


class DummyDatabase:
    """Mock database service for container tests."""

    def __init__(self, uri: str = "sqlite:///:memory:") -> None:
        self.uri = uri
        self.connected = True


class DummyClient:
    """Mock client service depending on DummyDatabase."""

    def __init__(self, db: DummyDatabase, client_id: str = "default") -> None:
        self.db = db
        self.client_id = client_id


class TestServiceContainer(unittest.TestCase):
    """Comprehensive test suite for ServiceContainer lifecycle and operations."""

    def setUp(self) -> None:
        """Create a fresh isolated container instance for each test."""
        self.container = ServiceContainer()
        # Also ensure global container is cleared
        clear()

    def tearDown(self) -> None:
        """Clean up state after each test."""
        self.container.clear()
        clear()

    def test_register_and_resolve_singleton(self) -> None:
        """Verify registering and resolving a singleton returns the exact same instance."""
        db = DummyDatabase("sqlite:///data/test.db")
        self.container.register_singleton("db", db)

        self.assertTrue(self.container.exists("db"))
        self.assertTrue(self.container.is_singleton("db"))
        self.assertFalse(self.container.is_factory("db"))

        resolved_1 = self.container.resolve("db")
        resolved_2 = self.container.resolve("db")

        self.assertIs(resolved_1, db)
        self.assertIs(resolved_2, db)
        self.assertIs(resolved_1, resolved_2)

    def test_register_and_resolve_factory(self) -> None:
        """Verify factory returns a newly produced instance on each resolve call."""
        counter = {"count": 0}

        def create_service() -> dict[str, int]:
            counter["count"] += 1
            return {"instance_id": counter["count"]}

        self.container.register_factory("service", create_service)

        self.assertTrue(self.container.exists("service"))
        self.assertFalse(self.container.is_singleton("service"))
        self.assertTrue(self.container.is_factory("service"))

        instance_1 = self.container.resolve("service")
        instance_2 = self.container.resolve("service")

        self.assertEqual(instance_1["instance_id"], 1)
        self.assertEqual(instance_2["instance_id"], 2)
        self.assertIsNot(instance_1, instance_2)

    def test_factory_container_injection(self) -> None:
        """Verify factory callable can accept the container parameter for DI."""
        db = DummyDatabase()
        self.container.register_singleton("db", db)

        def client_factory(c: ServiceContainer) -> DummyClient:
            resolved_db = c.resolve("db")
            return DummyClient(db=resolved_db, client_id="injected")

        self.container.register_factory("client", client_factory)

        client = self.container.resolve("client")
        self.assertIsInstance(client, DummyClient)
        self.assertIs(client.db, db)
        self.assertEqual(client.client_id, "injected")

    def test_resolve_forwarded_arguments(self) -> None:
        """Verify positional and keyword arguments are forwarded to factory."""

        def configurable_factory(prefix: str, multiplier: int = 1) -> str:
            return prefix * multiplier

        self.container.register_factory("calc", configurable_factory)

        result_1 = self.container.resolve("calc", "JARVIS-", 3)
        self.assertEqual(result_1, "JARVIS-JARVIS-JARVIS-")

        result_2 = self.container.resolve("calc", "AI-", multiplier=2)
        self.assertEqual(result_2, "AI-AI-")

    def test_prevent_duplicate_registration_singleton_over_singleton(self) -> None:
        """Verify duplicate singleton registration raises ServiceAlreadyRegisteredError."""
        self.container.register_singleton("service", "first")

        with self.assertRaises(ServiceAlreadyRegisteredError) as ctx:
            self.container.register_singleton("service", "second")

        self.assertIn("already registered", str(ctx.exception))
        self.assertEqual(self.container.resolve("service"), "first")

    def test_prevent_duplicate_registration_factory_over_factory(self) -> None:
        """Verify duplicate factory registration raises ServiceAlreadyRegisteredError."""
        self.container.register_factory("service", lambda: "first")

        with self.assertRaises(ServiceAlreadyRegisteredError):
            self.container.register_factory("service", lambda: "second")

        self.assertEqual(self.container.resolve("service"), "first")

    def test_prevent_duplicate_registration_factory_over_singleton(self) -> None:
        """Verify factory registration over existing singleton raises error."""
        self.container.register_singleton("service", "first")

        with self.assertRaises(ServiceAlreadyRegisteredError):
            self.container.register_factory("service", lambda: "second")

        self.assertEqual(self.container.resolve("service"), "first")

    def test_prevent_duplicate_registration_singleton_over_factory(self) -> None:
        """Verify singleton registration over existing factory raises error."""
        self.container.register_factory("service", lambda: "first")

        with self.assertRaises(ServiceAlreadyRegisteredError):
            self.container.register_singleton("service", "second")

        self.assertEqual(self.container.resolve("service"), "first")

    def test_allow_override_flag(self) -> None:
        """Verify allow_override=True replaces existing singleton or factory."""
        self.container.register_singleton("service", "first")
        self.container.register_singleton("service", "second", allow_override=True)
        self.assertEqual(self.container.resolve("service"), "second")

        # Override singleton with factory
        self.container.register_factory(
            "service", lambda: "from_factory", allow_override=True
        )
        self.assertTrue(self.container.is_factory("service"))
        self.assertFalse(self.container.is_singleton("service"))
        self.assertEqual(self.container.resolve("service"), "from_factory")

        # Override factory with singleton
        self.container.register_singleton("service", "final_singleton", allow_override=True)
        self.assertTrue(self.container.is_singleton("service"))
        self.assertEqual(self.container.resolve("service"), "final_singleton")

    def test_resolve_unregistered_service_raises(self) -> None:
        """Verify resolving an unregistered service raises ServiceNotFoundError."""
        with self.assertRaises(ServiceNotFoundError) as ctx:
            self.container.resolve("unknown_service")

        self.assertIn("unknown_service", str(ctx.exception))
        self.assertTrue(issubclass(ServiceNotFoundError, ContainerError))
        self.assertTrue(issubclass(ServiceNotFoundError, JarvisException))

    def test_invalid_service_names(self) -> None:
        """Verify empty, whitespace, or non-string names raise InvalidServiceError."""
        for invalid_name in ["", "   ", None, 123, [], {}]:  # type: ignore[list-item]
            with self.assertRaises(InvalidServiceError):
                self.container.register_singleton(invalid_name, "val")  # type: ignore[arg-type]

            with self.assertRaises(InvalidServiceError):
                self.container.register_factory(invalid_name, lambda: "val")  # type: ignore[arg-type]

            with self.assertRaises(InvalidServiceError):
                self.container.resolve(invalid_name)  # type: ignore[arg-type]

    def test_invalid_factory_callable(self) -> None:
        """Verify registering a non-callable factory raises InvalidServiceError."""
        for non_callable in ["not_a_function", 123, None, object()]:
            with self.assertRaises(InvalidServiceError):
                self.container.register_factory("bad_factory", non_callable)  # type: ignore[arg-type]

    def test_factory_execution_error_wrapped(self) -> None:
        """Verify exceptions in factory are wrapped in ServiceResolutionError with cause."""

        def failing_factory() -> Any:
            raise ValueError("Database connection failed")

        self.container.register_factory("faulty", failing_factory)

        with self.assertRaises(ServiceResolutionError) as ctx:
            self.container.resolve("faulty")

        self.assertIn("Database connection failed", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, ValueError)

    def test_circular_dependency_detection_direct(self) -> None:
        """Verify direct circular dependency (A -> B -> A) is detected and raises."""
        self.container.register_factory(
            "service_a", lambda c: c.resolve("service_b")
        )
        self.container.register_factory(
            "service_b", lambda c: c.resolve("service_a")
        )

        with self.assertRaises(CircularDependencyError) as ctx:
            self.container.resolve("service_a")

        self.assertIn("Circular dependency detected", str(ctx.exception))
        self.assertIn("service_a -> service_b -> service_a", str(ctx.exception))

    def test_circular_dependency_detection_self(self) -> None:
        """Verify self-referencing factory (A -> A) raises CircularDependencyError."""
        self.container.register_factory(
            "self_ref", lambda c: c.resolve("self_ref")
        )

        with self.assertRaises(CircularDependencyError) as ctx:
            self.container.resolve("self_ref")

        self.assertIn("self_ref -> self_ref", str(ctx.exception))

    def test_circular_dependency_detection_chain(self) -> None:
        """Verify 3-step circular chain (A -> B -> C -> A) is properly detected."""
        self.container.register_factory("a", lambda c: c.resolve("b"))
        self.container.register_factory("b", lambda c: c.resolve("c"))
        self.container.register_factory("c", lambda c: c.resolve("a"))

        with self.assertRaises(CircularDependencyError) as ctx:
            self.container.resolve("a")

        self.assertIn("a -> b -> c -> a", str(ctx.exception))

        # Ensure container remains usable after cycle detection unwinds
        self.container.register_singleton("d", "valid_service")
        self.assertEqual(self.container.resolve("d"), "valid_service")

    def test_clear_removes_all_registrations(self) -> None:
        """Verify clear() wipes out all singletons and factories."""
        self.container.register_singleton("singleton_1", "val1")
        self.container.register_factory("factory_1", lambda: "val2")

        self.assertEqual(len(self.container), 2)
        self.assertTrue(self.container.exists("singleton_1"))
        self.assertTrue(self.container.exists("factory_1"))

        self.container.clear()

        self.assertEqual(len(self.container), 0)
        self.assertFalse(self.container.exists("singleton_1"))
        self.assertFalse(self.container.exists("factory_1"))

        with self.assertRaises(ServiceNotFoundError):
            self.container.resolve("singleton_1")

    def test_unregister(self) -> None:
        """Verify unregister() removes specific services."""
        self.container.register_singleton("s1", "v1")
        self.container.register_factory("f1", lambda: "v2")

        self.assertTrue(self.container.unregister("s1"))
        self.assertFalse(self.container.exists("s1"))
        self.assertFalse(self.container.unregister("s1"))  # Already removed

        self.assertTrue(self.container.unregister("f1"))
        self.assertFalse(self.container.exists("f1"))

        self.assertFalse(self.container.unregister("non_existent"))
        self.assertFalse(self.container.unregister(""))

    def test_registered_services_list(self) -> None:
        """Verify registered_services() returns sorted unique names."""
        self.container.register_singleton("bravo", 1)
        self.container.register_factory("alpha", lambda: 2)
        self.container.register_singleton("charlie", 3)

        self.assertEqual(
            self.container.registered_services(),
            ["alpha", "bravo", "charlie"],
        )

    def test_container_dunder_methods(self) -> None:
        """Verify container __contains__, __getitem__, __len__, and __repr__."""
        self.container.register_singleton("config", {"env": "test"})
        self.container.register_factory("token", lambda: "secret")

        # __contains__
        self.assertTrue("config" in self.container)
        self.assertTrue("token" in self.container)
        self.assertFalse("missing" in self.container)

        # __getitem__
        self.assertEqual(self.container["config"], {"env": "test"})
        self.assertEqual(self.container["token"], "secret")

        # __len__
        self.assertEqual(len(self.container), 2)

        # __repr__
        repr_str = repr(self.container)
        self.assertIn("ServiceContainer", repr_str)
        self.assertIn("singletons=1", repr_str)
        self.assertIn("factories=1", repr_str)

    def test_alias_container(self) -> None:
        """Verify Container is an alias to ServiceContainer."""
        self.assertIs(Container, ServiceContainer)
        alias_instance = Container()
        self.assertIsInstance(alias_instance, ServiceContainer)


class TestGlobalContainerFunctions(unittest.TestCase):
    """Test suite for module-level functions operating on global container."""

    def setUp(self) -> None:
        """Ensure global container is empty before each test."""
        clear()

    def tearDown(self) -> None:
        """Clean up global container after each test."""
        clear()

    def test_global_register_and_resolve_singleton(self) -> None:
        """Verify register_singleton, resolve, and exists on module-level."""
        register_singleton("global_db", "db_instance")

        self.assertTrue(exists("global_db"))
        self.assertEqual(resolve("global_db"), "db_instance")

    def test_global_register_and_resolve_factory(self) -> None:
        """Verify register_factory on module-level."""
        register_factory("counter_svc", lambda: object())

        self.assertTrue(exists("counter_svc"))
        inst1 = resolve("counter_svc")
        inst2 = resolve("counter_svc")
        self.assertIsNot(inst1, inst2)

    def test_global_clear(self) -> None:
        """Verify clear() empties the global container."""
        register_singleton("temp", 123)
        self.assertTrue(exists("temp"))

        clear()
        self.assertFalse(exists("temp"))

    def test_global_container_instance(self) -> None:
        """Verify exported container is an instance of ServiceContainer."""
        self.assertIsInstance(container, ServiceContainer)


class TestContainerThreadSafety(unittest.TestCase):
    """Stress tests verifying thread-safety of ServiceContainer."""

    def setUp(self) -> None:
        """Prepare fresh container."""
        self.container = ServiceContainer()

    def tearDown(self) -> None:
        """Clean up container."""
        self.container.clear()

    def test_concurrent_singleton_and_factory_registrations(self) -> None:
        """Verify concurrent registrations and resolutions across multiple threads."""
        num_threads = 20
        services_per_thread = 50

        def worker(thread_idx: int) -> list[str]:
            registered: list[str] = []
            for i in range(services_per_thread):
                s_name = f"thread_{thread_idx}_singleton_{i}"
                f_name = f"thread_{thread_idx}_factory_{i}"

                self.container.register_singleton(s_name, f"val_{thread_idx}_{i}")
                self.container.register_factory(
                    f_name, lambda idx=thread_idx, num=i: f"factory_{idx}_{num}"
                )

                registered.append(s_name)
                registered.append(f_name)

                # Concurrently resolve immediately
                resolved_s = self.container.resolve(s_name)
                resolved_f = self.container.resolve(f_name)
                assert resolved_s == f"val_{thread_idx}_{i}"
                assert resolved_f == f"factory_{thread_idx}_{i}"

            return registered

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, t) for t in range(num_threads)]
            for future in as_completed(futures):
                result = future.result()
                self.assertEqual(len(result), services_per_thread * 2)

        expected_total = num_threads * services_per_thread * 2
        self.assertEqual(len(self.container), expected_total)

    def test_concurrent_resolution_of_same_services(self) -> None:
        """Verify many threads reading the same singletons and factories concurrently."""
        self.container.register_singleton("shared_singleton", {"status": "ok"})
        self.container.register_factory("shared_factory", lambda: DummyDatabase())

        num_threads = 24
        iterations = 100

        def reader_worker() -> bool:
            for _ in range(iterations):
                s = self.container.resolve("shared_singleton")
                assert s["status"] == "ok"

                f = self.container.resolve("shared_factory")
                assert isinstance(f, DummyDatabase)

                assert self.container.exists("shared_singleton")
                assert self.container.exists("shared_factory")
            return True

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(reader_worker) for _ in range(num_threads)]
            for future in as_completed(futures):
                self.assertTrue(future.result())

    def test_concurrent_independent_circular_chains_per_thread(self) -> None:
        """Verify thread-local cycle detection works concurrently across different threads."""
        self.container.register_factory("loop_a", lambda c: c.resolve("loop_b"))
        self.container.register_factory("loop_b", lambda c: c.resolve("loop_a"))
        self.container.register_singleton("safe_service", "safe")

        num_threads = 10

        def cycle_worker() -> bool:
            # Each thread attempts to resolve the loop, must catch CircularDependencyError
            try:
                self.container.resolve("loop_a")
                return False
            except CircularDependencyError:
                # After detecting cycle, thread should still resolve safe services fine
                res = self.container.resolve("safe_service")
                return res == "safe"

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(cycle_worker) for _ in range(num_threads)]
            for future in as_completed(futures):
                self.assertTrue(future.result())


if __name__ == "__main__":
    unittest.main()
