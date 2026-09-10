"""Unit tests for Swarm Supervisor & Dynamic Load Balancer (Sprint 19.4).

Covers heartbeat registration, health report generation, timeout/stuck agent detection,
auto-healing recovery, workload distribution, least-loaded assignment, queue rebalancing,
task migration, circuit breaking, and deterministic behavior.
"""

import time
import unittest

from app.ai.planner.multi_agent.base import WorkerAgent
from app.ai.planner.multi_agent.models import AgentCapability, AgentManifest, AgentRole, AgentStatus
from app.ai.planner.swarm.balancer import BalancingPolicy, DynamicLoadBalancer
from app.ai.planner.swarm.models import SubSwarmConfig, SwarmStatus
from app.ai.planner.swarm.subswarm import SubSwarm
from app.ai.planner.swarm.supervisor import SwarmHealthReport, SwarmSupervisor


def _create_mock_agent(agent_id: str, role: AgentRole = AgentRole.WORKER) -> WorkerAgent:
    manifest = AgentManifest(
        agent_id=agent_id,
        role=role,
        capabilities=[AgentCapability(name="work", supported_actions={"work"})],
    )
    return WorkerAgent(manifest=manifest)


class TestSwarmSupervisor(unittest.TestCase):
    """Test suite for SwarmSupervisor."""

    def test_heartbeat_registration_and_healthy_swarm_detection(self) -> None:
        """Verify registering healthy swarm and reporting healthy status."""
        supervisor = SwarmSupervisor(heartbeat_timeout=10.0, stuck_threshold=15.0)
        swarm = SubSwarm(config=SubSwarmConfig(swarm_id="swarm_alpha"))

        agent = _create_mock_agent("agent_1")
        swarm.registry.register(agent)
        supervisor.register_swarm(swarm)

        report = supervisor.check_health()
        self.assertTrue(report.healthy)
        self.assertEqual(report.total_swarms, 1)
        self.assertEqual(report.active_agent_count, 1)
        self.assertEqual(len(report.stuck_agents), 0)
        self.assertEqual(len(report.faulted_agents), 0)

    def test_timeout_and_stuck_agent_detection(self) -> None:
        """Verify supervisor detects agents stuck in BUSY past threshold."""
        supervisor = SwarmSupervisor(heartbeat_timeout=0.05, stuck_threshold=0.05)
        swarm = SubSwarm(config=SubSwarmConfig(swarm_id="swarm_beta"))

        agent = _create_mock_agent("stuck_worker")
        swarm.registry.register(agent)
        supervisor.register_swarm(swarm)

        # Set agent to BUSY
        agent.set_status(AgentStatus.BUSY)
        supervisor.check_health()  # primes _agent_busy_since

        # Wait for timeout
        time.sleep(0.08)

        report = supervisor.check_health()
        self.assertFalse(report.healthy)
        self.assertIn("stuck_worker", report.stuck_agents)

    def test_auto_healing_resets_stuck_and_faulted_agents(self) -> None:
        """Verify auto_heal resets stuck and faulted agents back to IDLE."""
        supervisor = SwarmSupervisor(heartbeat_timeout=0.02, stuck_threshold=0.02)
        swarm = SubSwarm(config=SubSwarmConfig(swarm_id="swarm_gamma"))

        a1 = _create_mock_agent("a_stuck")
        a2 = _create_mock_agent("a_faulted")
        swarm.registry.register(a1)
        swarm.registry.register(a2)
        supervisor.register_swarm(swarm)

        a1.set_status(AgentStatus.BUSY)
        a2.set_status(AgentStatus.FAILED)
        swarm.status = SwarmStatus.FAILED

        supervisor.check_health()
        time.sleep(0.04)

        # Perform auto-healing
        actions = supervisor.auto_heal()
        self.assertGreaterEqual(len(actions), 3)

        # Verify all statuses are restored
        self.assertEqual(a1.status, AgentStatus.IDLE)
        self.assertEqual(a2.status, AgentStatus.IDLE)
        self.assertEqual(swarm.status, SwarmStatus.READY)

        # Health report should now be clean
        report = supervisor.check_health()
        self.assertTrue(report.healthy)

    def test_health_report_generation_and_serialization(self) -> None:
        """Verify SwarmHealthReport serialization."""
        report = SwarmHealthReport(
            healthy=True,
            total_swarms=2,
            swarm_statuses={"s1": "READY", "s2": "READY"},
            stuck_agents=[],
            faulted_agents=[],
            active_agent_count=4,
        )
        d = report.to_dict()
        self.assertTrue(d["healthy"])
        self.assertEqual(d["total_swarms"], 2)

        restored = SwarmHealthReport.from_dict(d)
        self.assertTrue(restored.healthy)
        self.assertEqual(restored.active_agent_count, 4)

    def test_load_distribution_calculation(self) -> None:
        """Verify load distribution calculation across agents and swarms."""
        supervisor = SwarmSupervisor()
        swarm = SubSwarm(config=SubSwarmConfig(swarm_id="swarm_load"))

        a1 = _create_mock_agent("a_idle")
        a2 = _create_mock_agent("a_busy")
        swarm.registry.register(a1)
        swarm.registry.register(a2)
        supervisor.register_swarm(swarm)

        a1.set_status(AgentStatus.IDLE)
        a2.set_status(AgentStatus.BUSY)

        dist = supervisor.get_load_distribution()
        self.assertEqual(dist["a_idle"], 0.0)
        self.assertEqual(dist["a_busy"], 1.0)
        self.assertEqual(dist["swarm_load"], 0.5)


class TestDynamicLoadBalancer(unittest.TestCase):
    """Test suite for DynamicLoadBalancer."""

    def test_least_loaded_assignment_and_deterministic_tie_breaking(self) -> None:
        """Verify tasks are assigned to least-loaded agents with deterministic tie-breaking."""
        balancer = DynamicLoadBalancer(policy=BalancingPolicy.LEAST_LOADED)

        # 3 candidates with different loads
        loads = {"agent_z": 5, "agent_a": 2, "agent_b": 2}
        assigned = balancer.assign_task("task_1", ["agent_z", "agent_a", "agent_b"], agent_loads=loads)

        # agent_a and agent_b tie with load 2; deterministic tie-break selects 'agent_a' (alphabetical)
        self.assertEqual(assigned, "agent_a")

    def test_rebalance_evens_out_queue_variance(self) -> None:
        """Verify rebalance shifts tasks from overloaded queues to underloaded ones."""
        balancer = DynamicLoadBalancer()

        queues = {
            "worker_heavy": ["t1", "t2", "t3", "t4", "t5"],
            "worker_light": ["t6"],
        }

        rebalanced = balancer.rebalance(queues, max_imbalance_threshold=1)
        # 6 tasks across 2 workers -> 3 and 3
        self.assertEqual(len(rebalanced["worker_heavy"]), 3)
        self.assertEqual(len(rebalanced["worker_light"]), 3)
        # Total tasks preserved
        all_tasks = set(rebalanced["worker_heavy"] + rebalanced["worker_light"])
        self.assertEqual(all_tasks, {"t1", "t2", "t3", "t4", "t5", "t6"})

    def test_task_migration_from_agent(self) -> None:
        """Verify migrating tasks evacuates from source agent to target agents."""
        balancer = DynamicLoadBalancer()

        # Seed source queue
        balancer.assign_task("m1", ["agent_source"])
        balancer.assign_task("m2", ["agent_source"])
        balancer.assign_task("m3", ["agent_source"])

        migrated = balancer.migrate_tasks("agent_source", ["target_x", "target_y"], max_tasks=2)
        self.assertEqual(len(migrated), 2)
        self.assertEqual(migrated, ["m1", "m2"])

        # Remaining in source
        self.assertEqual(balancer._agent_queues["agent_source"], ["m3"])
        # Transferred to targets
        total_in_targets = len(balancer._agent_queues.get("target_x", [])) + len(
            balancer._agent_queues.get("target_y", [])
        )
        self.assertEqual(total_in_targets, 2)

    def test_circuit_breaker_trips_and_avoids_failed_agent(self) -> None:
        """Verify circuit breaker trips on repeated failures and balancer avoids tripped agent."""
        balancer = DynamicLoadBalancer(failure_threshold=2, cooldown_seconds=0.1)

        self.assertFalse(balancer.is_circuit_broken("flaky_agent"))

        # 1st failure
        tripped = balancer.apply_circuit_breaker("flaky_agent", failure_count=1)
        self.assertFalse(tripped)

        # 2nd failure -> Trips!
        tripped = balancer.apply_circuit_breaker("flaky_agent", failure_count=1)
        self.assertTrue(tripped)
        self.assertTrue(balancer.is_circuit_broken("flaky_agent"))

        # Balancer should bypass flaky_agent in favor of backup_agent
        assigned = balancer.assign_task("task_test", ["flaky_agent", "backup_agent"])
        self.assertEqual(assigned, "backup_agent")

        # After cooldown, circuit resets
        time.sleep(0.12)
        self.assertFalse(balancer.is_circuit_broken("flaky_agent"))


if __name__ == "__main__":
    unittest.main()
