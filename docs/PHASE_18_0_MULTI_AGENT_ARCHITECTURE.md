# Phase 18.0 – Multi-Agent Planner Architecture

**Status**: Completed & Verified  
**Date**: September 2026  
**Module**: `app/ai/planner/multi_agent`  

---

## 1. Architecture Overview

Phase 18.0 introduces a modular, thread-safe, and asynchronous multi-agent coordination layer built directly on top of the Adaptive Planner DAG Engine (Phase 16), Observability Foundation (Phase 17.0), and Execution Infrastructure (Phase 17.5).

Rather than forcing execution through a monolithic planner or hardcoded single-thread worker, the Multi-Agent Planner enables specialized autonomous agents (planners, workers, critics, summarizers, and human surrogates) to collaborate, delegate tasks, communicate over a unified bus, synchronize via a shared blackboard memory, and iteratively refine plans via reflection and conflict resolution.

### Architectural Principles:
- **100% Backward Compatibility**: Single-agent plans run unchanged. Existing `Planner`, `Executor`, `ExecutionController`, and `TimeoutManager` APIs operate without breakage.
- **Zero Global State**: Every registry, bus, blackboard, resolver, and coordinator is an instantiable object configured via dependency injection. Multiple isolated multi-agent clusters can coexist safely in the same process.
- **Event-Driven & Dual-Mode**: Agents can communicate and poll synchronously (`threading.RLock`, `queue.Queue`) or asynchronously (`asyncio.Queue`, non-blocking coroutines).
- **Sub-Millisecond Overhead**: Registry queries, message routing, blackboard operations, and conflict resolution execute within microsecond budgets (< 2% total end-to-end task execution overhead).

---

## 2. Design Goals

1. **Role Specialization**: Decouple domain expertise (e.g., Code Specialist, Web Scraper, Data Critic) into self-contained agents with declared capabilities, supported actions, and affinity scores.
2. **Dynamic Capability Discovery & Selection**: Automatically route DAG tasks to the optimal agent using multi-criteria weighted scoring (action match, domain affinity, load, confidence).
3. **Structured Delegation**: Support asynchronous delegation with correlation IDs, timeout controls, and structured error propagation.
4. **Coordinated Blackboard Memory**: Provide atomic, optimistic-concurrency shared state and artifact storage with version tracking.
5. **Robust Conflict Resolution**: Adjudicate conflicting proposals or outputs through confidence weighting, priority escalation, or majority voting.
6. **Iterative Reflection & Critique**: Provide bounded self-critique loops where critic agents evaluate candidate plans and task outputs to enforce quality bars before finalizing.

---

## 3. Component Diagram

```mermaid
graph TD
    User([User / Application]) --> Coordinator[MultiAgentCoordinator]
    
    subgraph Coordination Layer
        Coordinator --> Registry[AgentRegistry & Selector]
        Coordinator --> CommBus[AgentCommunicationBus]
        Coordinator --> Memory[SharedAgentMemory & ArtifactStore]
        Coordinator --> Resolver[ConflictResolver]
        Coordinator --> Reflection[ReflectionPipeline]
        Coordinator --> Aggregator[ResultAggregator]
    end

    subgraph Agent Pool
        Registry -. discovers .-> WorkerA[WorkerAgent: Code Specialist]
        Registry -. discovers .-> WorkerB[WorkerAgent: Research Specialist]
        Registry -. discovers .-> CriticA[CriticAgent: Security / Quality]
        Registry -. discovers .-> CoordAgent[BaseAgent: Sub-Coordinator]
    end

    subgraph Observability
        CommBus -. publishes .-> EventBus[PlannerEventBus]
        EventBus -. traces .-> MetricsCollector[Metrics & Observability]
    end

    subgraph Execution
        WorkerA --> CommBus
        WorkerB --> CommBus
        CriticA --> Reflection
        WorkerA <--> Memory
        WorkerB <--> Memory
    end
```

---

## 4. Agent Lifecycle

Every agent in the swarm transitions through well-defined lifecycle states:

```mermaid
stateDiagram-v2
    [*] --> REGISTERED: Initialize & Register Manifest
    REGISTERED --> IDLE: Ready for Tasks
    IDLE --> BUSY: Task Delegated / Message Processing
    BUSY --> IDLE: Task Completed Successfully
    BUSY --> ERROR: Unhandled Exception / Failure
    ERROR --> IDLE: Recovered / Reset
    IDLE --> PAUSED: Swarm Paused / Suspended
    PAUSED --> IDLE: Swarm Resumed
    IDLE --> TERMINATED: Deregistered / Shut Down
    ERROR --> TERMINATED: Fatal Error
    TERMINATED --> [*]
```

### Lifecycle States (`AgentStatus`):
- `REGISTERED`: Agent instantiated and published its manifest to the registry.
- `IDLE`: Ready to accept delegations or process messages.
- `BUSY`: Actively executing a task or delegating to sub-agents.
- `PAUSED`: Execution suspended by coordinator or execution controller.
- `ERROR`: Agent experienced a fault; awaiting recovery or task retry.
- `TERMINATED`: Deregistered and removed from active scheduling.

---

## 5. Communication Protocol

Inter-agent communication is mediated by the `AgentCommunicationBus`:

- **Message Schema (`AgentMessage`)**:
  - `message_id`: Unique UUID4 per message.
  - `sender_id` & `recipient_id`: Addressing identifiers (or wildcard `*` for broadcast).
  - `message_type`: `TASK_REQUEST`, `TASK_RESPONSE`, `CRITIQUE`, `BROADCAST`, `HEARTBEAT`, `ERROR`.
  - `correlation_id`: Matches requests with responses across async hops.
  - `payload`: Structured dictionary.
  - `timestamp`: Monotonic execution timestamp.
- **Dispatch Modes**:
  - **Asynchronous Mailboxes**: Dual sync `queue.Queue` and async `asyncio.Queue` per registered agent.
  - **Synchronous Request-Reply (`send_and_wait`)**: Blocks caller with timeout until matching correlation response is delivered.
  - **Broadcast**: Fans out notifications to all active agents.
  - **Planner Event Bus Mirroring**: Outbound messages are converted to `AgentMessageSent` events on `PlannerEventBus`.

---

## 6. Shared Execution Memory

Shared memory is provided by `SharedAgentMemory` and `ArtifactStore`:

- **Three Visibility Scopes**:
  1. `global` / blackboard: Readable and writable across all agents in the swarm.
  2. `agent:<agent_id>`: Private workspace storage for an individual agent.
  3. `task:<task_id>`: Isolated context and transient results for a single DAG task.
- **Optimistic Concurrency Control**:
  - Each entry tracks a monotonic `version` integer.
  - Atomic conditional writes (`set(..., expected_version=V)`) prevent race conditions.
- **Artifact Store**:
  - Thread-safe storage for large binary or structured payloads (files, reports, embeddings) referenced by URI.

---

## 7. Agent Registry & Capability Discovery

- **`AgentManifest`**:
  - `agent_id`: Unique identifier.
  - `role`: `PLANNER`, `WORKER`, `CRITIC`, `COORDINATOR`, `SUMMARIZER`, `HUMAN_PROXY`.
  - `capabilities`: List of `AgentCapability` descriptors (action name, domain affinity, performance score, timeout).
  - `max_concurrency`: Maximum parallel tasks supported by the agent.
- **Inverted Indexing**:
  - `AgentRegistry` indexes agents into hash sets by role, domain, and action.
  - Constant-time lookup (`O(1)`) to discover all eligible agents for any given task action.

---

## 8. Dynamic Agent Selection

`AgentSelector` selects the optimal agent for a given task using a normalized multi-criteria scoring algorithm:

$$\text{Score}(A) = w_a \cdot S_{\text{action}} + w_d \cdot S_{\text{domain}} + w_l \cdot S_{\text{load}} + w_c \cdot S_{\text{conf}}$$

- **Action Match ($w_a = 0.40$)**: 1.0 if agent explicitly advertises capability for `task.action`.
- **Domain Affinity ($w_d = 0.25$)**: Match score between agent capability domain and task context.
- **Load / Availability ($w_l = 0.20$)**: Higher score for idle agents with remaining capacity.
- **Confidence Rating ($w_c = 0.15$)**: Agent historical capability performance rating.

---

## 9. Task Delegation Model

1. **Task Arrival**: `MultiAgentCoordinator` inspects task dependencies in the DAG.
2. **Assignment**: If `task.assigned_agent` is specified, direct delegation occurs; otherwise, `AgentSelector` ranks and chooses the top agent.
3. **Execution**: The coordinator wraps the task into a `DelegationRequest` and invokes `agent.execute_delegation(request)` or dispatches via the communication bus.
4. **State Transition**: The agent transitions `IDLE -> BUSY -> IDLE`.
5. **Output Delivery**: A `DelegationResponse` is returned with status, execution artifacts, and telemetry.

---

## 10. Parallel Execution Model

- Multi-agent execution integrates seamlessly with DAG parallelism.
- Tasks in independent branches of the DAG can be assigned to different worker agents simultaneously.
- Thread-safe mailboxes, fine-grained locks, and optimistic memory versioning ensure that concurrent workers do not block or corrupt shared blackboard state.

---

## 11. Result Aggregation

`ResultAggregator` synthesizes task outputs from multiple agents:
- **Concatenation (`CONCATENATE`)**: Combines sequential text or item lists.
- **Merge (`MERGE`)**: Recursively merges JSON/dictionary structures.
- **First Success (`FIRST_SUCCESS`)**: Selects the fastest winning result among redundant parallel agents.
- **Adjudicated Best (`BEST_CONFIDENCE`)**: Selects highest confidence result validated by critics.

---

## 12. Conflict Resolution

When multiple agents propose contradictory plans or outputs:
- **`ConflictResolver`** supports 4 distinct strategies:
  1. `CONFIDENCE_WEIGHTED`: Selects proposal with highest weighted score: $\text{score} = \text{confidence} \times \text{priority}$.
  2. `PRIORITY`: Highest priority role wins (e.g., Critic > Worker).
  3. `VOTING`: Plurality voting across multiple evaluating agents.
  4. `ADJUDICATION`: Arbitrated by designated `COORDINATOR` agent.

---

## 13. Reflection & Self-Critique Pipeline

`ReflectionPipeline` provides self-correction without human intervention:
- **Plan Critique**: Validates proposed DAG against completeness, safety, and efficiency criteria.
- **Task Critique**: Analyzes task outputs (score 0.0 to 1.0, issues detected, improvement suggestions).
- **Refinement Loops**: Iteratively refines outputs up to `max_refinement_iterations` (default: 3) or until score meets `min_acceptable_score` (default: 0.8).

---

## 14. Failure Handling & Recovery Strategy

- **Agent Timeout**: If an agent fails to respond within `timeout_seconds`, the delegation returns `Status.FAILED` with a timeout reason.
- **Automatic Fallback Delegation**: If the primary agent faults, `MultiAgentCoordinator` falls back to the second-highest scoring agent in the registry.
- **Circuit Breaking**: Repeated failures increment the agent's error count, causing it to enter `AgentStatus.ERROR` and be bypassed by `AgentSelector`.

---

## 15. Performance Considerations

- **Registration Latency**: 3.04 µs (Design budget: < 20 µs)
- **Selection Latency**: 11.99 µs (Design budget: < 2,000 µs)
- **Message Dispatch**: 2.62 µs (Design budget: < 50 µs)
- **Shared Memory Access**: 1.35 µs (Design budget: < 10 µs)
- **Conflict Resolution**: 7.80 µs (Design budget: < 50 µs)
- **End-to-End Coordination Overhead**: **1.61%** (Design budget: < 5%)

---

## 16. Security Considerations

- **Agent Isolation**: Agents only have access to designated blackboard namespaces and cannot execute arbitrary OS commands without explicit tools.
- **Payload Validation**: All inter-agent messages are strictly typed and validated using Pydantic dataclasses.
- **Zero Global Mutation**: No global registry or singleton state prevents cross-tenant state leakage in multi-tenant deployments.

---

## 17. Testing Strategy

The multi-agent architecture is covered by an extensive automated test matrix:
1. `test_agent_models.py`: Model serialization, immutability, role permissions.
2. `test_agent_communication.py`: Direct messaging, request-response wait, broadcast, and async queues.
3. `test_agent_registry.py`: Capability indexing, query filtering, multi-criteria scoring.
4. `test_shared_memory.py`: Blackboard isolation, optimistic concurrency conflicts, artifact storage.
5. `test_conflict_resolution.py`: Confidence weighting, priority tie-breaking, voting adjudication.
6. `test_reflection_pipeline.py`: Plan and task critique, iterative refinement loops.
7. `test_multi_agent_coordinator.py`: End-to-end multi-agent DAG coordination and fallback delegation.
8. `test_architecture.py`: Module isolation and zero global state verification.

---

## 18. Sprint Breakdown Summary

| Sprint | Milestone | Key Deliverables | Status |
|---|---|---|---|
| **18.1** | Foundation & Protocols | `models.py`, `base.py`, `protocol.py`, events | Completed |
| **18.2** | Registry & Discovery | `registry.py` (Registry & Selector) | Completed |
| **18.3** | Blackboard & Memory | `memory.py` (SharedAgentMemory & ArtifactStore) | Completed |
| **18.4** | Conflict & Reflection | `conflict.py`, `critic.py` (ReflectionPipeline) | Completed |
| **18.5** | Coordinator & Integration | `coordinator.py`, benchmarks, docs | Completed |

---

## 19. Future Roadmap

- **Hierarchical Sub-Swarms**: Nested coordinator agents for recursive sub-goal decomposition.
- **Dynamic Learning / Skill Accumulation**: Integrating memory embeddings to record agent performance history across runs.
- **Distributed Agent Transport**: Optional gRPC / WebSocket communication bus adapter for physical multi-node agent swarms.
