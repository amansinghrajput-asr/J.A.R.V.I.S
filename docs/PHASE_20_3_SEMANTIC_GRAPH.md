# Phase 20.3 – Semantic Knowledge Graph & Temporal Fact Engine

**Status**: Verified & Production-Ready  
**Milestone**: Phase 20.0 (Sprint 20.3)  
**Module**: `app/ai/planner/metacognition/knowledge_graph.py`

---

## 1. Architectural Overview

Sprint 20.3 introduces the **Semantic Knowledge Graph & Temporal Fact Engine** into J.A.R.V.I.S.

In previous phases, memory was represented either as unstructured conversation strings (Phase 5), a task blackboard (Phase 18 `SharedAgentMemory`), or token-indexed trajectory runs (Phase 19 `EpisodicMemoryStore`). 

The `SemanticKnowledgeGraph` elevates this infrastructure into a formal, directed knowledge graph of typed $(S, P, O)$ entity relations, augmented with:
- **Epistemic Confidence Scoring**: Every relation carries an explicit $[0.0, 1.0]$ certainty rating.
- **Temporal Fact Expiration (TTL)**: Facts can be transient or permanent. Expired facts are automatically filtered and efficiently pruned.
- **Multi-Index Querying**: Primary key mapping plus dedicated Subject, Predicate, and Object inverted indices.
- **Topological & Shortest-Path Traversal**: Deterministic Breadth-First Search (BFS) for relationship pathfinding across entities.

```
       +-----------------------------------------------------------+
       |                  SemanticKnowledgeGraph                   |
       |  - Re-entrant RLock Thread Safety                         |
       |  - Zero Global State (Pure Dependency Injection)          |
       +-----------------------------+-----------------------------+
                                     |
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
     +-----------------+    +-----------------+    +-----------------+
     |  Subject Index  |    | Predicate Index |    |  Object Index   |
     | Dict[str, Set]  |    | Dict[str, Set]  |    | Dict[str, Set]  |
     +--------+--------+    +--------+--------+    +--------+--------+
              |                      |                      |
              +----------------------+----------------------+
                                     |
                                     v
                       +---------------------------+
                       |    Relations Storage      |
                       | Dict[Tuple, TripletRecord]|
                       +---------------------------+
                                     |
                                     v
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
       [Query & Filter]       [TTL Expiration]      [BFS Path Search]
       - Confidence >= min    - is_expired(now)     - Shortest Path
       - O(1) Candidate Set   - remove_expired()    - Deterministic
```

---

## 2. Component Specifications

### 2.1 `KnowledgeTriplet` (`models.py`)
Immutable, frozen dataclass modeling an atomic semantic relation:
```python
@dataclass(frozen=True)
class KnowledgeTriplet:
    subject: str
    predicate: str
    object_: str
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        if self.expires_at is None:
            return False
        now = current_time if current_time is not None else time.time()
        return now >= self.expires_at
```

### 2.2 `SemanticKnowledgeGraph` (`knowledge_graph.py`)

#### Public API:
- `add_relation(subject, predicate, object_, confidence=1.0, valid_duration_sec=None) -> KnowledgeTriplet`
  Inserts or overwrites an active relation with latest confidence and expiration timestamp. Updates all 3 index sets.
- `query(subject=None, predicate=None, object_=None, min_confidence=0.0) -> List[KnowledgeTriplet]`
  Performs set intersection over active index candidates. Automatically filters out expired relations. Returns deterministically sorted results.
- `contains(subject, predicate, object_, min_confidence=0.0) -> bool`
  $O(1)$ existence check for non-expired triplets meeting minimum confidence.
- `remove_expired() -> int`
  Scans active relations, deletes expired entries from storage, cleans index references, and returns total count pruned.
- `relation_count() -> int`
  Returns the count of active (non-expired) relations in the graph.
- `find_neighbors(node: str) -> List[str]`
  Returns directly connected entity neighbors across incoming and outgoing active edges in deterministic alphabetical order.
- `find_path(start_node: str, end_node: str) -> Optional[List[str]]`
  Finds the shortest entity traversal path between `start_node` and `end_node` using deterministic BFS. Returns `None` if disconnected.
- `export_dict() -> Dict[str, Any]` & `from_dict(data) -> None`
  Full round-trip JSON-compatible serialization preserving timestamps and confidence.
- `clear() -> None`
  Purges all relations and clears all subject/predicate/object indices.

---

## 3. Algorithmic Complexity

| Operation | Index Utilized | Average Time Complexity | Worst-Case Time Complexity |
|---|---|---|---|
| **`add_relation`** | Key dict + 3 index sets | $O(1)$ | $O(1)$ |
| **`contains`** | Direct tuple key lookup | $O(1)$ | $O(1)$ |
| **`query` (Single filter)** | `_by_subject` / `_by_predicate` / `_by_object` | $O(K \log K)$ ($K$ = matches) | $O(N \log N)$ |
| **`query` (Multi filter)** | Set intersection of indices | $O(\min(K_1, K_2) + K \log K)$ | $O(N \log N)$ |
| **`find_neighbors`** | `_by_subject[node] \cup _by_object[node]` | $O(D \log D)$ ($D$ = degree) | $O(N \log N)$ |
| **`find_path` (BFS)** | Adjacency via neighbor indices | $O(V + E)$ | $O(V + E)$ |
| **`remove_expired`** | Linear relation scan + set discards | $O(E_{\text{expired}})$ | $O(N)$ |

---

## 4. Empirical Performance Benchmarks

Measured on Windows 11 under Python 3.13 via `benchmarks/metacognition_graph_perf.py`:

| Subsystem / Operation | Workload | Measured Latency | Architectural Budget | Status |
|---|---|---|---|---|
| **Knowledge Insertion** | 1,000-node graph chain | **5.060 µs** | < 100.0 µs | **PASS** |
| **Knowledge Lookup** | Multi-index query on 1,000 nodes | **2.104 µs** | < 500.0 µs | **PASS** |
| **BFS Shortest Path** | Multi-hop shortest path search | **107.482 µs** | < 1,000.0 µs | **PASS** |
| **Expiration Cleanup** | Pruning 1,000 expired triplets | **1,009.700 µs** | < 2,000.0 µs | **PASS** |

**Summary**: 4 of 4 performance targets passed comfortably under budget.

---

## 5. Usage Examples

```python
from app.ai.planner.metacognition.knowledge_graph import SemanticKnowledgeGraph

# 1. Initialize an isolated, thread-safe graph
kg = SemanticKnowledgeGraph()

# 2. Add permanent and transient facts
kg.add_relation("Jarvis", "controls", "DesktopAutomation", confidence=0.99)
kg.add_relation("DesktopAutomation", "depends_on", "PyAutoGUI", confidence=0.95)
kg.add_relation("UserSession", "status", "active", confidence=1.0, valid_duration_sec=3600.0)

# 3. Query with confidence filter
results = kg.query(subject="Jarvis", min_confidence=0.9)
for triplet in results:
    print(f"{triplet.subject} --[{triplet.predicate}]--> {triplet.object_}")

# 4. Find BFS shortest path
path = kg.find_path("Jarvis", "PyAutoGUI")
# Output: ['Jarvis', 'DesktopAutomation', 'PyAutoGUI']

# 5. Clean up expired transient relations
removed = kg.remove_expired()
```

---

## 6. Thread Safety & Design Invariants

- **Re-entrant Lock (`threading.RLock`)**: All state-mutating operations and reader queries acquire `self._lock`, preventing race conditions during concurrent modifications.
- **Zero Global State**: No singletons, static class variables, or module-level mutable lists are used. Every instance of `SemanticKnowledgeGraph` is fully isolated and suitable for dependency injection into agents, planners, and sub-swarms.
- **Deterministic Ordering**: Query outputs and graph neighbor traversals sort results alphabetically by `(subject, predicate, object_)`, eliminating non-deterministic iteration order across executions.
