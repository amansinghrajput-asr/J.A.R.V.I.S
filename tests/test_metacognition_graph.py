"""Unit tests for Phase 20 Sprint 20.3 Semantic Knowledge Graph & Temporal Fact Engine.

Covers:
- Relation insertion and indexing
- Duplicate relation overwrite with latest confidence
- Confidence filtering
- Temporal expiration detection
- Purging expired relations via remove_expired
- Serialization round-trips (export_dict, from_dict, create_from_dict)
- Query combinations (subject, predicate, object, multi-criteria)
- Neighbor discovery (incoming and outgoing)
- Shortest path BFS traversal
- Disconnected graph path query
- Deterministic result ordering
- Active relation count tracking
- Concurrent mutations, lookups, and thread safety
"""

import concurrent.futures
import time
import unittest

from app.ai.planner.metacognition.knowledge_graph import SemanticKnowledgeGraph
from app.ai.planner.metacognition.models import KnowledgeTriplet


class TestSemanticKnowledgeGraph(unittest.TestCase):
    """Comprehensive test suite for SemanticKnowledgeGraph."""

    def setUp(self) -> None:
        self.graph = SemanticKnowledgeGraph()

    def test_relation_insertion_and_contains(self) -> None:
        """Verify inserting a relation and querying contains."""
        triplet = self.graph.add_relation("Alice", "knows", "Bob", confidence=0.95)
        self.assertIsInstance(triplet, KnowledgeTriplet)
        self.assertEqual(triplet.subject, "Alice")
        self.assertEqual(triplet.predicate, "knows")
        self.assertEqual(triplet.object_, "Bob")
        self.assertEqual(triplet.confidence, 0.95)
        self.assertTrue(self.graph.contains("Alice", "knows", "Bob"))
        self.assertFalse(self.graph.contains("Bob", "knows", "Alice"))

    def test_duplicate_relation_overwrite(self) -> None:
        """Verify adding an identical relation overwrites with latest confidence and TTL."""
        self.graph.add_relation("Server1", "status", "online", confidence=0.7)
        self.assertEqual(self.graph.relation_count(), 1)
        res = self.graph.query(subject="Server1")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].confidence, 0.7)

        # Overwrite with higher confidence
        self.graph.add_relation("Server1", "status", "online", confidence=0.99)
        self.assertEqual(self.graph.relation_count(), 1)
        res2 = self.graph.query(subject="Server1")
        self.assertEqual(len(res2), 1)
        self.assertEqual(res2[0].confidence, 0.99)

    def test_confidence_filtering(self) -> None:
        """Verify query returns only relations meeting min_confidence threshold."""
        self.graph.add_relation("EntityA", "type", "Class1", confidence=0.4)
        self.graph.add_relation("EntityB", "type", "Class1", confidence=0.8)
        self.graph.add_relation("EntityC", "type", "Class1", confidence=0.95)

        low = self.graph.query(predicate="type", min_confidence=0.0)
        self.assertEqual(len(low), 3)

        mid = self.graph.query(predicate="type", min_confidence=0.7)
        self.assertEqual(len(mid), 2)
        self.assertEqual([t.subject for t in mid], ["EntityB", "EntityC"])

        high = self.graph.query(predicate="type", min_confidence=0.9)
        self.assertEqual(len(high), 1)
        self.assertEqual(high[0].subject, "EntityC")

    def test_temporal_expiration_in_query(self) -> None:
        """Verify expired relations are automatically omitted from queries."""
        # Add expired relation (TTL = 0.01 sec)
        self.graph.add_relation("TempFact", "valid_for", "brief_moment", valid_duration_sec=0.02)
        # Add permanent relation
        self.graph.add_relation("PermanentFact", "valid_for", "eternity")

        # Immediately available
        self.assertTrue(self.graph.contains("TempFact", "valid_for", "brief_moment"))
        self.assertEqual(self.graph.relation_count(), 2)

        # Wait for expiration
        time.sleep(0.04)

        # TempFact should now be omitted from query and contains
        self.assertFalse(self.graph.contains("TempFact", "valid_for", "brief_moment"))
        res = self.graph.query(subject="TempFact")
        self.assertEqual(len(res), 0)
        self.assertEqual(self.graph.relation_count(), 1)

    def test_remove_expired(self) -> None:
        """Verify remove_expired purges expired entries and returns exact count removed."""
        self.graph.add_relation("F1", "rel", "O1", valid_duration_sec=0.01)
        self.graph.add_relation("F2", "rel", "O2", valid_duration_sec=0.01)
        self.graph.add_relation("F3", "rel", "O3", valid_duration_sec=10.0)

        time.sleep(0.03)

        removed = self.graph.remove_expired()
        self.assertEqual(removed, 2)
        self.assertEqual(self.graph.relation_count(), 1)

        # Second cleanup should remove 0
        self.assertEqual(self.graph.remove_expired(), 0)

    def test_query_combinations(self) -> None:
        """Verify querying by subject only, predicate only, object only, and multi-criteria."""
        self.graph.add_relation("User", "reads", "Doc1")
        self.graph.add_relation("User", "edits", "Doc1")
        self.graph.add_relation("Admin", "edits", "Doc1")
        self.graph.add_relation("User", "reads", "Doc2")

        # Subject only
        self.assertEqual(len(self.graph.query(subject="User")), 3)

        # Predicate only
        self.assertEqual(len(self.graph.query(predicate="edits")), 2)

        # Object only
        self.assertEqual(len(self.graph.query(object_="Doc1")), 3)

        # Subject and predicate
        res = self.graph.query(subject="User", predicate="reads")
        self.assertEqual(len(res), 2)

        # All 3
        res_exact = self.graph.query(subject="Admin", predicate="edits", object_="Doc1")
        self.assertEqual(len(res_exact), 1)

        # Non-existent
        self.assertEqual(len(self.graph.query(subject="NonExistent")), 0)

    def test_neighbor_discovery(self) -> None:
        """Verify finding bidirectional neighbors of an entity."""
        self.graph.add_relation("NodeA", "links_to", "NodeB")
        self.graph.add_relation("NodeC", "links_to", "NodeA")
        self.graph.add_relation("NodeA", "links_to", "NodeD")

        neighbors = self.graph.find_neighbors("NodeA")
        # Neighbors of NodeA should be NodeB, NodeC, NodeD in sorted order
        self.assertEqual(neighbors, ["NodeB", "NodeC", "NodeD"])

        # Neighbors of leaf node
        self.assertEqual(self.graph.find_neighbors("NodeB"), ["NodeA"])

        # Non-existent node
        self.assertEqual(self.graph.find_neighbors("Unknown"), [])

    def test_find_path_direct_and_multi_hop(self) -> None:
        """Verify BFS shortest path discovery across 1-hop and multi-hop connections."""
        # A -> B -> C -> D
        self.graph.add_relation("A", "to", "B")
        self.graph.add_relation("B", "to", "C")
        self.graph.add_relation("C", "to", "D")

        # Direct 1-hop
        path_ab = self.graph.find_path("A", "B")
        self.assertEqual(path_ab, ["A", "B"])

        # 3-hop
        path_ad = self.graph.find_path("A", "D")
        self.assertEqual(path_ad, ["A", "B", "C", "D"])

    def test_find_path_shortest_selection(self) -> None:
        """Verify BFS chooses the shortest path when multiple alternate paths exist."""
        # Path 1: A -> B -> C -> D (3 hops)
        # Path 2: A -> E -> D (2 hops)
        self.graph.add_relation("A", "to", "B")
        self.graph.add_relation("B", "to", "C")
        self.graph.add_relation("C", "to", "D")
        self.graph.add_relation("A", "to", "E")
        self.graph.add_relation("E", "to", "D")

        path = self.graph.find_path("A", "D")
        self.assertEqual(path, ["A", "E", "D"])

    def test_find_path_disconnected_graph(self) -> None:
        """Verify finding path between disconnected components returns None."""
        self.graph.add_relation("Component1_A", "to", "Component1_B")
        self.graph.add_relation("Component2_X", "to", "Component2_Y")

        path = self.graph.find_path("Component1_A", "Component2_Y")
        self.assertIsNone(path)

    def test_find_path_same_start_and_end(self) -> None:
        """Verify searching path from node to itself returns single-node list."""
        self.graph.add_relation("X", "rel", "Y")
        self.assertEqual(self.graph.find_path("X", "X"), ["X"])

    def test_deterministic_ordering(self) -> None:
        """Verify query returns results in deterministic alphabetical order."""
        self.graph.add_relation("Charlie", "role", "Worker")
        self.graph.add_relation("Alice", "role", "Leader")
        self.graph.add_relation("Bob", "role", "Worker")

        results = self.graph.query(predicate="role")
        subjects = [t.subject for t in results]
        self.assertEqual(subjects, ["Alice", "Bob", "Charlie"])

    def test_serialization_round_trip(self) -> None:
        """Verify export_dict and from_dict round-trip preservation."""
        self.graph.add_relation("Agent1", "has_skill", "PythonCoder", confidence=0.92)
        self.graph.add_relation("Agent2", "has_skill", "FactCritic", confidence=0.88)

        data = self.graph.export_dict()
        self.assertEqual(data["count"], 2)
        self.assertEqual(len(data["triplets"]), 2)

        # Load into clean graph
        new_graph = SemanticKnowledgeGraph.create_from_dict(data)
        self.assertEqual(new_graph.relation_count(), 2)
        self.assertTrue(new_graph.contains("Agent1", "has_skill", "PythonCoder", min_confidence=0.9))
        self.assertTrue(new_graph.contains("Agent2", "has_skill", "FactCritic", min_confidence=0.8))

    def test_clear(self) -> None:
        """Verify clear removes all relations and indexes completely."""
        self.graph.add_relation("X", "rel", "Y")
        self.graph.add_relation("Y", "rel", "Z")
        self.assertEqual(self.graph.relation_count(), 2)

        self.graph.clear()
        self.assertEqual(self.graph.relation_count(), 0)
        self.assertEqual(len(self.graph.query()), 0)
        self.assertEqual(self.graph.find_neighbors("X"), [])

    def test_invalid_arguments_raise_value_error(self) -> None:
        """Verify empty subject, predicate, or object raises ValueError."""
        with self.assertRaises(ValueError):
            self.graph.add_relation("", "predicate", "object")
        with self.assertRaises(ValueError):
            self.graph.add_relation("subject", "", "object")
        with self.assertRaises(ValueError):
            self.graph.add_relation("subject", "predicate", "")

    def test_concurrent_insert_and_query_thread_safety(self) -> None:
        """Verify thread-safety during concurrent relations additions and queries."""
        def _inserter(worker_id: int) -> None:
            for i in range(50):
                self.graph.add_relation(f"Worker_{worker_id}", "item", f"Item_{i}", confidence=0.8)

        def _reader() -> None:
            for _ in range(50):
                self.graph.query(predicate="item")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            insert_futures = [pool.submit(_inserter, w) for w in range(4)]
            read_futures = [pool.submit(_reader) for _ in range(4)]

            for f in insert_futures + read_futures:
                f.result()

        self.assertEqual(self.graph.relation_count(), 200)

    def test_expired_relation_ignored_in_find_path(self) -> None:
        """Verify BFS path search ignores expired edges."""
        # A -> B (expired) -> C
        self.graph.add_relation("NodeA", "to", "NodeB", valid_duration_sec=0.01)
        self.graph.add_relation("NodeB", "to", "NodeC")

        time.sleep(0.03)

        path = self.graph.find_path("NodeA", "NodeC")
        self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()
