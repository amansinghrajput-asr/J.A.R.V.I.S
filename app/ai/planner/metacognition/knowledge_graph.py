"""Semantic Knowledge Graph and Temporal Fact Engine for Phase 20 Metacognition.

Provides thread-safe storage, indexed querying, temporal expiration pruning,
neighbor discovery, and BFS path traversal over semantic entity-predicate-entity relations.
"""

from __future__ import annotations

from collections import defaultdict, deque
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.ai.planner.metacognition.models import KnowledgeTriplet

logger = logging.getLogger("app.ai.planner.metacognition.knowledge_graph")


class SemanticKnowledgeGraph:
    """Thread-safe semantic knowledge graph with temporal expiration and multi-index lookup."""

    def __init__(self) -> None:
        """Initialize an isolated semantic knowledge graph."""
        self._lock = threading.RLock()
        self._relations: Dict[Tuple[str, str, str], KnowledgeTriplet] = {}
        self._by_subject: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
        self._by_predicate: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
        self._by_object: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)

    def add_relation(
        self,
        subject: str,
        predicate: str,
        object_: str,
        confidence: float = 1.0,
        valid_duration_sec: Optional[float] = None,
    ) -> KnowledgeTriplet:
        """Add or overwrite a relation in the graph.

        Args:
            subject: Subject entity identifier.
            predicate: Relationship predicate name.
            object_: Target entity or literal object.
            confidence: Epistemic certainty rating between 0.0 and 1.0.
            valid_duration_sec: Optional TTL in seconds from current time.

        Returns:
            The created KnowledgeTriplet instance.

        Raises:
            ValueError: If subject, predicate, or object_ is empty.
        """
        if not subject or not subject.strip():
            raise ValueError("Subject cannot be empty.")
        if not predicate or not predicate.strip():
            raise ValueError("Predicate cannot be empty.")
        if not object_ or not object_.strip():
            raise ValueError("Object cannot be empty.")

        s = subject.strip()
        p = predicate.strip()
        o = object_.strip()
        conf = max(0.0, min(1.0, float(confidence)))

        now = time.time()
        expires_at = (now + valid_duration_sec) if valid_duration_sec is not None else None

        triplet = KnowledgeTriplet(
            subject=s,
            predicate=p,
            object_=o,
            confidence=conf,
            created_at=now,
            expires_at=expires_at,
        )

        key = (s, p, o)

        with self._lock:
            # Overwrites existing relation with latest confidence and expiration
            self._relations[key] = triplet
            self._by_subject[s].add(key)
            self._by_predicate[p].add(key)
            self._by_object[o].add(key)
            logger.debug("Added relation (%s, %s, %s) with confidence %.2f", s, p, o, conf)

        return triplet

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[KnowledgeTriplet]:
        """Query relations with optional subject, predicate, object, and confidence filters.

        Expired relations are automatically excluded. Results are returned in deterministic order.

        Args:
            subject: Optional subject entity to match.
            predicate: Optional relation predicate to match.
            object_: Optional object entity to match.
            min_confidence: Minimum confidence threshold.

        Returns:
            Deterministically sorted list of matching active KnowledgeTriplets.
        """
        now = time.time()

        with self._lock:
            candidate_keys: Optional[Set[Tuple[str, str, str]]] = None

            if subject is not None:
                s_clean = subject.strip()
                s_keys = self._by_subject.get(s_clean, set())
                candidate_keys = set(s_keys)

            if predicate is not None:
                p_clean = predicate.strip()
                p_keys = self._by_predicate.get(p_clean, set())
                candidate_keys = p_keys if candidate_keys is None else candidate_keys.intersection(p_keys)

            if object_ is not None:
                o_clean = object_.strip()
                o_keys = self._by_object.get(o_clean, set())
                candidate_keys = o_keys if candidate_keys is None else candidate_keys.intersection(o_keys)

            if candidate_keys is None:
                candidate_keys = set(self._relations.keys())

            results: List[KnowledgeTriplet] = []
            for key in candidate_keys:
                triplet = self._relations.get(key)
                if triplet is None:
                    continue
                if triplet.is_expired(now):
                    continue
                if triplet.confidence < min_confidence:
                    continue
                results.append(triplet)

        # Deterministic ordering: subject ASC, predicate ASC, object_ ASC, confidence DESC
        results.sort(key=lambda t: (t.subject, t.predicate, t.object_, -t.confidence))
        return results

    def contains(
        self,
        subject: str,
        predicate: str,
        object_: str,
        min_confidence: float = 0.0,
    ) -> bool:
        """Check whether an active relation exists in the graph.

        Args:
            subject: Subject entity.
            predicate: Relation predicate.
            object_: Target entity.
            min_confidence: Minimum acceptable confidence.

        Returns:
            True if non-expired matching relation exists, False otherwise.
        """
        if not subject or not predicate or not object_:
            return False

        key = (subject.strip(), predicate.strip(), object_.strip())
        now = time.time()

        with self._lock:
            triplet = self._relations.get(key)
            if triplet is None:
                return False
            if triplet.is_expired(now):
                return False
            return triplet.confidence >= min_confidence

    def remove_expired(self) -> int:
        """Purge all expired relations and clean internal indexes.

        Returns:
            Count of expired relations removed.
        """
        now = time.time()
        removed_count = 0

        with self._lock:
            expired_keys = [
                key for key, triplet in self._relations.items()
                if triplet.is_expired(now)
            ]

            by_subj = self._by_subject
            by_pred = self._by_predicate
            by_obj = self._by_object
            relations = self._relations

            for key in expired_keys:
                if key in relations:
                    del relations[key]
                    s, p, o = key

                    s_set = by_subj.get(s)
                    if s_set is not None:
                        s_set.discard(key)
                        if not s_set:
                            del by_subj[s]

                    p_set = by_pred.get(p)
                    if p_set is not None:
                        p_set.discard(key)
                        if not p_set:
                            del by_pred[p]

                    o_set = by_obj.get(o)
                    if o_set is not None:
                        o_set.discard(key)
                        if not o_set:
                            del by_obj[o]

                    removed_count += 1

            if removed_count > 0:
                logger.debug("Pruned %d expired relations from knowledge graph.", removed_count)

        return removed_count

    def relation_count(self) -> int:
        """Return count of active (non-expired) relations in the graph.

        Returns:
            Integer count of valid relations.
        """
        now = time.time()
        with self._lock:
            return sum(1 for t in self._relations.values() if not t.is_expired(now))

    def find_neighbors(self, node: str) -> List[str]:
        """Return directly connected neighbor nodes for a given entity.

        Traverses both outgoing and incoming active edges.

        Args:
            node: Target entity node identifier.

        Returns:
            Deterministically sorted list of distinct adjacent entity names.
        """
        if not node or not node.strip():
            return []

        target = node.strip()
        now = time.time()
        neighbors: Set[str] = set()

        with self._lock:
            # Outgoing neighbors (node as subject)
            for key in self._by_subject.get(target, ()):
                triplet = self._relations.get(key)
                if triplet and not triplet.is_expired(now):
                    if triplet.object_ != target:
                        neighbors.add(triplet.object_)

            # Incoming neighbors (node as object)
            for key in self._by_object.get(target, ()):
                triplet = self._relations.get(key)
                if triplet and not triplet.is_expired(now):
                    if triplet.subject != target:
                        neighbors.add(triplet.subject)

        return sorted(list(neighbors))

    def find_path(self, start_node: str, end_node: str) -> Optional[List[str]]:
        """Find the shortest path between start_node and end_node using Breadth-First Search (BFS).

        Traversal evaluates adjacent neighbors in deterministic alphabetical order.

        Args:
            start_node: Starting entity node.
            end_node: Target entity node.

        Returns:
            List of node names representing the path from start to end, or None if no path exists.
        """
        if not start_node or not end_node:
            return None

        start = start_node.strip()
        end = end_node.strip()

        if start == end:
            return [start]

        with self._lock:
            visited: Set[str] = {start}
            queue: deque[List[str]] = deque([[start]])

            while queue:
                current_path = queue.popleft()
                current_node = current_path[-1]

                neighbors = self.find_neighbors(current_node)
                for neighbor in neighbors:
                    if neighbor == end:
                        return current_path + [neighbor]

                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(current_path + [neighbor])

        return None

    def export_dict(self) -> Dict[str, Any]:
        """Export all active relations to a serializable dictionary.

        Returns:
            Dictionary containing metadata and active KnowledgeTriplets.
        """
        now = time.time()
        with self._lock:
            active_triplets = [
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object_": t.object_,
                    "confidence": t.confidence,
                    "created_at": t.created_at,
                    "expires_at": t.expires_at,
                }
                for t in self._relations.values()
                if not t.is_expired(now)
            ]

        # Deterministic export ordering
        active_triplets.sort(key=lambda item: (item["subject"], item["predicate"], item["object_"]))

        return {
            "version": "1.0",
            "count": len(active_triplets),
            "triplets": active_triplets,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load relations into the graph from a serialized dictionary.

        Args:
            data: Serialized graph dictionary exported via export_dict.
        """
        triplets_data = data.get("triplets", [])
        with self._lock:
            self.clear()
            for item in triplets_data:
                s = item.get("subject", "")
                p = item.get("predicate", "")
                o = item.get("object_", "")
                if s and p and o:
                    conf = float(item.get("confidence", 1.0))
                    created_at = float(item.get("created_at", time.time()))
                    expires_at = item.get("expires_at")
                    exp = float(expires_at) if expires_at is not None else None

                    triplet = KnowledgeTriplet(
                        subject=s,
                        predicate=p,
                        object_=o,
                        confidence=conf,
                        created_at=created_at,
                        expires_at=exp,
                    )
                    key = (s, p, o)
                    self._relations[key] = triplet
                    self._by_subject[s].add(key)
                    self._by_predicate[p].add(key)
                    self._by_object[o].add(key)

    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> SemanticKnowledgeGraph:
        """Factory method creating and populating a new graph from a dictionary."""
        graph = cls()
        graph.from_dict(data)
        return graph

    def clear(self) -> None:
        """Remove all relations and reset all indexes."""
        with self._lock:
            self._relations.clear()
            self._by_subject.clear()
            self._by_predicate.clear()
            self._by_object.clear()
