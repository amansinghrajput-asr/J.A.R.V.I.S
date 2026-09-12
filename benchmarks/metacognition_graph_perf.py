"""Performance Benchmarking Suite for Phase 20.3 Semantic Knowledge Graph.

Measures latency and throughput metrics against Phase 20 architectural performance budgets:
- Knowledge Insertion Latency on 1000-node graph (Target: < 100.0 µs)
- Knowledge Lookup Latency on 1000-node graph (Target: < 500.0 µs)
- BFS Shortest Path Traversal Latency (Target: < 1000.0 µs)
- Expiration Cleanup Latency (Target: < 2000.0 µs)
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from app.ai.planner.metacognition.knowledge_graph import SemanticKnowledgeGraph


def benchmark_insertion(node_count: int = 1000) -> float:
    """Measure single relation insertion latency into a growing graph."""
    graph = SemanticKnowledgeGraph()

    t0 = time.perf_counter()
    for i in range(node_count):
        # Create a linked chain: node_0 -> node_1 -> ... -> node_999
        graph.add_relation(f"node_{i}", "links_to", f"node_{(i + 1) % node_count}", confidence=0.9)
    t1 = time.perf_counter()

    avg_us = ((t1 - t0) / node_count) * 1_000_000.0
    return avg_us


def benchmark_lookup(node_count: int = 1000, queries_count: int = 5000) -> float:
    """Measure relation query latency on a 1000-node indexed graph."""
    graph = SemanticKnowledgeGraph()
    for i in range(node_count):
        graph.add_relation(f"node_{i}", "links_to", f"node_{(i + 1) % node_count}", confidence=0.9)

    t0 = time.perf_counter()
    for i in range(queries_count):
        idx = i % node_count
        graph.query(subject=f"node_{idx}", predicate="links_to")
    t1 = time.perf_counter()

    avg_us = ((t1 - t0) / queries_count) * 1_000_000.0
    return avg_us


def benchmark_bfs_traversal(node_count: int = 200) -> float:
    """Measure BFS shortest-path traversal across multi-hop graph."""
    graph = SemanticKnowledgeGraph()
    # Build tree/grid-like graph
    for i in range(node_count):
        graph.add_relation(f"node_{i}", "connected", f"node_{i + 1}")
        if i % 5 == 0 and i + 5 < node_count:
            graph.add_relation(f"node_{i}", "shortcut", f"node_{i + 5}")

    iterations = 500
    t0 = time.perf_counter()
    for _ in range(iterations):
        path = graph.find_path("node_0", "node_50")
        assert path is not None
    t1 = time.perf_counter()

    avg_us = ((t1 - t0) / iterations) * 1_000_000.0
    return avg_us


def benchmark_expiration_cleanup(triplets_count: int = 1000) -> float:
    """Measure purging latency for expired relations on 1000-node graph."""
    graph = SemanticKnowledgeGraph()
    # Add expired relations
    for i in range(triplets_count):
        graph.add_relation(f"exp_s_{i}", "temp_rel", f"exp_o_{i}", valid_duration_sec=0.001)

    time.sleep(0.01)

    t0 = time.perf_counter()
    removed = graph.remove_expired()
    t1 = time.perf_counter()

    assert removed == triplets_count
    elapsed_us = (t1 - t0) * 1_000_000.0
    return elapsed_us


def main() -> None:
    print("=" * 65)
    print("   J.A.R.V.I.S. Phase 20.3 Semantic Knowledge Graph Benchmark    ")
    print("=" * 65)
    print()

    # 1. Insertion Latency
    print("[1/4] Benchmarking Knowledge Insertion Latency (1,000 nodes)...")
    ins_us = benchmark_insertion(node_count=1000)
    ins_budget = 100.0
    ins_pass = ins_us < ins_budget
    ins_status = "PASS" if ins_pass else "FAIL"
    print(f"  Result: {ins_us:.3f} µs (Budget: < {ins_budget} µs) -> {ins_status}\n")

    # 2. Lookup Latency
    print("[2/4] Benchmarking Knowledge Lookup Latency (1,000 nodes)...")
    lookup_us = benchmark_lookup(node_count=1000)
    lookup_budget = 500.0
    lookup_pass = lookup_us < lookup_budget
    lookup_status = "PASS" if lookup_pass else "FAIL"
    print(f"  Result: {lookup_us:.3f} µs (Budget: < {lookup_budget} µs) -> {lookup_status}\n")

    # 3. BFS Traversal Latency
    print("[3/4] Benchmarking BFS Shortest Path Traversal Latency...")
    bfs_us = benchmark_bfs_traversal(node_count=200)
    bfs_budget = 1000.0
    bfs_pass = bfs_us < bfs_budget
    bfs_status = "PASS" if bfs_pass else "FAIL"
    print(f"  Result: {bfs_us:.3f} µs (Budget: < {bfs_budget} µs) -> {bfs_status}\n")

    # 4. Expiration Cleanup Latency
    print("[4/4] Benchmarking Expiration Cleanup Latency (1,000 triplets)...")
    exp_us = benchmark_expiration_cleanup(triplets_count=1000)
    exp_budget = 2000.0
    exp_pass = exp_us < exp_budget
    exp_status = "PASS" if exp_pass else "FAIL"
    print(f"  Result: {exp_us:.3f} µs (Budget: < {exp_budget} µs) -> {exp_status}\n")

    all_passed = ins_pass and lookup_pass and bfs_pass and exp_pass

    print("=" * 65)
    print("                     BENCHMARK SUMMARY                           ")
    print("=" * 65)
    print(f"Knowledge Insertion:    {ins_us:.3f} µs  [{ins_status}]")
    print(f"Knowledge Lookup:       {lookup_us:.3f} µs  [{lookup_status}]")
    print(f"BFS Shortest Path:      {bfs_us:.3f} µs  [{bfs_status}]")
    print(f"Expiration Cleanup:     {exp_us:.3f} µs  [{exp_status}]")
    print("=" * 65)
    if all_passed:
        print("OVERALL STATUS: ALL PERFORMANCE BUDGETS MET [PASS]")
    else:
        print("OVERALL STATUS: AT LEAST ONE BUDGET FAILED [FAIL]")
    print("=" * 65)


if __name__ == "__main__":
    main()
