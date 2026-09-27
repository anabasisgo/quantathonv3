# Vendored from the team feature pipeline (data_first/feature_v3/graph_features.py), unchanged except this header.
"""Deterministic graph proxies; no claim about the simulator's internal order.

Edges contain [qubit_a, qubit_b, operation multiplicity]. RCM uses an unweighted
interaction graph. Cuts and spans retain multiplicity. Multi-qubit clique edges
must be labeled separately from true two-qubit operations by the caller.
"""
from collections import deque
import math


def reverse_cuthill_mckee(n, edges):
    adjacency = [set() for _ in range(n)]
    for a, b, weight in edges:
        if a != b and weight > 0:
            adjacency[a].add(b)
            adjacency[b].add(a)
    degree = list(map(len, adjacency))
    seen = set()
    traversal = []
    for root in sorted(range(n), key=lambda q: (degree[q], q)):
        if root in seen:
            continue
        seen.add(root)
        queue = deque([root])
        while queue:
            q = queue.popleft()
            traversal.append(q)
            for neighbor in sorted(adjacency[q] - seen, key=lambda q: (degree[q], q)):
                seen.add(neighbor)
                queue.append(neighbor)
    return list(reversed(traversal))


def quantile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    x = (len(values) - 1) * p
    i = int(x)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)


def ordering_features(n, edges, order):
    position = {q: p for p, q in enumerate(order)}
    delta = [0] * (n + 1)
    total = span_sum = max_span = distant = 0
    for a, b, weight in edges:
        left, right = sorted((position[a], position[b]))
        span = right - left
        delta[left] += weight
        delta[right] -= weight
        total += weight
        span_sum += weight * span
        max_span = max(max_span, span)
        distant += weight * (span > n / 4)
    cuts = []
    running = 0
    for i in range(max(0, n - 1)):
        running += delta[i]
        cuts.append(running)
    mean = sum(cuts) / max(1, len(cuts))
    return {
        'max_cut': max(cuts, default=0), 'mean_cut': mean,
        'p50_cut': quantile(cuts, .5), 'p90_cut': quantile(cuts, .9),
        'std_cut': math.sqrt(sum((x - mean) ** 2 for x in cuts) / max(1, len(cuts))),
        'max_span': max_span, 'mean_span': span_sum / total if total else 0.0,
        'long_span_fraction': distant / total if total else 0.0,
    }


def graph_features(n, edges):
    adjacency = [set() for _ in range(n)]
    for a, b, weight in edges:
        if a == b or not (0 <= a < n and 0 <= b < n) or weight <= 0:
            raise ValueError('Invalid interaction edge')
        adjacency[a].add(b)
        adjacency[b].add(a)
    degree = list(map(len, adjacency))
    degree_sum = sum(degree)
    components = 0
    visited = set()
    for root in range(n):
        if root in visited:
            continue
        components += 1
        stack = [root]
        visited.add(root)
        while stack:
            q = stack.pop()
            for other in adjacency[q] - visited:
                visited.add(other)
                stack.append(other)
    result = {
        'unique_edges': degree_sum // 2,
        'interaction_multiplicity': sum(e[2] for e in edges),
        'degree_mean': degree_sum / n if n else 0.0,
        'degree_max': max(degree, default=0),
        'density': degree_sum / (n * (n - 1)) if n > 1 else 0.0,
        'components': components, 'isolated_qubits': degree.count(0),
        'degree_mass_entropy': -sum((d / degree_sum) * math.log2(d / degree_sum)
                                    for d in degree if d) if degree_sum else 0.0,
    }
    for label, order in [('native', list(range(n))), ('rcm', reverse_cuthill_mckee(n, edges))]:
        result.update({label + '_' + k: v for k, v in ordering_features(n, edges, order).items()})
    return result
