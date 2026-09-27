"""Compact circuit features (26) from the team's v3 structural extractor.

Values are only reported when the extractor marks the underlying quantity exact; otherwise they are missing (NaN)
and the network's preprocessing imputes them. This mirrors how the training features were built.
"""
import math

from .config import COMPACT_FEATURES
from .extractor import extract_features
from .graph import graph_features

NAN = float('nan')


def compact_features(qasm_text: str, budget_s: float) -> dict:
    """Map reliable extractor counts, graph statistics, and depth to model inputs.

    Values without an exactness flag remain NaN and are imputed by the network.
    """
    res = extract_features(qasm_text, budget_s=budget_s)
    f = {k: NAN for k in COMPACT_FEATURES}
    n = res.get('n_qubits')
    if n is not None:
        f['n_qubits'] = n
    if res.get('counts_exact'):
        for k in ('unitary_count', 'twoq_count', 'multiq_count', 'measurement_count', 'reset_count'):
            f[k] = res[k]
        f['generic_entanglement_ratio_twoq'] = res['twoq_count'] / max(1, res['unitary_count'])
    if res.get('graph_exact') and n:
        edges = res.get('twoq_edges') or []
        g = graph_features(n, edges)
        for k in ('unique_edges', 'density', 'degree_max', 'components', 'native_max_cut', 'native_mean_cut',
                  'native_mean_span', 'rcm_max_cut', 'rcm_mean_cut', 'rcm_mean_span'):
            f['twoq_' + k] = g[k]
        f['generic_program_communication_twoq'] = g['density']
        weights = [e[2] for e in edges]
        total = float(sum(weights))
        if total:
            p = [w / total for w in weights]
            f['twoq_reuse'] = total / len(weights)
            f['twoq_top_pair_share'] = max(p)
            f['twoq_pair_mass_entropy'] = -sum(x * math.log2(x) for x in p)
        else:
            f['twoq_reuse'] = f['twoq_top_pair_share'] = f['twoq_pair_mass_entropy'] = 0.0
    if res.get('depth_exact'):
        depth, uc, tq = res['gate_depth'], res['unitary_count'], res['twoq_count']
        f['gate_depth'], f['twoq_depth'] = depth, res['twoq_depth']
        f['generic_parallelism'] = max(0.0, (uc / depth - 1) / (n - 1)) if depth and n > 1 else 0.0
        f['generic_liveness'] = res['active_qubit_layers'] / (n * depth) if n and depth else 0.0
        f['generic_critical_depth_twoq'] = res['critical_path_twoq_count'] / tq if tq else 0.0
    return f
