"""featurize: QASM text -> the 44 model features (plus stage timings).

Stages: compact structural features -> probe parse -> structural bond bound -> truncated MPS probe.
Every stage is bounded; a stage that cannot run leaves its features missing, and the network imputes them.
"""
import time

from . import config
from .compact import compact_features
from .probe import bond_bound, simulate
from .qasm_ops import parse

NAN = float('nan')


def featurize(qasm_text: str) -> dict:
    """Combine compact, bound, and MPS features, recording elapsed time per stage.

    Missing stages return NaNs for network imputation. The probe runs only when
    compact extraction finishes before the configured watchdog threshold.
    """
    t0 = time.perf_counter()
    feats = compact_features(qasm_text, budget_s=config.EXTRACTOR_BUDGET_S)
    t1 = time.perf_counter()
    timings = {'t_compact': t1 - t0}
    if t1 - t0 < config.PROBE_SKIP_AFTER_S:
        C = parse(qasm_text, config.PROBE_MAX_CHARS, config.PROBE_MAX_OPS)
        t2 = time.perf_counter()
        feats.update(bond_bound(C))
        t3 = time.perf_counter()
        feats.update(simulate(C))
        t4 = time.perf_counter()
        timings.update(t_probe_parse=t2 - t1, t_bond_bound=t3 - t2, t_probe_sim=t4 - t3)
    else:  # watchdog: no time left for the probe
        feats.update({k: NAN for k in config.BOUND_FEATURES + config.PROBE_FEATURES})
    feats['_timings'] = timings
    return feats
