"""Portable ExtraTrees runtime inference from a bounded circuit probe."""
import gzip
import json
import math
from pathlib import Path

from probe_runtime import featurize_text

CAP_SECONDS = 14400.0
MAX_PROBE_OPS = 1_000_000
STATIC = (
    "qubits", "ops", "depth", "oneq", "twoq", "multiq", "cx", "swap",
    "measure", "reset", "custom_calls", "control", "twoq_density", "log_qasm_chars",
    "name_ccx", "name_cp", "name_rx", "name_ry", "name_rz", "name_u", "name_u1",
    "name_u2", "name_u3", "name_h", "name_x", "name_t", "name_tdg", "name_cz",
    "name_swap", "name_reset", "name_barrier", "qasm3",
)
STRUCTURAL = (
    "span_mean", "span_max", "bound_max_log2", "bound_mean_log2",
    "struct_logcost_16", "struct_frac_bonds_ge_16", "struct_logcost_64",
    "struct_frac_bonds_ge_64", "struct_logcost_512", "struct_frac_bonds_ge_512",
)
PROBE = (
    "probe_ok", "probe_frac_ops_done", "probe_twoq_done",
    "probe_frac_bonds_saturated", "probe_mean_log2_bond", "probe_max_entropy",
    "probe_discarded", "probe_log_discarded", "probe_trunc_frac",
    "probe_first_sat_frac", "probe_log_flops",
)
FEATURE_NAMES = STATIC + ("log2_threshold",) + STRUCTURAL + PROBE


def vector(static, probe, threshold):
    """Use exactly the feature order recorded in the training artifact."""
    return ([float(static.get(key, 0.0) or 0.0) for key in STATIC] +
            [math.log2(max(1, int(threshold)))] +
            [float(probe.get(key, 0.0) or 0.0) for key in STRUCTURAL + PROBE])


class ProbeRuntimeModel:
    def __init__(self, artifacts_dir=None):
        base = Path(artifacts_dir) if artifacts_dir is not None else Path(__file__).resolve().parent / "artifacts"
        with gzip.open(base / "probe-extra-trees.json.gz", "rt", encoding="utf-8") as stream:
            self.artifact = json.load(stream)
        if self.artifact.get("kind") != "probe-extra-trees-v1" or tuple(self.artifact.get("features", ())) != FEATURE_NAMES:
            raise ValueError("Unsupported probe artifact schema")

    def featurize(self, qasm_text, deadline=None):
        try:
            return featurize_text(qasm_text, chi=16, parse_budget=4.0, probe_budget=1.0, deadline=deadline)
        except Exception:
            return {"usable": False, "reason": "probe extraction failed"}

    def predict(self, static, probe, threshold):
        if not probe.get("usable"):
            return None
        x = vector(static, probe, threshold)
        total = 0.0
        for tree in self.artifact["trees"]:
            node = 0
            while tree["feature"][node] >= 0:
                feature = tree["feature"][node]
                node = tree["left"][node] if x[feature] <= tree["threshold"][node] else tree["right"][node]
            total += tree["value"][node]
        log_seconds = total / len(self.artifact["trees"])
        return float(min(CAP_SECONDS, max(0.001, 10 ** log_seconds)))
