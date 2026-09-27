"""OpenQASM feature extraction and compact log-runtime model."""

import gzip
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

from tree_runtime import RuntimeModel as TreeRuntimeModel
from probe_model import MAX_PROBE_OPS, ProbeRuntimeModel

CAP_SECONDS = 4 * 60 * 60
FEATURES = (
    "log_qubits", "log_ops", "log_depth", "log_1q", "log_2q",
    "log_multi", "log_cx", "log_swap", "log_measure", "log_reset",
    "twoq_density", "log_custom_calls", "log_control",
)
KNN_FEATURES = FEATURES + (
    "log_ops_per_qubit", "log_2q_per_qubit", "log_depth_per_qubit",
    "depth_per_op", "log_custom_per_qubit", "log_measure_per_qubit",
)
TEXT_GATE_NAMES = (
    "ccx", "cp", "rx", "ry", "rz", "u", "u1", "u2", "u3",
    "h", "x", "t", "tdg", "cz", "swap", "reset", "barrier",
)
TEXT_FEATURES = ("log_qasm_chars",) + tuple("name_" + name for name in TEXT_GATE_NAMES) + ("qasm3",)
KNN_FEATURES += TEXT_FEATURES
_DECL = re.compile(r"\bqreg\s+[\w$]+\s*\[\s*(\d+)\s*\]|\bqubit\s*(?:\[\s*(\d+)\s*\])?")
_GATE_BLOCK = re.compile(r"(?ms)^[ \t]*gate\s+([\w$]+)(?:\([^)]*\))?\s+([^{}]*)\{.*?^[ \t]*\}")
_STMT = re.compile(r"(?m)^[ \t]*(?:if\s*\([^\n]*?\)\s*)?([\w$]+)\s*(?:\([^;\n]*\))?\s+([^;\n]*?)\s*;")
_MEAS_ASSIGN = re.compile(r"(?m)^[ \t]*[A-Za-z_$][\w$]*(?:\s*\[\s*\d+\s*\])?\s*=\s*measure\s+[^;\n]+;")
_CONTROL = re.compile(r"(?m)^[ \t]*(?:if|for|while|switch)\b")
_FAST_NAME = re.compile(r"(?m)^[ \t]*([A-Za-z_$][\w$]*)[ \t]*(?=[( \t])")
_OPERAND = re.compile(r"[\w$]+\s*\[\s*\d+\s*\]|\b[A-Za-z_$][\w$]*\b")
_QREF = re.compile(r"[A-Za-z_$][\w$]*(?:\s*\[\s*\d+\s*\])?")
_TWO_Q = set("cx cy cz ch swap iswap csx cu1 cu3 cp crx cry crz rxx ryy rzz ecr xx yy zz dcx cnot".split())
_MEASURE = {"measure"}
_RESET = {"reset"}


def _add_text_features(qasm_text, features):
    """Capture syntax and gate mix from a bounded QASM sample."""
    sample = qasm_text if len(qasm_text) <= 1_000_000 else qasm_text[:250_000] + "\n" + qasm_text[-250_000:]
    names = Counter(name.lower() for name in _FAST_NAME.findall(sample))
    total = sum(names.values()) or 1
    features["log_qasm_chars"] = math.log1p(len(qasm_text))
    for name in TEXT_GATE_NAMES:
        features["name_" + name] = names[name] / total
    features["qasm3"] = float("OPENQASM 3" in sample[:80])
    return features


def featurize(qasm_text: str) -> dict:
    """Count executed QASM statements and estimate dependency depth."""
    if len(qasm_text) > 200_000_000:
        # QASM exporters can emit hundreds of thousands of gate definitions.
        # Their bodies are declarations; the executable circuit follows the last body.
        last_gate = qasm_text.rfind("\ngate ")
        last_close = qasm_text.find("\n}", last_gate) if last_gate >= 0 else -1
        if last_close >= 0:
            executed = qasm_text[last_close + 2:]
            qubits = sum(int(m.group(1) or m.group(2) or 1) for m in _DECL.finditer(executed))
            names = Counter(_FAST_NAME.findall(executed))
            ignored = {"openqasm", "include", "qreg", "qubit", "creg", "bit", "gate", "barrier", "if", "for", "while", "switch"}
            oneq_names = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "id", "p", "rx", "ry", "rz", "u", "u1", "u2", "u3"}
            oneq = twoq = multiq = measurements = resets = cx_count = swap_count = 0
            for raw_name, count in names.items():
                name = raw_name.lower()
                if name in ignored:
                    continue
                if name == "measure":
                    measurements += count
                elif name == "reset":
                    resets += count
                elif name in oneq_names:
                    oneq += count
                elif name in _TWO_Q:
                    twoq += count
                    cx_count += count if name in {"cx", "cnot"} else 0
                    swap_count += count if name in {"swap", "iswap"} else 0
                else:
                    multiq += count
            measurements += executed.count("= measure ") + executed.count("=measure ")
            control = sum(executed.count("\n" + keyword + " ") for keyword in ("if", "for", "while", "switch"))
            ops = oneq + twoq + multiq
            return _add_text_features(qasm_text, {
                "qubits": qubits, "ops": ops, "depth": round(ops * (0.9 if multiq > ops / 2 else 0.18)),
                "oneq": oneq, "twoq": twoq, "multiq": multiq, "cx": cx_count,
                "swap": swap_count, "measure": measurements, "reset": resets,
                "custom_calls": multiq, "control": control, "twoq_density": twoq / max(1, ops),
            })
    qubits = 0
    custom_arities = {}
    cx_count = swap_count = 0
    used_qubits = set()
    last_layer = {}
    depth = 0
    custom_calls = control = measurements = resets = oneq = twoq = multiq = 0
    sample_stride = 1 if len(qasm_text) < 12_000_000 else max(4, 2 * math.ceil(len(qasm_text) / 10_000_000))
    scheduled = 0
    for match in _DECL.finditer(qasm_text):
        qubits += int(match.group(1) or match.group(2) or 1)
    # Definitions are parsed for arity, then their bodies are omitted from executed counts.
    definitions = []
    for match in _GATE_BLOCK.finditer(qasm_text):
        custom_arities[match.group(1).lower()] = len(_OPERAND.findall(match.group(2)))
        definitions.append((match.start(), match.end()))
    segments = []
    cursor = 0
    for start, end in definitions:
        if cursor < start:
            segments.append((cursor, start))
        cursor = end
    if cursor < len(qasm_text):
        segments.append((cursor, len(qasm_text)))
    if len(qasm_text) > 100_000_000:
        # Count statement names in C-backed regex code on very large circuits.
        # These files have millions of gates, so depth is a coarse workload proxy.
        outside = "".join(qasm_text[start:end] for start, end in segments)
        names = Counter(_FAST_NAME.findall(outside))
        control = sum(outside.count("\n" + keyword + " ") for keyword in ("if", "for", "while", "switch"))
        measurements = outside.count("= measure ") + outside.count("=measure ")
        for name, count in names.items():
            name = name.lower()
            if name in {"openqasm", "include", "qreg", "qubit", "creg", "bit", "gate", "barrier", "if", "for", "while", "switch"}:
                continue
            if name in _MEASURE:
                measurements += count
            elif name in _RESET:
                resets += count
            else:
                custom_calls += count if name in custom_arities else 0
                cx_count += count if name in {"cx", "cnot"} else 0
                swap_count += count if name in {"swap", "iswap"} else 0
                arity = custom_arities.get(name, 2 if name in _TWO_Q else (3 if name in {"ccx", "cswap", "ccz"} else 1))
                if arity == 1:
                    oneq += count
                elif arity == 2:
                    twoq += count
                else:
                    multiq += count
        ops = oneq + twoq + multiq
        depth = round(ops * (0.9 if multiq > ops / 2 else (0.18 if twoq > ops / 5 else 0.1)))
        return _add_text_features(qasm_text, {
            "qubits": qubits, "ops": ops, "depth": depth, "oneq": oneq,
            "twoq": twoq, "multiq": multiq, "cx": cx_count, "swap": swap_count,
            "measure": measurements, "reset": resets, "custom_calls": custom_calls,
            "control": control, "twoq_density": twoq / max(1, ops),
        })
    for start, end in segments:
        control += sum(1 for _ in _CONTROL.finditer(qasm_text, start, end))
        for match in _STMT.finditer(qasm_text, start, end):
            name, argtext = match.group(1).lower(), match.group(2)
            if name in {"qreg", "qubit", "creg", "bit", "gate"}:
                continue
            arity = custom_arities.get(name, 2 if name in _TWO_Q else (3 if name in {"ccx", "cswap", "ccz"} else 1))
            if name in _MEASURE:
                measurements += 1
            elif name in _RESET:
                resets += 1
            else:
                cx_count += name in {"cx", "cnot"}
                swap_count += name in {"swap", "iswap"}
                custom_calls += int(name in custom_arities)
                if arity == 1:
                    oneq += 1
                elif arity == 2:
                    twoq += 1
                else:
                    multiq += 1
            # Sample dependency edges on huge circuits to stay below the parse cap.
            scheduled += 1
            if scheduled % sample_stride:
                continue
            qrefs = [q.strip() for q in _QREF.findall(argtext) if "[" in q or q.strip() in {"q", "qubits"}]
            if not qrefs:
                qrefs = [f"_op{oneq + twoq + multiq + measurements + resets}"]
            used_qubits.update(qrefs)
            layer = 1 + max((last_layer.get(q, 0) for q in qrefs), default=0)
            for q in qrefs:
                last_layer[q] = layer
            depth = max(depth, layer)
        measurements += sum(1 for _ in _MEAS_ASSIGN.finditer(qasm_text, start, end))
    depth = min(scheduled + measurements, depth * sample_stride)

    ops = oneq + twoq + multiq
    return _add_text_features(qasm_text, {
        "qubits": max(qubits, len(used_qubits)), "ops": ops, "depth": depth,
        "oneq": oneq, "twoq": twoq, "multiq": multiq,
        "cx": cx_count,
        "swap": swap_count,
        "measure": measurements, "reset": resets, "custom_calls": custom_calls,
        "control": control,
        "twoq_density": twoq / max(1, ops),
    })


def _vector(features):
    return [
        math.log1p(features[k]) for k in ("qubits", "ops", "depth", "oneq", "twoq", "multiq", "cx", "swap", "measure", "reset")
    ] + [features["twoq_density"], math.log1p(features["custom_calls"]), math.log1p(features["control"])]


def _knn_vector(features):
    q = max(1.0, features["qubits"])
    ops = features["ops"]
    return _vector(features) + [
        math.log1p(ops / q),
        math.log1p(features["twoq"] / q),
        math.log1p(features["depth"] / q),
        features["depth"] / max(1.0, ops),
        math.log1p(features["custom_calls"] / q),
        math.log1p(features["measure"] / q),
    ] + [features.get(name, 0.0) for name in TEXT_FEATURES]


def _knn_log_duration(features, neighbors):
    """Weighted mean log runtime of the nearest circuit feature vectors."""
    vector = _knn_vector(features)
    scale = neighbors["scale"]
    feature_weights = neighbors.get("feature_weights", [1.0] * len(scale))
    distance_power = neighbors.get("distance_power", 2.0)
    distances = []
    for index, (candidate, target) in enumerate(zip(neighbors["vectors"], neighbors["targets"])):
        distance_sq = sum((((a - b) / s) * weight) ** 2
                          for a, b, s, weight in zip(vector, candidate, scale, feature_weights))
        distances.append((distance_sq, index, target))
    distances.sort()
    chosen = distances[: neighbors.get("k", 5)]
    weights = [1.0 / max(distance_sq, 1e-12) ** (distance_power / 2.0) for distance_sq, _, _ in chosen]
    return sum(weight * target for weight, (_, _, target) in zip(weights, chosen)) / sum(weights)


class RuntimeModel:
    def __init__(self, artifacts_dir="artifacts"):
        path = Path(artifacts_dir) / "runtime_model.json"
        if not path.exists():
            path = Path(__file__).resolve().parent / artifacts_dir / "runtime_model.json"
        self.model = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        tree_dir = None if Path(artifacts_dir) == Path("artifacts") else artifacts_dir
        self.tree_model = TreeRuntimeModel(artifacts_dir=tree_dir)
        self.probe_model = ProbeRuntimeModel(artifacts_dir=tree_dir)
        timeout_path = Path(artifacts_dir) / "timeout-classifier.json.gz"
        if not timeout_path.exists():
            timeout_path = Path(__file__).resolve().parent / artifacts_dir / "timeout-classifier.json.gz"
        with gzip.open(timeout_path, "rt", encoding="utf-8") as stream:
            self.timeout_model = json.load(stream)
        if self.timeout_model.get("kind") != "knn-timeout-v1":
            raise ValueError("Unsupported timeout classifier artifact")

    def featurize(self, qasm_text: str) -> dict:
        start = time.perf_counter()
        knn = featurize(qasm_text)
        tree = self.tree_model.featurize(qasm_text)
        if knn["ops"] >= MAX_PROBE_OPS or time.perf_counter() - start > 8.0:
            probe = {"usable": False, "reason": "large circuit budget"}
        else:
            probe = self.probe_model.featurize(qasm_text, deadline=start + 12.0)
        return {"knn": knn, "tree": tree, "probe": probe}

    def _predict_knn(self, features: dict, threshold: int) -> float:
        if self.model and self.model.get("kind") in {"knn-v1", "knn-v2"} and str(threshold) in self.model["by_threshold"]:
            log_seconds = _knn_log_duration(features, self.model["by_threshold"][str(threshold)])
        elif self.model:
            x = _vector(features)
            model = self.model["by_threshold"].get(str(threshold), self.model["fallback"])
            if self.model.get("kind") in {"knn-v1", "knn-v2"}:
                model = self.model["fallback"]
            log_seconds = model["intercept"] + sum(a * b for a, b in zip(model["coefficients"], x))
        else:
            # Usable cold-start estimate until train_model.py creates the artifact.
            x = _vector(features)
            log_seconds = -5.0 + 0.13 * x[0] + 0.85 * x[1] + 0.75 * x[2] + 0.35 * math.log(max(threshold, 1) / 16)
        if log_seconds >= math.log10(CAP_SECONDS):
            return float(CAP_SECONDS)
        return float(max(0.001, 10 ** log_seconds))

    def _timeout_probability(self, features: dict, threshold: int) -> float:
        """Estimate timeout probability from labeled neighboring circuits."""
        bank = self.timeout_model["by_threshold"].get(str(threshold))
        if bank is None:
            return 0.0
        vector = _knn_vector(features)
        distances = []
        for candidate, timeout in zip(bank["vectors"], bank["timeout"]):
            distance_sq = sum((((a - b) / scale) * weight) ** 2
                              for a, b, scale, weight in zip(
                                  vector, candidate, bank["scale"], bank["feature_weights"]))
            distances.append((distance_sq, timeout))
        distances.sort(key=lambda item: item[0])
        neighbors = distances[:self.timeout_model["k"]]
        power = self.timeout_model["distance_power"] / 2.0
        weights = [1.0 / max(distance_sq, 1e-8) ** power for distance_sq, _ in neighbors]
        return sum(weight * timeout for weight, (_, timeout) in zip(weights, neighbors)) / sum(weights)

    def predict(self, features: dict, threshold: int) -> float:
        """Use the learned timeout classifier, then blend runtime estimates."""
        if self._timeout_probability(features["knn"], threshold) >= self.timeout_model["decision_probability"]:
            return float(CAP_SECONDS)
        knn_seconds = self._predict_knn(features["knn"], threshold)
        tree_seconds = self.tree_model.predict(features["tree"], threshold)
        current_seconds = math.sqrt(knn_seconds * tree_seconds)
        probe_seconds = self.probe_model.predict(features["knn"], features["probe"], threshold)
        if probe_seconds is None:
            return float(min(CAP_SECONDS, current_seconds))
        return float(min(CAP_SECONDS, math.sqrt(current_seconds * probe_seconds)))
