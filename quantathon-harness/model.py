"""Predict simulator seconds from QASM structure and simulator threshold.

Inference uses only the standard library. The fitted trees are stored in
artifacts/runtime-model.json next to this module. Filenames, comments, and
algorithm names are never model inputs.

The artifact records its fitted model, feature schema, and validation results.
Version 1 forests and version 2 routed/calibrated models are supported. All
regressors predict log10(seconds); timeout routing returns the four-hour cap.
The original model and feature cache are preserved in reports/model-improvement.

Run the nested improvement experiment from the repository root, using approved
training dependencies (scikit-learn and xgboost-cpu):
    uv run --with scikit-learn==1.9.1 --with xgboost-cpu==3.4.1 \
        python quantathon-harness/train_improved.py

The simpler original three-model training comparison is also available:
    uv run --with scikit-learn==1.9.1 python quantathon-harness/model.py \
        --train --circuits training_circuits --labels /path/to/runtime-data.csv

Training compares models with circuit-grouped cross-validation, then fits the
selected model on all labels and writes artifacts/runtime-model.json next to
this module. --output selects another path; RuntimeModel(artifacts_dir) loads
runtime-model.json from a custom directory.
"""

import ast
import json
import math
import re
from array import array
from collections import Counter
from pathlib import Path

CAP_SECONDS = 4 * 60 * 60
SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "runtime-model.json"
_SAMPLE_CHARS = 1_000_000
_SAMPLE_WINDOWS = 16
_DEFINITION_LIMIT = 2048
_NAME = r"[A-Za-z_]\w*"
_COMMENTS = re.compile(r'"[^"\n]*"|//[^\n]*|/\*.*?\*/', re.S)
_DEFINITION = re.compile(
    rf"\bgate\s+({_NAME})\s*(?:\([^{{}}]*?\))?\s*([^{{}}]*?)\{{([^{{}}]*)\}}"
)
_DECLARATION = re.compile(
    rf"\bqreg\s+({_NAME})\s*\[\s*(\d+)\s*\]\s*;"
    rf"|\bqubit\s*(?:\[\s*(\d+)\s*\])?\s+({_NAME})\s*;"
)
_STATEMENT = re.compile(r"[^;{}]+[;{}]")
_OPERATION = re.compile(rf"^({_NAME})\s*(?:\((.*)\))?\s+(.+)$", re.S)
_OPERAND = re.compile(rf"^({_NAME})(?:\[\s*(\d+)\s*\])?$")
_CONDITION = re.compile(r"^if\s*\([^)]*\)\s*")
_IGNORED = {"OPENQASM", "include", "qreg", "creg", "qubit", "bit", "opaque",
            "input", "output", "const", "let", "barrier"}
_GATES = (
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "id",
    "rx", "ry", "rz", "p", "u", "u1", "u2", "u3", "cx", "cy", "cz",
    "ch", "cp", "crx", "cry", "crz", "cu1", "cu3", "cu", "swap",
    "iswap", "cswap", "ccx", "ccz", "rccx", "rxx", "ryy", "rzz", "rzx",
    "ecr", "dcx", "reset", "measure", "other",
)
_KNOWN = frozenset(_GATES) - {"other"}
_DIAGONAL = frozenset(("z", "s", "sdg", "t", "tdg", "rz", "p", "u1",
                       "cz", "cp", "crz", "cu1", "rzz", "ccz"))
_CLIFFORD = frozenset(("h", "x", "y", "z", "s", "sdg", "sx", "sxdg",
                       "id", "cx", "cy", "cz", "swap", "dcx", "ecr"))
_ROTATIONS = frozenset(("rx", "ry", "rz", "p", "u1", "u2", "u3", "u",
                        "cp", "crx", "cry", "crz", "rxx", "ryy", "rzz"))
FEATURE_NAMES = (
    "n_qubits", "n_active", "n_registers", "n_ops", "n_1q", "n_2q", "n_3q",
    "n_measure", "n_reset", "n_conditional", "n_definitions", "definition_ops",
    "expanded_ops", "n_custom", "n_diagonal", "n_clifford", "n_rotation",
    "n_t", "depth", "two_qubit_depth", "parallelism", "entangling_fraction",
    "diagonal_fraction", "clifford_fraction", "rotation_fraction", "t_fraction",
    "measurement_fraction", "custom_fraction", "ops_per_qubit", "depth_per_op",
    "n_edges", "edge_density", "mean_degree", "max_degree", "degree_std",
    "n_components", "largest_component", "mean_span", "max_span", "adjacent_fraction",
    "pair_reuse", "qubit_load_max", "qubit_load_cv", "gate_entropy", "n_gate_types",
    "parameter_fraction", "mean_parameter_chars", "sample_fraction",
) + tuple("gate_" + gate for gate in _GATES) + ("threshold", "log2_threshold",)
BASE_FEATURE_NAMES = FEATURE_NAMES
DETAIL_FEATURE_NAMES = (
    "reset_fraction", "operations_after_measurement_fraction", "expanded_depth",
    "expanded_two_qubit_ops", "expanded_three_qubit_ops", "expanded_rotation_ops",
    "expanded_clifford_fraction", "expanded_nonclifford_fraction",
    "expanded_edge_density", "expanded_max_degree", "expanded_mean_degree",
    "expanded_max_cut", "expanded_mean_cut", "custom_profile_coverage",
    "numeric_angle_fraction", "half_pi_angle_fraction", "quarter_pi_angle_fraction",
    "small_angle_fraction", "mean_absolute_angle", "reset_per_active_qubit",
    "entangling_segment_cv", "three_qubit_segment_cv", "reset_segment_cv",
    "measurement_position_mean", "three_qubit_fraction", "expanded_ops_per_depth",
    "reset_entangling_product", "active_qubit_log_ops",
)
FEATURE_NAMES = BASE_FEATURE_NAMES[:-2] + DETAIL_FEATURE_NAMES + BASE_FEATURE_NAMES[-2:]


def _sample_program(program, max_chars=_SAMPLE_CHARS):
    """Evenly spaced complete-statement windows, including both ends.

    This bounds Python-level parsing work on circuits with millions of gates.
    Estimated additive counts are scaled by the exact semicolon count. Depth
    is a structural estimate (conditionals are treated as potentially executed).
    """
    if len(program) <= max_chars:
        return [program], 1.0
    width = max_chars // _SAMPLE_WINDOWS
    windows = []
    for i in range(_SAMPLE_WINDOWS):
        start = i * (len(program) - width) // (_SAMPLE_WINDOWS - 1)
        end = start + width
        if start:
            start = program.find(";", start, end) + 1
            if start == 0:
                continue
        if end < len(program):
            end = program.rfind(";", start, end) + 1
        if end > start:
            windows.append(program[start:end])
    sampled = sum(window.count(";") for window in windows)
    return windows, max(1.0, program.count(";") / max(sampled, 1))


def _gate_size(name, definitions, cache, visiting):
    """Bounded expansion of custom gates into an approximate primitive count."""
    if name in cache:
        return cache[name]
    if name in visiting or len(visiting) >= 16 or name not in definitions:
        return 1.0
    visiting.add(name)
    windows, scale = _sample_program(definitions[name])
    count = 0.0
    for window in windows:
        for match in _STATEMENT.finditer(window):
            op = _OPERATION.match(match[0][:-1].strip())
            if op and op[1] not in _IGNORED:
                count += _gate_size(op[1], definitions, cache, visiting)
    visiting.remove(name)
    cache[name] = max(1.0, min(count * scale, 1e12))
    return cache[name]


def _numeric_angle(expression, cache):
    """Evaluate bounded arithmetic only; never execute QASM or Python code."""
    if expression in cache:
        return cache[expression]
    value = None
    if len(expression) <= 96:
        try:
            tree = ast.parse(expression, mode="eval")
            if len(list(ast.walk(tree))) > 32:
                raise ValueError("Expression too large")

            def visit(node):
                if isinstance(node, ast.Expression):
                    return visit(node.body)
                if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                    return float(node.value)
                if isinstance(node, ast.Name) and node.id == "pi":
                    return math.pi
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                    return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
                if isinstance(node, ast.BinOp):
                    a, b = visit(node.left), visit(node.right)
                    if isinstance(node.op, ast.Add): return a + b
                    if isinstance(node.op, ast.Sub): return a - b
                    if isinstance(node.op, ast.Mult): return a * b
                    if isinstance(node.op, ast.Div): return a / b
                    if isinstance(node.op, ast.Pow) and abs(b) <= 8 and abs(a) <= 1e6:
                        return a ** b
                raise ValueError("Unsupported expression")

            result = visit(tree)
            if isinstance(result, (float, int)) and math.isfinite(result) and abs(result) < 1e12:
                value = float(result)
        except (ValueError, SyntaxError, OverflowError, ZeroDivisionError, RecursionError):
            pass
    if len(cache) < 8192:
        cache[expression] = value
    return value


def _gate_profile(name, definitions, formals, cache, visiting, budget):
    """Summarize custom gate bodies once, with bounded recursion and work.

    Depth composes summaries conservatively at each call. It is an estimate,
    not a full dependency-preserving expansion. Symbolic angles are not bound.
    """
    if name in cache:
        return cache[name]
    if name not in definitions or name in visiting or len(visiting) >= 16 or budget[0] <= 0:
        return None
    visiting.add(name)
    indexes = {q.strip(): i for i, q in enumerate(formals[name])}
    counts, levels, edges = Counter(), {}, set()
    windows, scale = _sample_program(definitions[name], 200_000)
    complete = scale == 1
    for window in windows:
        for match in _STATEMENT.finditer(window):
            budget[0] -= 1
            if budget[0] < 0:
                complete = False
                break
            op = _OPERATION.match(match[0][:-1].strip())
            if not op or op[1] in _IGNORED:
                continue
            qubits = [indexes[q.strip()] for q in op[3].split(",") if q.strip() in indexes]
            if not qubits:
                continue
            child = _gate_profile(op[1], definitions, formals, cache, visiting, budget)
            if child is None:
                counts[op[1].lower() if op[1].lower() in _KNOWN else "other"] += 1
                counts[f"arity_{min(len(qubits), 3)}"] += 1
                depth = 1
                child_edges = [(0, j) for j in range(1, len(qubits))]
                complete = complete and op[1] not in definitions
            else:
                counts.update(child["counts"])
                depth = child["depth"]
                child_edges = child["edges"]
                complete = complete and child["complete"]
            level = max(levels.get(q, 0) for q in qubits) + depth
            for q in qubits:
                levels[q] = level
            for a, b in child_edges:
                if max(a, b) < len(qubits):
                    edges.add(tuple(sorted((qubits[a], qubits[b]))))
        if budget[0] < 0:
            break
    visiting.remove(name)
    profile = {"counts": Counter({k: v * scale for k, v in counts.items()}),
               "depth": max(levels.values(), default=1) * scale,
               "edges": edges, "complete": complete}
    cache[name] = profile
    return profile


class RuntimeModel:
    def __init__(self, artifacts_dir=None, *, load_model=True):
        """Load weights relative to this module, or from an explicit directory.

        load_model=False permits offline feature extraction and training before
        an artifact exists. Prediction requires a fitted model.
        """
        self.model = None
        if not load_model:
            return
        artifact = (Path(artifacts_dir) / "runtime-model.json"
                    if artifacts_dir is not None else DEFAULT_ARTIFACT_PATH)
        try:
            self.model = json.loads(artifact.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Model artifact not found: {artifact}. "
                "Run model.py --train --labels <runtime-data.csv> to train it, "
                "or supply artifacts_dir for an existing model."
            ) from error
        if (self.model.get("schema_version") not in (1, 2)
                or not self.model.get("features")
                or not set(self.model["features"]).issubset(FEATURE_NAMES)):
            raise ValueError("Model artifact uses an incompatible feature schema")

    def featurize(self, qasm_text: str) -> dict:
        """Parse QASM 2/3 declarations, operations, and interaction structure.

        Gate bodies are excluded from outer-operation counts. Bounded recursive
        summaries contribute separate primitive counts, depth estimates, and
        interaction statistics when called. Indexed and whole-register operands,
        QASM 3 measurement assignments, resets, and if statements are supported.
        Large programs use deterministic sampling; no circuit is simulated.
        """
        if not isinstance(qasm_text, str):
            raise TypeError("qasm_text must be a string")
        if "//" in qasm_text or "/*" in qasm_text:
            qasm_text = _COMMENTS.sub(
                lambda m: m[0] if m[0].startswith('"') else " ", qasm_text
            )
        definitions, formals = {}, {}
        definition_stats = [0, 0]

        def remove_definition(match):
            definition_stats[0] += 1
            definition_stats[1] += match[3].count(";")
            if len(definitions) < _DEFINITION_LIMIT:
                definitions[match[1]] = match[3]
                formals[match[1]] = match[2].strip().split(",")
            return " "

        program = _DEFINITION.sub(remove_definition, qasm_text)
        registers = {}
        n_qubits = 0
        for match in _DECLARATION.finditer(program):
            name = match[1] or match[4]
            size = int(match[2] or match[3] or 1)
            if name not in registers:
                registers[name] = (n_qubits, size)
                n_qubits += size
        # Sparse dictionaries avoid allocating by untrusted register sizes.
        levels, two_levels, loads = {}, {}, Counter()
        pairs = Counter()
        counts, gates = Counter(), Counter()
        expansion_cache = {}
        profiles, profile_budget, angle_cache = {}, [100_000], {}
        expanded_counts, expanded_levels, expanded_pairs = Counter(), {}, set()
        angle_counts = Counter()
        segments = {key: [0] * 16 for key in ("entangling", "three_qubit", "reset")}
        seen_measurement = False
        after_measurement = profile_calls = complete_calls = measure_positions = 0
        windows, scale = _sample_program(program)
        span_sum = max_span = adjacent = entangling = 0
        parameter_chars = parameter_ops = 0
        for window_index, window in enumerate(windows):
            for match in _STATEMENT.finditer(window):
                statement = match[0][:-1].strip()
                if not statement:
                    continue
                condition = _CONDITION.match(statement)
                if condition:
                    counts["n_conditional"] += 1
                    statement = statement[condition.end():].strip()
                    if not statement:
                        continue
                # QASM 3 uses classical_target = measure quantum_operand.
                if re.search(r"\bmeasure\b", statement):
                    statement = statement[statement.index("measure"):]
                op = _OPERATION.match(statement)
                if not op or op[1] in _IGNORED:
                    continue
                name, params, operands = op.groups()
                operands = operands.split("->", 1)[0].strip()
                operand_groups = []
                for operand in operands.split(","):
                    ref = _OPERAND.match(operand.strip())
                    if not ref or ref[1] not in registers:
                        continue
                    offset, size = registers[ref[1]]
                    if ref[2] is not None:
                        index = int(ref[2])
                        if index < size:
                            operand_groups.append([offset + index])
                    else:
                        operand_groups.append(list(range(offset, offset + min(size, 4096))))
                if not operand_groups:
                    continue
                width = max(map(len, operand_groups))
                if any(len(group) not in (1, width) for group in operand_groups):
                    continue
                normalized = name.lower()
                known_name = normalized if normalized in _KNOWN else "other"
                expanded = _gate_size(name, definitions, expansion_cache, set())
                profile = _gate_profile(name, definitions, formals, profiles, set(), profile_budget)
                position = (window_index + match.start() / max(len(window), 1)) / len(windows)
                segment = min(15, int(position * 16))
                angles = [] if params is None else [_numeric_angle(p.strip(), angle_cache) for p in params.split(",")]
                for k in range(width):
                    qubits = tuple(dict.fromkeys(group[k % len(group)] for group in operand_groups))
                    arity = len(qubits)
                    if not arity:
                        continue
                    counts["n_ops"] += 1
                    counts["expanded_ops"] += expanded
                    gates[known_name] += 1
                    counts["n_custom"] += name in definitions or known_name == "other"
                    if profile is not None:
                        profile_calls += 1
                        complete_calls += profile["complete"]
                        expanded_counts.update(profile["counts"])
                        detail_depth = profile["depth"]
                        detail_edges = profile["edges"]
                    else:
                        expanded_counts[known_name] += 1
                        if normalized not in ("measure", "reset"):
                            expanded_counts[f"arity_{min(arity, 3)}"] += 1
                        detail_depth = 1
                        detail_edges = [(0, j) for j in range(1, arity)]
                    detail_level = max(expanded_levels.get(q, 0) for q in qubits) + detail_depth
                    for q in qubits:
                        expanded_levels[q] = detail_level
                    for a, b in detail_edges:
                        if max(a, b) < arity:
                            expanded_pairs.add(tuple(sorted((qubits[a], qubits[b]))))
                    after_measurement += seen_measurement and normalized != "measure"
                    if normalized == "measure":
                        seen_measurement = True
                        measure_positions += position
                    segments["entangling"][segment] += arity >= 2
                    segments["three_qubit"][segment] += arity >= 3
                    segments["reset"][segment] += normalized == "reset"
                    for angle in angles:
                        angle_counts["total"] += 1
                        if angle is not None:
                            angle_counts["numeric"] += 1
                            half = angle / (math.pi / 2)
                            quarter = angle / (math.pi / 4)
                            angle_counts["half_pi"] += abs(half - round(half)) < 1e-7
                            angle_counts["quarter_pi"] += abs(quarter - round(quarter)) < 1e-7
                            angle_counts["small"] += abs(angle) < 1e-3
                            angle_counts["absolute"] += abs(angle)
                    if normalized in ("measure", "reset"):
                        counts["n_" + normalized] += 1
                    else:
                        counts["n_" + str(min(arity, 3)) + "q"] += 1
                    counts["n_diagonal"] += normalized in _DIAGONAL
                    counts["n_clifford"] += normalized in _CLIFFORD
                    counts["n_rotation"] += normalized in _ROTATIONS
                    counts["n_t"] += normalized in ("t", "tdg")
                    if params is not None:
                        parameter_ops += 1
                        parameter_chars += len(params)
                    level = max(levels.get(q, 0) for q in qubits) + 1
                    for q in qubits:
                        levels[q] = level
                        loads[q] += 1
                    if arity >= 2 and normalized not in ("measure", "reset"):
                        entangling += 1
                        two_level = max(two_levels.get(q, 0) for q in qubits) + 1
                        for q in qubits:
                            two_levels[q] = two_level
                        span = max(qubits) - min(qubits)
                        span_sum += span
                        max_span = max(max_span, span)
                        adjacent += span == 1
                        # Star edges bound work for wide custom gates.
                        for q in qubits[1:]:
                            pairs[tuple(sorted((qubits[0], q)))] += 1

        features = {key: float(value * scale) for key, value in counts.items()}
        features.update({"n_qubits": n_qubits, "n_registers": len(registers),
                         "n_active": len(loads), "n_definitions": definition_stats[0],
                         "definition_ops": definition_stats[1], "sample_fraction": 1 / scale})
        total = max(counts["n_ops"], 1)
        depth = max(levels.values(), default=0)
        features["depth"] = depth * scale
        features["two_qubit_depth"] = max(two_levels.values(), default=0) * scale
        features["parallelism"] = sum(loads.values()) / max(depth, 1)
        features["ops_per_qubit"] = counts["n_ops"] * scale / max(n_qubits, 1)
        features["depth_per_op"] = depth / total
        features["entangling_fraction"] = entangling / total
        for kind, key in (("diagonal", "diagonal"), ("clifford", "clifford"),
                          ("rotation", "rotation"), ("t", "t"),
                          ("measurement", "measure"), ("custom", "custom")):
            features[kind + "_fraction"] = counts["n_" + key] / total
        features["parameter_fraction"] = parameter_ops / total
        features["mean_parameter_chars"] = parameter_chars / max(parameter_ops, 1)
        features["gate_entropy"] = -sum((c / total) * math.log2(c / total) for c in gates.values())
        features["n_gate_types"] = len(gates)
        for gate in _GATES:
            features["gate_" + gate] = gates[gate] / total

        neighbors = {q: set() for q in loads}
        for a, b in pairs:
            neighbors[a].add(b)
            neighbors[b].add(a)
        degrees = [len(neighbors[q]) for q in neighbors]
        mean_degree = sum(degrees) / max(n_qubits, 1)
        features["n_edges"] = len(pairs)
        features["edge_density"] = 2 * len(pairs) / max(n_qubits * (n_qubits - 1), 1)
        features["mean_degree"] = mean_degree
        features["max_degree"] = max(degrees, default=0)
        features["degree_std"] = math.sqrt(max(0, sum(d*d for d in degrees) / max(n_qubits, 1) - mean_degree**2))
        unseen = set(neighbors)
        components = []
        while unseen:
            todo = [unseen.pop()]
            size = 0
            while todo:
                q = todo.pop()
                size += 1
                new = neighbors[q] & unseen
                unseen.difference_update(new)
                todo.extend(new)
            components.append(size)
        features["n_components"] = len(components) + max(0, n_qubits - len(loads))
        features["largest_component"] = max(components, default=min(n_qubits, 1))
        features["mean_span"] = span_sum / max(entangling, 1)
        features["max_span"] = max_span
        features["adjacent_fraction"] = adjacent / max(entangling, 1)
        features["pair_reuse"] = sum(pairs.values()) * scale / max(len(pairs), 1)
        mean_load = sum(loads.values()) / max(n_qubits, 1)
        features["qubit_load_max"] = max(loads.values(), default=0) * scale
        variance = sum(c*c for c in loads.values()) / max(n_qubits, 1) - mean_load**2
        features["qubit_load_cv"] = math.sqrt(max(variance, 0)) / max(mean_load, 1)
        primitive_total = max(sum(expanded_counts[g] for g in _GATES), 1)
        primitive_clifford = sum(expanded_counts[g] for g in _CLIFFORD)
        features.update({
            "reset_fraction": counts["n_reset"] / total,
            "three_qubit_fraction": counts["n_3q"] / total,
            "operations_after_measurement_fraction": after_measurement / total,
            "expanded_depth": max(expanded_levels.values(), default=0) * scale,
            "expanded_two_qubit_ops": expanded_counts["arity_2"] * scale,
            "expanded_three_qubit_ops": expanded_counts["arity_3"] * scale,
            "expanded_rotation_ops": sum(expanded_counts[g] for g in _ROTATIONS) * scale,
            "expanded_clifford_fraction": primitive_clifford / primitive_total,
            "expanded_nonclifford_fraction": sum(expanded_counts[g] for g in ("t", "tdg", "ccx", "ccz", "cswap")) / primitive_total,
            "custom_profile_coverage": complete_calls / max(profile_calls, 1),
            "numeric_angle_fraction": angle_counts["numeric"] / max(angle_counts["total"], 1),
            "half_pi_angle_fraction": angle_counts["half_pi"] / max(angle_counts["numeric"], 1),
            "quarter_pi_angle_fraction": angle_counts["quarter_pi"] / max(angle_counts["numeric"], 1),
            "small_angle_fraction": angle_counts["small"] / max(angle_counts["numeric"], 1),
            "mean_absolute_angle": angle_counts["absolute"] / max(angle_counts["numeric"], 1),
            "reset_per_active_qubit": counts["n_reset"] * scale / max(len(loads), 1),
            "measurement_position_mean": measure_positions / max(counts["n_measure"], 1),
            "expanded_ops_per_depth": primitive_total / max(max(expanded_levels.values(), default=0), 1),
            "reset_entangling_product": counts["n_reset"] * entangling * scale**2,
            "active_qubit_log_ops": len(loads) * math.log1p(counts["n_ops"] * scale),
        })
        detailed_degrees, cut_changes = Counter(), Counter()
        for a, b in expanded_pairs:
            detailed_degrees[a] += 1
            detailed_degrees[b] += 1
            cut_changes[a] += 1
            cut_changes[b] -= 1
        current = maximum = area = previous = 0
        for point in sorted(cut_changes):
            area += current * (point - previous)
            current += cut_changes[point]
            maximum = max(maximum, current)
            previous = point
        features.update({"expanded_edge_density": 2 * len(expanded_pairs) / max(n_qubits * (n_qubits - 1), 1),
                         "expanded_max_degree": max(detailed_degrees.values(), default=0),
                         "expanded_mean_degree": sum(detailed_degrees.values()) / max(n_qubits, 1),
                         "expanded_max_cut": maximum,
                         "expanded_mean_cut": area / max(n_qubits - 1, 1)})
        for kind, values in segments.items():
            mean = sum(values) / len(values)
            features[kind + "_segment_cv"] = math.sqrt(sum((v - mean)**2 for v in values) / len(values)) / max(mean, 1)
        return {key: float(features.get(key, 0.0)) for key in FEATURE_NAMES[:-2]}

    def predict(self, features: dict, threshold: int) -> float:
        """Predict positive wall-clock seconds, including possible timeouts."""
        if self.model is None:
            raise RuntimeError("No fitted model; load a trained artifact before predicting")
        # sklearn's ordinary regression trees compare float32 input features.
        vector = array("f", _feature_vector(features, threshold, self.model["features"]))
        if "pipeline" in self.model:
            prediction = _predict_component(self.model["pipeline"], vector)
        else:
            prediction = _predict_component(self.model, vector)
        if prediction == math.log10(CAP_SECONDS):
            return float(CAP_SECONDS)
        low, high = self.model["log_bounds"]
        return float(10 ** min(high, max(low, prediction)))


def _calibration_features(prediction, log_threshold):
    return [1.0, prediction, log_threshold / 9,
            max(0.0, prediction), max(0.0, prediction - 1),
            max(0.0, prediction - 2), max(0.0, prediction - 3)]


def _predict_component(component, vector):
    """Portable inference for forests, routing, and residual corrections."""
    kind = component.get("kind", "forest")
    if kind == "routed":
        probabilities = _predict_component(component["classifier"], vector)
        if probabilities[2] >= component["timeout_confidence"]:
            return math.log10(CAP_SECONDS)
        if probabilities[0] >= component["fast_confidence"]:
            return _predict_component(component["fast"], vector)
        if probabilities[1] >= component["regular_confidence"]:
            return _predict_component(component["regular"], vector)
        return _predict_component(component["base"], vector)
    if kind == "calibrated":
        prediction = _predict_component(component["base"], vector)
        inputs = _calibration_features(prediction, vector[component["threshold_index"]])
        correction = sum(a * b for a, b in zip(inputs, component["coefficients"]))
        return prediction + max(-component["clip"], min(component["clip"], correction))
    indexes = component.get("feature_indices")
    if indexes is not None:
        vector = [vector[i] for i in indexes]
    prediction = component["bias"]
    if isinstance(prediction, list):
        prediction = prediction.copy()
    less_than = component.get("comparison") == "lt"
    for nodes in component["trees"]:
        index = 0
        while nodes[index][0] >= 0:
            feature, split, left, right = nodes[index]
            goes_left = vector[feature] < split if less_than else vector[feature] <= split
            index = left if goes_left else right
        value = nodes[index][1]
        if isinstance(prediction, list):
            for i, p in enumerate(value):
                prediction[i] += p
        else:
            prediction += value
    return prediction


def _feature_vector(features, threshold, names=FEATURE_NAMES):
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    inputs = dict(features, threshold=threshold, log2_threshold=math.log2(threshold))
    values = [float(inputs.get(key, 0.0)) for key in names]
    if not all(math.isfinite(value) and abs(value) < 1e30 for value in values):
        raise ValueError("features must be finite numbers smaller than 1e30")
    return values


def _export_model(estimator, metadata, log_bounds):
    """Export ordinary sklearn trees to portable JSON."""
    if hasattr(estimator, "learning_rate"):
        trees = estimator.estimators_.ravel()
        bias = float(estimator.init_.constant_[0, 0])
        weight = estimator.learning_rate
    else:
        trees = estimator.estimators_
        bias = 0.0
        weight = 1 / len(trees)
    forest = []
    for estimator_tree in trees:
        tree = estimator_tree.tree_
        nodes = []
        for index in range(tree.node_count):
            if tree.children_left[index] < 0:
                nodes.append([-1, float(tree.value[index, 0, 0]) * weight])
            else:
                nodes.append([int(tree.feature[index]), float(tree.threshold[index]),
                              int(tree.children_left[index]), int(tree.children_right[index])])
        forest.append(nodes)
    return {"schema_version": SCHEMA_VERSION, "features": list(FEATURE_NAMES),
            "bias": bias, "trees": forest, "log_bounds": log_bounds, "training": metadata}


def _train(args):
    """Offline training only; never invoked by the submission harness."""
    import csv
    import hashlib
    import time
    import numpy as np
    import sklearn
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from run import read_qasm

    with args.labels.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    paths = {}
    for path in sorted(args.circuits.rglob("*")):
        if path.is_file() and (path.name.endswith(".qasm") or path.name.endswith(".qasm.zst")):
            name = path.name.removesuffix(".zst")
            if name in paths:
                raise ValueError(f"Duplicate circuit basename: {name}")
            paths[name] = path
    missing = {row["filename"].strip() for row in rows} - paths.keys()
    if missing:
        raise ValueError(f"Missing labeled circuits: {sorted(missing)[:10]}")
    parser = RuntimeModel(load_model=False)
    features, groups = {}, {}
    max_parse = 0.0
    for i, name in enumerate(sorted({row["filename"].strip() for row in rows})):
        source = read_qasm(paths[name])
        start = time.perf_counter()
        features[name] = parser.featurize(source)
        max_parse = max(max_parse, time.perf_counter() - start)
        # Identical feature vectors cannot straddle validation folds, even if
        # duplicate circuits have different names or differ only in comments.
        signature = json.dumps(features[name], sort_keys=True, separators=(",", ":"))
        groups[name] = hashlib.sha256(signature.encode()).hexdigest()
        if (i + 1) % 50 == 0 or i + 1 == len(paths):
            print(f"Parsed {i + 1} circuits; slowest {max_parse:.3f}s", flush=True)
    if max_parse > 15:
        raise RuntimeError(f"Feature parsing exceeded the 15s limit: {max_parse:.3f}s")
    x, y, group_ids, timeout, keys = [], [], [], [], []
    seen = set()
    for row in rows:
        name = row["filename"].strip()
        threshold = float(row["threshold"])
        key = (name, threshold)
        if key in seen:
            raise ValueError(f"Duplicate runtime label: {key}")
        seen.add(key)
        timed_out = row.get("status", "").strip().lower() == "timeout"
        if not timed_out and not row.get("duration_s", "").strip():
            continue
        duration = CAP_SECONDS if timed_out else float(row["duration_s"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid runtime for {key}: {duration}")
        x.append(_feature_vector(features[name], threshold))
        y.append(math.log10(duration))
        group_ids.append(groups[name])
        timeout.append(timed_out)
        keys.append(key)
    x, y, timeout = np.asarray(x), np.asarray(y), np.asarray(timeout)
    candidates = {
        "boosted_absolute": lambda: GradientBoostingRegressor(
            loss="absolute_error", n_estimators=350, learning_rate=0.05,
            max_depth=3, min_samples_leaf=5, random_state=42),
        "boosted_squared": lambda: GradientBoostingRegressor(
            loss="squared_error", n_estimators=350, learning_rate=0.04,
            max_depth=3, min_samples_leaf=5, random_state=42),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=2,
            max_features=0.85, random_state=42, n_jobs=2),
    }
    splits = list(GroupKFold(n_splits=5, shuffle=True, random_state=42).split(x, y, group_ids))

    def score(pred):
        scored = np.where(timeout, np.minimum(pred, math.log10(CAP_SECONDS)), pred)
        return np.maximum(0, 1 - np.abs(scored - y) / 2)

    results, predictions = {}, {}
    for name, make_estimator in candidates.items():
        pred = np.zeros(len(y))
        fold_scores = []
        for fold, (train, valid) in enumerate(splits):
            estimator = make_estimator()
            estimator.fit(x[train], y[train])
            pred[valid] = estimator.predict(x[valid])
            fold_scores.append(float(score(pred)[valid].mean()))
            print(f"{name}: fold {fold+1}/5 score {fold_scores[-1]:.4%}", flush=True)
        results[name] = {"score": float(score(pred).mean()), "fold_scores": fold_scores,
                         "log10_mae": float(np.abs(pred-y).mean())}
        predictions[name] = pred
    winner = max(results, key=lambda name: results[name]["score"])
    estimator = candidates[winner]()
    estimator.fit(x, y)
    baseline = np.zeros(len(y))
    for train, valid in splits:
        baseline[valid] = np.median(y[train])
    metadata = {
        "model": winner, "seed": 42, "sklearn_version": sklearn.__version__,
        "circuits": len(features), "runs": len(y), "timeouts": int(timeout.sum()),
        "validation_groups": len(set(group_ids)), "validation": "5-fold grouped by circuit features",
        "candidates": results, "constant_baseline_score": float(score(baseline).mean()),
        "max_parse_seconds": max_parse,
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "top_features": sorted(zip(FEATURE_NAMES, map(float, estimator.feature_importances_)),
                               key=lambda item: item[1], reverse=True)[:12],
    }
    # score.py caps timeouts only; preserve successful durations above four hours.
    artifact = _export_model(estimator, metadata, [-9.0, max(math.log10(CAP_SECONDS), float(y.max()))])
    parser.model = artifact
    exported = np.array([math.log10(parser.predict(features[name], threshold)) for name, threshold in keys])
    expected = np.clip(estimator.predict(x), *artifact["log_bounds"])
    if not np.allclose(exported, expected, rtol=0, atol=1e-10):
        raise RuntimeError("Exported model predictions differ from sklearn")
    if args.validation_out:
        args.validation_out.parent.mkdir(parents=True, exist_ok=True)
        with args.validation_out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["team", "filename", "threshold", "pred_duration_s"])
            for (name, threshold), prediction in zip(keys, predictions[winner]):
                writer.writerow(["cross-validation", name, int(threshold), 10**prediction])
    payload = json.dumps(artifact, separators=(",", ":"), allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(f"Wrote model artifact to {args.output}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


def _main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Train and validate from labeled circuits")
    parser.add_argument("--circuits", type=Path, default=Path("training_circuits"))
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH,
                        help="Model JSON destination (default: artifacts/runtime-model.json next to this module)")
    parser.add_argument("--validation-out", type=Path, help="Write predictions made on held-out folds")
    args = parser.parse_args()
    if not args.train or args.labels is None:
        parser.error("training requires --train and --labels")
    _train(args)


if __name__ == "__main__":
    _main()
