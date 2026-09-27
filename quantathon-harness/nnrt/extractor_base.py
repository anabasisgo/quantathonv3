# Vendored from the team feature pipeline (data_first/feature_v3/base_extractor.py); local docstrings added.
"""Version 2 structural OpenQASM extractor for runtime-prediction research.

The public entry point is ``extract_features(qasm_text, budget_s=10)``. It uses
only the Python standard library. Counts are expanded through user-defined
gate bodies by memoizing a definition summary; graph edges are mapped through
formal gate operands at each call site. Large/unsupported inputs return null
for unavailable fields and explicit reliability flags instead of false zeros.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import time
from typing import Any


_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_INDEXED = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\[(\d+)\]$")
_QREG = re.compile(r"qreg\s+([A-Za-z_][A-Za-z_0-9]*)\[(\d+)\]$")
_QUBIT = re.compile(r"qubit(?:\[(\d+)\])?\s+([A-Za-z_][A-Za-z_0-9]*)$")
_COMMENT = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/")
_TOKEN = re.compile(r"(?m)^[ \t]*([A-Za-z_][A-Za-z_0-9]*)")
_CONTROL = re.compile(r"^(?:if|for|while|switch|box|def|return|break|continue|end|extern|opaque)\b", re.I)
_SKIP_DECL = re.compile(r"^(?:OPENQASM|include|creg|bit|input|output|const|let|int|float|bool)\b", re.I)

# Standard QASM 2/3 gates present in the training corpus, plus common variants.
# A custom definition with the same name takes precedence over this list.
_ONE_Q = frozenset("h x y z s sdg t tdg sx sxdg rx ry rz p u u1 u2 u3 U id phase".split())
_TWO_Q = frozenset("cx cy cz ch swap iswap ecr rxx ryy rzz crx cry crz cp cu cu1 cu2 cu3 dcx csx".split())
_THREE_Q = frozenset("ccx cswap fredkin rccx".split())
_VARIABLE_Q = frozenset("mcx mcphase mcx_gray mcx_recursive c3x c4x rc3x".split())
_BUILTIN = _ONE_Q | _TWO_Q | _THREE_Q | _VARIABLE_Q
_DETAILED_LIMIT_CHARS = 4_000_000
_DEPTH_OPERATION_LIMIT = 250_000


class _Unsupported(Exception):
    pass


class _Budget(Exception):
    pass


@dataclass(frozen=True)
class _Macro:
    formals: tuple[str, ...]
    body: tuple[str, ...]


@dataclass
class _Summary:
    counts: Counter[str]
    twoq_count: int
    multiq_count: int
    edges: Counter[tuple[int, int]]
    twoq_edges: Counter[tuple[int, int]]
    operation_count: int
    active_qubit_layers: int


def _blank(n: int | None, issues: list[str]) -> dict[str, Any]:
    """Return the standard result shape with unavailable fields marked null."""
    return {
        "n_qubits": n,
        "primitive_counts": None,
        "unitary_count": None,
        "twoq_count": None,
        "multiq_count": None,
        "measurement_count": None,
        "reset_count": None,
        "gate_depth": None,
        "twoq_depth": None,
        "critical_path_twoq_count": None,
        "active_qubit_layers": None,
        "edges": None,
        "twoq_edges": None,
        "counts_exact": False,
        "graph_exact": False,
        "depth_exact": False,
        "issues": list(dict.fromkeys(issues)),
    }


def _quick_registers(text: str) -> tuple[int | None, dict[str, tuple[int, int]]]:
    """Use C-level substring search to recover declarations even in 200 MB QASM."""
    found: list[tuple[int, str, int]] = []
    for marker in ("qreg ", "qubit[", "qubit "):
        start = 0
        while True:
            pos = text.find(marker, start)
            if pos < 0:
                break
            line_start = text.rfind("\n", 0, pos) + 1
            if not text[line_start:pos].strip():
                end = text.find(";", pos)
                if 0 <= end - pos < 256:
                    stmt = text[pos:end].strip()
                    qreg = _QREG.fullmatch(stmt)
                    qubit = _QUBIT.fullmatch(stmt)
                    if qreg:
                        found.append((pos, qreg.group(1), int(qreg.group(2))))
                    elif qubit:
                        found.append((pos, qubit.group(2), int(qubit.group(1) or 1)))
            start = pos + len(marker)
    found.sort()
    offset = 0
    registers: dict[str, tuple[int, int]] = {}
    for _, name, size in found:
        if name not in registers:
            registers[name] = (offset, size)
            offset += size
    return (offset if registers else None), registers


def _check_budget(deadline: float) -> None:
    """Stop extraction once the caller's wall-clock deadline is reached."""
    if time.perf_counter() >= deadline:
        raise _Budget


def _parse_header(header: str) -> tuple[str, tuple[str, ...]]:
    """Read a custom gate's name and formal qubit operands."""
    rest = header.strip()[4:].strip()  # remove `gate`
    match = _NAME.match(rest)
    if not match:
        raise _Unsupported("bad_gate_header")
    name = match.group(0)
    rest = rest[match.end():].strip()
    if rest.startswith("("):
        depth = 0
        close = -1
        for i, char in enumerate(rest):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close < 0:
            raise _Unsupported("bad_gate_parameters")
        rest = rest[close + 1:].strip()
    formals = tuple(part.strip() for part in rest.split(",") if part.strip())
    if not formals or any(not _NAME.fullmatch(item) for item in formals):
        raise _Unsupported("bad_gate_formals")
    if len(set(formals)) != len(formals):
        raise _Unsupported("duplicate_gate_formal")
    return name, formals


def _split_structure(text: str, deadline: float) -> tuple[dict[str, _Macro], list[str], list[str]]:
    """Split semicolon statements and gate blocks, supporting single-line blocks."""
    clean = _COMMENT.sub("", text)
    parts = re.split(r"([{};])", clean)
    macros: dict[str, _Macro] = {}
    top: list[str] = []
    issues: list[str] = []
    body: list[str] | None = None
    macro_name = ""
    formals: tuple[str, ...] = ()
    pending = ""
    for i, part in enumerate(parts):
        if i % 4096 == 0:
            _check_budget(deadline)
        if part == ";":
            statement = pending.strip()
            pending = ""
            if statement:
                (body if body is not None else top).append(statement)
        elif part == "{":
            header = pending.strip()
            pending = ""
            if body is not None:
                issues.append("nested_block")
            elif header.startswith("gate "):
                try:
                    macro_name, formals = _parse_header(header)
                    body = []
                except _Unsupported as exc:
                    issues.append(str(exc))
            else:
                issues.append("unsupported_block")
        elif part == "}":
            if pending.strip():
                (body if body is not None else top).append(pending.strip())
            pending = ""
            if body is not None:
                if macro_name in macros:
                    issues.append("duplicate_gate_definition")
                else:
                    macros[macro_name] = _Macro(formals, tuple(body))
                body = None
                macro_name = ""
                formals = ()
            else:
                issues.append("unmatched_closing_brace")
        else:
            pending += part
    if pending.strip():
        top.append(pending.strip())
    if body is not None:
        issues.append("unclosed_gate_definition")
    return macros, top, issues


def _parse_call(statement: str) -> tuple[str, tuple[str, ...]]:
    """Read a gate call and its operands, skipping angle parameters."""
    statement = statement.strip()
    match = _NAME.match(statement)
    if not match:
        raise _Unsupported("unknown_statement")
    name = match.group(0)
    rest = statement[match.end():].strip()
    if rest.startswith("("):
        depth = 0
        close = -1
        for i, char in enumerate(rest):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close < 0:
            raise _Unsupported("bad_gate_parameters")
        rest = rest[close + 1:].strip()
    if not rest:
        return name, ()
    return name, tuple(part.strip() for part in rest.split(","))


def _arity_ok(name: str, arity: int) -> bool:
    """Check a built-in gate's qubit count against its supported arity."""
    if name in _ONE_Q:
        return arity == 1
    if name in _TWO_Q:
        return arity == 2
    if name in _THREE_Q:
        return arity == 3
    if name in _VARIABLE_Q:
        return arity >= 2
    return False


def _pairs(values: tuple[int, ...] | list[int]):
    """Yield sorted distinct qubit pairs induced by a multi-qubit call."""
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            if a != b:
                yield (a, b) if a < b else (b, a)


def _summary_for(
    name: str,
    macros: dict[str, _Macro],
    cache: dict[str, _Summary],
    active: set[str],
    deadline: float,
) -> _Summary:
    """Memoize a custom gate's expanded counts and formal-qubit graph edges."""
    if name in cache:
        return cache[name]
    if name in active:
        raise _Unsupported("macro_cycle")
    macro = macros[name]
    active.add(name)
    formal_indices = {formal: i for i, formal in enumerate(macro.formals)}
    counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[int, int]] = Counter()
    twoq_edges: Counter[tuple[int, int]] = Counter()
    twoq = multiq = operation_count = active_qubit_layers = 0
    for i, statement in enumerate(macro.body):
        if i % 1024 == 0:
            _check_budget(deadline)
        gate, args = _parse_call(statement)
        if _CONTROL.match(statement) or gate in ("measure", "reset", "barrier"):
            raise _Unsupported("nonunitary_or_control_in_gate_definition")
        if any(arg not in formal_indices for arg in args):
            raise _Unsupported("unresolved_gate_formal")
        mapped = tuple(formal_indices[arg] for arg in args)
        if len(set(mapped)) != len(mapped):
            raise _Unsupported("duplicate_gate_operand")
        if gate in macros:
            child = _summary_for(gate, macros, cache, active, deadline)
            if len(mapped) != len(macros[gate].formals):
                raise _Unsupported("macro_arity_mismatch")
            counts.update(child.counts)
            twoq += child.twoq_count
            multiq += child.multiq_count
            operation_count += child.operation_count
            active_qubit_layers += child.active_qubit_layers
            for (a, b), weight in child.edges.items():
                pair = tuple(sorted((mapped[a], mapped[b])))
                edge_counts[pair] += weight
            for (a, b), weight in child.twoq_edges.items():
                pair = tuple(sorted((mapped[a], mapped[b])))
                twoq_edges[pair] += weight
        elif gate in _BUILTIN and _arity_ok(gate, len(mapped)):
            counts[gate] += 1
            operation_count += 1
            active_qubit_layers += len(mapped)
            if len(mapped) == 2:
                twoq += 1
                twoq_edges[next(_pairs(mapped))] += 1
            elif len(mapped) >= 3:
                multiq += 1
            edge_counts.update(_pairs(mapped))
        else:
            raise _Unsupported("unknown_gate_in_definition:" + gate)
    active.remove(name)
    result = _Summary(counts, twoq, multiq, edge_counts, twoq_edges, operation_count, active_qubit_layers)
    cache[name] = result
    return result


def _resolve_operand(token: str, registers: dict[str, tuple[int, int]], env: dict[str, int] | None = None) -> list[int]:
    """Resolve an indexed qubit, whole register, or gate formal to indices."""
    token = token.strip()
    if env is not None and token in env:
        return [env[token]]
    indexed = _INDEXED.fullmatch(token)
    if indexed:
        name, index = indexed.group(1), int(indexed.group(2))
        if name not in registers:
            raise _Unsupported("unknown_quantum_register:" + name)
        offset, size = registers[name]
        if index >= size:
            raise _Unsupported("qubit_index_out_of_range")
        return [offset + index]
    if token in registers:
        offset, size = registers[token]
        return list(range(offset, offset + size))
    raise _Unsupported("unknown_quantum_operand:" + token[:40])


def _broadcast(args: tuple[str, ...], registers: dict[str, tuple[int, int]], env: dict[str, int] | None = None) -> list[tuple[int, ...]]:
    """Expand register operands into individual gate invocations."""
    groups = [_resolve_operand(arg, registers, env) for arg in args]
    if not groups:
        raise _Unsupported("gate_without_qubits")
    width = max(map(len, groups))
    if any(len(group) not in (1, width) for group in groups):
        raise _Unsupported("incompatible_register_broadcast")
    invocations = []
    for i in range(width):
        mapped = tuple(group[i] if len(group) == width else group[0] for group in groups)
        if len(set(mapped)) != len(mapped):
            raise _Unsupported("duplicate_gate_operand")
        invocations.append(mapped)
    return invocations


def _measurement_operand(statement: str) -> str:
    """Extract the quantum operand from QASM 2 or QASM 3 measurement syntax."""
    if statement.startswith("measure "):
        return statement[8:].split("->", 1)[0].strip()
    match = re.search(r"=\s*measure\s+(.+)$", statement)
    if match:
        return match.group(1).strip()
    raise _Unsupported("bad_measurement")


def _edges_list(edges: Counter[tuple[int, int]]) -> list[list[int]]:
    """Serialize nonzero weighted edges in stable qubit order."""
    return [[a, b, count] for (a, b), count in sorted(edges.items()) if count]


def _depth_for(
    top: list[str], macros: dict[str, _Macro], registers: dict[str, tuple[int, int]],
    n: int, expanded_operations: int, deadline: float,
) -> tuple[int | None, int | None, int | None, list[str]]:
    """Compute ordered gate depth when the expanded operation count is bounded."""
    if expanded_operations > _DEPTH_OPERATION_LIMIT:
        return None, None, None, ["depth_operation_limit"]
    depth = [0] * n
    twoq_depth = [0] * n
    critical_twoq = [0] * n
    seen_nonunitary = False
    issues: list[str] = []
    processed = 0

    def apply(gate: str, mapped: tuple[int, ...], active: tuple[str, ...] = ()) -> None:
        nonlocal processed
        processed += 1
        if processed % 1024 == 0:
            _check_budget(deadline)
        if gate in macros:
            if gate in active:
                raise _Unsupported("macro_cycle")
            macro = macros[gate]
            env = dict(zip(macro.formals, mapped))
            for statement in macro.body:
                child, args = _parse_call(statement)
                if child in macros or child in _BUILTIN:
                    for qubits in _broadcast(args, registers, env):
                        apply(child, qubits, active + (gate,))
                else:
                    raise _Unsupported("unknown_gate_in_definition:" + child)
        else:
            # Dynamic programming on the full unitary dependency DAG.
            # Equal-depth predecessor paths choose the one with more 2q gates.
            predecessor = max(mapped, key=lambda q: (depth[q], critical_twoq[q]))
            layer = depth[predecessor] + 1
            critical_count = critical_twoq[predecessor] + int(len(mapped) == 2)
            for q in mapped:
                depth[q] = layer
                critical_twoq[q] = critical_count
            if len(mapped) == 2:
                twoq_layer = max(twoq_depth[q] for q in mapped) + 1
                for q in mapped:
                    twoq_depth[q] = twoq_layer

    try:
        for i, statement in enumerate(top):
            if i % 1024 == 0:
                _check_budget(deadline)
            if _QREG.fullmatch(statement) or _QUBIT.fullmatch(statement) or _SKIP_DECL.match(statement):
                continue
            if statement.startswith("measure ") or re.search(r"=\s*measure\b", statement):
                seen_nonunitary = True
                continue
            if statement.startswith("reset "):
                seen_nonunitary = True
                continue
            if statement.startswith("barrier"):
                args = statement[7:].strip()
                qubits = sorted({q for part in args.split(",") if part.strip() for q in _resolve_operand(part, registers)}) if args else list(range(n))
                if qubits:
                    predecessor = max(qubits, key=lambda q: (depth[q], critical_twoq[q]))
                    layer = depth[predecessor]
                    critical_count = critical_twoq[predecessor]
                    twoq_layer = max(twoq_depth[q] for q in qubits)
                    for q in qubits:
                        depth[q] = layer
                        critical_twoq[q] = critical_count
                        twoq_depth[q] = twoq_layer
                continue
            gate, args = _parse_call(statement)
            if gate not in macros and gate not in _BUILTIN:
                raise _Unsupported("unknown_statement:" + gate)
            for mapped in _broadcast(args, registers):
                if seen_nonunitary:
                    issues.append("gate_after_measure_or_reset")
                apply(gate, mapped)
    except _Budget:
        return None, None, None, ["depth_budget"]
    except _Unsupported as exc:
        return None, None, None, [str(exc)]
    if issues:
        return None, None, None, list(dict.fromkeys(issues))
    if n:
        endpoint = max(range(n), key=lambda q: (depth[q], critical_twoq[q]))
        return depth[endpoint], max(twoq_depth, default=0), critical_twoq[endpoint], []
    return 0, 0, 0, []


def extract_features(qasm_text: str, budget_s: float = 10) -> dict[str, Any]:
    """Return expanded structural features with per-field reliability flags.

    `edges` adds a clique for each k>=2 gate; `twoq_edges` includes only gates
    with exactly two operands. Gate-only depth uses zero-layer barrier sync.
    """
    start = time.perf_counter()
    deadline = start + max(float(budget_s), 0.01)
    n, registers = _quick_registers(qasm_text)
    if n is None:
        return _blank(None, ["missing_qubit_declaration"])
    if len(qasm_text) > _DETAILED_LIMIT_CHARS:
        return _blank(n, ["bounded_large_file", "counts_unavailable", "graph_unavailable", "depth_unavailable"])
    try:
        macros, top, parse_issues = _split_structure(qasm_text, deadline)
        if parse_issues:
            return _blank(n, parse_issues)
        if len(macros) > 1000:
            return _blank(n, ["too_many_gate_definitions"])
        summaries: dict[str, _Summary] = {}
        counts: Counter[str] = Counter()
        graph: Counter[tuple[int, int]] = Counter()
        twoq_graph: Counter[tuple[int, int]] = Counter()
        twoq = multiq = measured = reset = expanded_operations = active_qubit_layers = 0
        has_unsupported = False
        issues: list[str] = []
        for i, statement in enumerate(top):
            if i % 512 == 0:
                _check_budget(deadline)
            qreg = _QREG.fullmatch(statement)
            qubit = _QUBIT.fullmatch(statement)
            if qreg or qubit or _SKIP_DECL.match(statement):
                continue
            if _CONTROL.match(statement):
                issues.append("unsupported_control_flow")
                has_unsupported = True
                break
            if statement.startswith("measure ") or re.search(r"=\s*measure\b", statement):
                measured += len(_resolve_operand(_measurement_operand(statement), registers))
                continue
            if statement.startswith("reset "):
                reset += len(_resolve_operand(statement[6:].strip(), registers))
                continue
            if statement.startswith("barrier"):
                # A barrier is not a unitary or measurement; depth handles it.
                continue
            gate, args = _parse_call(statement)
            if gate not in macros and gate not in _BUILTIN:
                issues.append("unknown_statement:" + gate)
                has_unsupported = True
                break
            for mapped in _broadcast(args, registers):
                if gate in macros:
                    summary = _summary_for(gate, macros, summaries, set(), deadline)
                    if len(mapped) != len(macros[gate].formals):
                        raise _Unsupported("macro_arity_mismatch")
                    counts.update(summary.counts)
                    twoq += summary.twoq_count
                    multiq += summary.multiq_count
                    expanded_operations += summary.operation_count
                    active_qubit_layers += summary.active_qubit_layers
                    for (a, b), weight in summary.edges.items():
                        graph[tuple(sorted((mapped[a], mapped[b])))] += weight
                    for (a, b), weight in summary.twoq_edges.items():
                        twoq_graph[tuple(sorted((mapped[a], mapped[b])))] += weight
                else:
                    if not _arity_ok(gate, len(mapped)):
                        raise _Unsupported("gate_arity_mismatch:" + gate)
                    counts[gate] += 1
                    expanded_operations += 1
                    active_qubit_layers += len(mapped)
                    if len(mapped) == 2:
                        twoq += 1
                        twoq_graph[next(_pairs(mapped))] += 1
                    elif len(mapped) >= 3:
                        multiq += 1
                    graph.update(_pairs(mapped))
        if has_unsupported:
            return _blank(n, issues)
        result = {
            "n_qubits": n,
            "primitive_counts": dict(sorted(counts.items())),
            "unitary_count": sum(counts.values()),
            "twoq_count": twoq,
            "multiq_count": multiq,
            "measurement_count": measured,
            "reset_count": reset,
            "gate_depth": None,
            "twoq_depth": None,
            "critical_path_twoq_count": None,
            "active_qubit_layers": active_qubit_layers,
            "edges": _edges_list(graph),
            "twoq_edges": _edges_list(twoq_graph),
            "counts_exact": True,
            "graph_exact": True,
            "depth_exact": False,
            "issues": [],
        }
        depth, twoq_depth, critical_path_twoq, depth_issues = _depth_for(
            top, macros, registers, n, expanded_operations, deadline
        )
        result["gate_depth"] = depth
        result["twoq_depth"] = twoq_depth
        result["critical_path_twoq_count"] = critical_path_twoq
        result["depth_exact"] = depth is not None and not depth_issues
        result["issues"] = depth_issues
        return result
    except _Budget:
        return _blank(n, ["budget_exceeded"])
    except _Unsupported as exc:
        return _blank(n, [str(exc)])
    except Exception as exc:
        return _blank(n, ["parser_error:" + type(exc).__name__])
