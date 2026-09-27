# Vendored from the team feature pipeline (data_first/feature_v3/extractor.py); only the import of the base module changed.
"""Bounded v3 extension: stream large QASM into unordered structural summaries.

The small-input path is the frozen v2 implementation. The streaming path does
not preserve execution order and therefore never claims exact circuit depth.
"""
from collections import Counter
from functools import lru_cache
import re
import time
from . import extractor_base as base

_DELIMITERS = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|[{};]')
# Fast common case; expressions with nested parentheses use the base parser.
_CALL = re.compile(r'^([A-Za-z_][A-Za-z_0-9]*)(?:\s*\([^()]*\))?\s+(.+)$', re.S)
_MAX_DEFINITIONS = 100_000
_MAX_BODY_STATEMENTS = 1_500_000
_MAX_UNIQUE_CALLS = 200_000


def _call_key(statement):
    match = _CALL.fullmatch(statement)
    if match:
        return match.group(1), match.group(2).strip()
    name, args = base._parse_call(statement)
    return name, ','.join(args)


def _stream_structure(text, deadline):
    """No re.split of the full source and no list of all top-level operations."""
    macros = {}
    top = Counter()
    start = 0
    comment_parts = []
    body = None
    name = None
    formals = ()
    body_statements = 0
    # Small bounded cache handles repeated textual operations cheaply. Parameter
    # values are irrelevant to these structural features, not to equivalence.
    call_key = lru_cache(maxsize=8192)(_call_key)
    for index, token in enumerate(_DELIMITERS.finditer(text)):
        if index % 2048 == 0:
            base._check_budget(deadline)
        delimiter = token.group(0)
        if delimiter.startswith('/'):
            comment_parts.append(text[start:token.start()])
            comment_parts.append(' ')  # comments separate lexical tokens
            start = token.end()
            continue
        if comment_parts:
            comment_parts.append(text[start:token.start()])
            statement = ''.join(comment_parts).strip()
            comment_parts = []
        else:
            statement = text[start:token.start()].strip()
        start = token.end()
        if delimiter == '{':
            if body is not None or not statement.startswith('gate '):
                raise base._Unsupported('unsupported_block')
            name, formals = base._parse_header(statement)
            if name in macros:
                raise base._Unsupported('duplicate_gate_definition')
            if len(macros) >= _MAX_DEFINITIONS:
                raise base._Unsupported('stream_definition_limit')
            body = []
        elif delimiter == '}':
            if body is None:
                raise base._Unsupported('unmatched_closing_brace')
            if statement:
                raise base._Unsupported('missing_semicolon_in_gate')
            macros[name] = base._Macro(formals, tuple(body))
            body = None
        elif statement:
            if body is not None:
                body.append(statement)
                body_statements += 1
                if body_statements > _MAX_BODY_STATEMENTS:
                    raise base._Unsupported('stream_body_statement_limit')
            else:
                # Almost every statement in a flat large file is a gate. Avoid
                # running all declaration/control regexes for those operations.
                key = call_key(statement)
                if key[0] in base._BUILTIN or key[0] in macros:
                    top[key] += 1
                    if len(top) > _MAX_UNIQUE_CALLS:
                        raise base._Unsupported('stream_unique_call_limit')
                    continue
                if base._CONTROL.match(statement):
                    raise base._Unsupported('unsupported_control_flow')
                if base._QREG.fullmatch(statement) or base._QUBIT.fullmatch(statement) or base._SKIP_DECL.match(statement):
                    continue
                if statement.startswith('measure ') or re.search(r'=\s*measure\b', statement):
                    top[('__measure__', base._measurement_operand(statement))] += 1
                elif statement.startswith('reset '):
                    top[('__reset__', statement[6:].strip())] += 1
                elif statement.startswith('barrier'):
                    # Validate operands later; barriers add no unitary count.
                    top[('__barrier__', statement[7:].strip())] += 1
                else:
                    top[key] += 1
                if len(top) > _MAX_UNIQUE_CALLS:
                    raise base._Unsupported('stream_unique_call_limit')
    tail = ''.join(comment_parts) + text[start:]
    if tail.strip():
        raise base._Unsupported('missing_final_semicolon')
    if body is not None:
        raise base._Unsupported('unclosed_gate_definition')
    return macros, top


def _stream_features(text, n, registers, deadline):
    macros, top = _stream_structure(text, deadline)
    counts, graph, twoq_graph = Counter(), Counter(), Counter()
    summaries = {}
    twoq = multiq = measured = reset = active = 0
    for index, ((gate, operands), frequency) in enumerate(top.items()):
        if index % 256 == 0:
            base._check_budget(deadline)
        if gate in ('__measure__', '__reset__'):
            count = len(base._resolve_operand(operands, registers)) * frequency
            if gate == '__measure__':
                measured += count
            else:
                reset += count
            continue
        if gate == '__barrier__':
            for part in operands.split(','):
                if part.strip():
                    base._resolve_operand(part, registers)
            continue
        if gate not in macros and gate not in base._BUILTIN:
            raise base._Unsupported('unknown_statement:' + gate)
        args = tuple(p.strip() for p in operands.split(','))
        for mapped in base._broadcast(args, registers):
            if gate in macros:
                summary = base._summary_for(gate, macros, summaries, set(), deadline)
                if len(mapped) != len(macros[gate].formals):
                    raise base._Unsupported('macro_arity_mismatch')
                for op, count in summary.counts.items():
                    counts[op] += count * frequency
                twoq += summary.twoq_count * frequency
                multiq += summary.multiq_count * frequency
                active += summary.active_qubit_layers * frequency
                for (a, b), count in summary.edges.items():
                    graph[tuple(sorted((mapped[a], mapped[b])))] += count * frequency
                for (a, b), count in summary.twoq_edges.items():
                    twoq_graph[tuple(sorted((mapped[a], mapped[b])))] += count * frequency
            else:
                if not base._arity_ok(gate, len(mapped)):
                    raise base._Unsupported('gate_arity_mismatch:' + gate)
                counts[gate] += frequency
                active += len(mapped) * frequency
                if len(mapped) == 2:
                    twoq += frequency
                    twoq_graph[next(base._pairs(mapped))] += frequency
                elif len(mapped) >= 3:
                    multiq += frequency
                for pair in base._pairs(mapped):
                    graph[pair] += frequency
    result = base._blank(n, ['stream_order_not_retained_depth_unavailable'])
    result.update(primitive_counts=dict(sorted(counts.items())), unitary_count=sum(counts.values()),
                  twoq_count=twoq, multiq_count=multiq, measurement_count=measured,
                  reset_count=reset, active_qubit_layers=active,
                  edges=base._edges_list(graph), twoq_edges=base._edges_list(twoq_graph),
                  counts_exact=True, graph_exact=True, stream_unique_calls=len(top),
                  stream_macro_definitions=len(macros))
    return result


def extract_features(qasm_text, budget_s=10):
    """Return v2-compatible features, bounded by elapsed-time and structure caps."""
    start = time.perf_counter()
    deadline = start + max(float(budget_s), 0.01)
    if len(qasm_text) <= base._DETAILED_LIMIT_CHARS:
        result = base.extract_features(qasm_text, budget_s=budget_s)
        if result['issues'] != ['too_many_gate_definitions']:
            result['extraction_method'] = 'v2_compatible'
            return result
    n, registers = base._quick_registers(qasm_text)
    if n is None:
        result = base._blank(None, ['missing_qubit_declaration'])
    else:
        try:
            base._check_budget(deadline)
            result = _stream_features(qasm_text, n, registers, deadline)
        except base._Budget:
            result = base._blank(n, ['stream_budget_exceeded'])
        except base._Unsupported as exc:
            result = base._blank(n, [str(exc)])
        except Exception as exc:
            result = base._blank(n, ['stream_parser_error:' + type(exc).__name__])
    result['extraction_method'] = 'stream_aggregated'
    return result
