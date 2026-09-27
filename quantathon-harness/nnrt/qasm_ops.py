"""QASM 2/3 -> flat list of 1- and 2-qubit operations for the probe.

Deterministic limits (a character prefix and an operation cap) keep the cost bounded and make the output
independent of machine speed. Gate definitions are expanded; ccx/cswap/rccx are decomposed; other many-qubit
gates are approximated by a controlled-phase chain; measurement, reset and barriers are ignored; the bodies of
classically conditioned blocks are applied. Angle expressions go through a restricted arithmetic evaluator.
"""
import ast
from functools import lru_cache
import math
import re

from .gates import ONEQ, TWOQ, decompose

_STMT = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|[{};]')
_CALL = re.compile(r'^([A-Za-z_]\w*)\s*(?:\((.*)\))?\s*(.*)$', re.S)
_IDX = re.compile(r'([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$')
_QREG = re.compile(r'qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$')
_QUBITS = re.compile(r'qubit\s*\[\s*(\d+)\s*\]\s*([A-Za-z_]\w*)$')
_QUBIT = re.compile(r'qubit\s+([A-Za-z_]\w*)$')
_GATE = re.compile(r'gate\s+([A-Za-z_]\w*)\s*(?:\((.*?)\))?\s*(.*)$', re.S)
_SKIP = ('OPENQASM', 'include', 'creg', 'bit', 'barrier', 'measure', 'reset')
UNRESOLVED_ANGLE = 0.3   # generic angle used when an expression cannot be evaluated

_CONST = {'pi': math.pi, 'π': math.pi, 'tau': math.tau, 'e': math.e}
_FUNC = {'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'exp': math.exp, 'ln': math.log, 'log': math.log,
         'sqrt': math.sqrt, 'asin': math.asin, 'acos': math.acos, 'atan': math.atan}
_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}


def evaluate(expr: str, env: dict) -> float:
    """Restricted arithmetic with cached constant expressions."""
    if not env:
        return _evaluate_constant(expr)
    return _evaluate(expr, env)


@lru_cache(maxsize=8192)
def _evaluate_constant(expr: str) -> float:
    return _evaluate(expr, {})


def _evaluate(expr: str, env: dict) -> float:
    """Numbers, constants, bound parameters, basic arithmetic, and selected math functions."""
    def walk(node, depth=0):
        if depth > 40:
            raise ValueError('too deep')
        if isinstance(node, ast.Expression):
            return walk(node.body, depth + 1)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                return float(env[node.id])
            return _CONST[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = walk(node.operand, depth + 1)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            a, b = walk(node.left, depth + 1), walk(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and (abs(b) > 64 or abs(a) > 1e6):
                raise ValueError('power too large')
            return _BIN[type(node.op)](a, b)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNC and len(node.args) == 1:
            return _FUNC[node.func.id](walk(node.args[0], depth + 1))
        raise ValueError('unsupported expression')
    try:
        if len(expr) > 256:
            raise ValueError('too long')
        v = walk(ast.parse(expr.strip().replace('^', '**'), mode='eval'))
        return v if math.isfinite(v) else UNRESOLVED_ANGLE
    except Exception:
        return UNRESOLVED_ANGLE


def _split(s):
    """Split comma-separated operands while respecting nested angle expressions."""
    if '(' not in s and ')' not in s:
        parts = [part.strip() for part in s.split(',')]
        if parts and not parts[-1]:
            parts.pop()
        return parts
    out, depth, cur = [], 0, []
    for ch in s:
        depth += (ch == '(') - (ch == ')')
        if ch == ',' and depth == 0:
            out.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if ''.join(cur).strip():
        out.append(''.join(cur).strip())
    return out


class Circuit:
    """Bounded probe parse: qubit count, flattened operations, and coverage flags."""

    def __init__(self):
        self.n = 0
        self.ops = []            # (name, qubit tuple, params tuple), 1q/2q only
        self.statements = 0      # statements read
        self.total_statements = 0
        self.truncated = False   # hit the character prefix or operation cap

    @property
    def coverage(self):
        return min(1.0, self.statements / max(self.total_statements, 1))


class _Full(Exception):
    pass


def parse(text: str, max_chars: int, max_ops: int) -> Circuit:
    """Expand QASM into ordered one- and two-qubit probe operations.

    Read at most ``max_chars`` input characters and emit at most ``max_ops``
    operations. ``total_statements`` still counts the full input for coverage.
    """
    C = Circuit()
    C.total_statements = text.count(';')
    if len(text) > max_chars:
        text, C.truncated = text[:max_chars], True
    regs, defs = {}, {}
    pending, start, body, head, in_block = [], 0, None, None, 0

    # Many operations reuse the same indexed qubits; declarations clear this cache below.
    @lru_cache(maxsize=8192)
    def operand(tok):
        m = _IDX.match(tok.strip())
        if m:
            off, _ = regs[m.group(1)]
            return [off + int(m.group(2))]
        off, size = regs[tok.strip()]
        return list(range(off, off + size))

    def emit(name, qs, params, depth=0):
        if len(C.ops) >= max_ops:
            C.truncated = True
            raise _Full
        if name in defs and depth < 20:
            fparams, fqubits, fbody = defs[name]
            env, qmap = dict(zip(fparams, params)), dict(zip(fqubits, qs))
            for gname, gparams, gargs in fbody:
                emit(gname, tuple(qmap[a] for a in gargs), tuple(evaluate(p, env) for p in gparams), depth + 1)
            return
        if (len(qs) == 1 and name in ONEQ) or (len(qs) == 2 and name in TWOQ):
            C.ops.append((name, qs, params))
            return
        parts = decompose(name, qs) if len(qs) == 3 else None
        if parts is not None:
            for g in parts:
                emit(*g, depth=depth + 1)
        elif len(qs) >= 2 and name not in ('barrier', 'measure', 'reset', 'delay'):
            for a, b in zip(qs[:-1], qs[1:]):     # unknown many-qubit gate: entangling chain approximation
                C.ops.append(('cp', (a, b), (math.pi / 2,)))

    try:
        for tok in _STMT.finditer(text):
            d = tok.group()
            fragment = text[start:tok.start()]
            stmt = (''.join(pending) + fragment).strip() if pending else fragment.strip()
            pending, start = [], tok.end()
            if d.startswith('/'):           # comment: keep the partial statement
                pending.append(stmt + ' ')
                continue
            C.statements += 1
            if d == '{':
                if stmt.startswith('gate '):
                    m = _GATE.match(stmt)
                    head = (m.group(1), [x.strip() for x in (m.group(2) or '').split(',') if x.strip()],
                            [x.strip() for x in m.group(3).split(',') if x.strip()])
                    body = []
                else:
                    in_block += 1           # if (...) { ... }: apply the body
                continue
            if d == '}':
                if body is not None and in_block == 0:
                    defs[head[0]] = (head[1], head[2], body)
                    body = head = None
                elif in_block:
                    in_block -= 1
            if not stmt:
                continue
            if body is not None and in_block == 0:
                m = _CALL.match(stmt)
                if m:
                    body.append((m.group(1), _split(m.group(2) or ''), [a.strip() for a in m.group(3).split(',') if a.strip()]))
                continue
            word = stmt.split(None, 1)[0]
            if word in _SKIP or word.startswith('bit['):
                continue
            if word == 'qreg':
                m = _QREG.match(stmt)
                if m:
                    regs[m.group(1)] = (C.n, int(m.group(2)))
                    C.n += int(m.group(2))
                    operand.cache_clear()
                    continue
            if word == 'qubit' or word.startswith('qubit['):
                m = _QUBITS.match(stmt)
                if m:
                    regs[m.group(2)] = (C.n, int(m.group(1)))
                    C.n += int(m.group(1))
                    operand.cache_clear()
                    continue
                m = _QUBIT.match(stmt)
                if m:
                    regs[m.group(1)] = (C.n, 1)
                    C.n += 1
                    operand.cache_clear()
                    continue
            if '=' in stmt and 'measure' in stmt:
                continue
            if stmt.startswith('if'):
                stmt = re.sub(r'^if\s*\(.*?\)\s*', '', stmt)
                if not stmt:
                    continue
            m = _CALL.match(stmt)
            if not m:
                continue
            params = tuple(evaluate(p, {}) for p in _split(m.group(2) or ''))
            try:
                groups = [operand(a) for a in _split(m.group(3))]
            except Exception:
                continue
            if not groups:
                continue
            if len(groups) == 1 and len(groups[0]) == 1:
                emit(m.group(1), (groups[0][0],), params)
                continue
            if len(groups) == 2 and len(groups[0]) == len(groups[1]) == 1:
                a, b = groups[0][0], groups[1][0]
                if a != b:
                    emit(m.group(1), (a, b), params)
                continue
            width = max(len(g) for g in groups)
            for k in range(width):
                qs = tuple(g[k] if len(g) == width else g[0] for g in groups)
                if len(set(qs)) == len(qs):
                    emit(m.group(1), qs, params)
    except _Full:
        pass
    return C
