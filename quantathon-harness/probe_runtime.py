"""Bounded truncated-MPS probe adapted from origin/berni (970d51a).

Prototype. Parses a practical QASM 2/3 subset into a gate stream, then
 (a) tracks a structural upper bound on each bond's log2 bond dimension, and
 (b) runs a numpy MPS with a small bond cap (chi_probe) and records how fast
     and how widely entanglement saturates the cap.
Everything is time-boxed; partial prefixes are reported with coverage fractions.
"""
from __future__ import annotations
import ast, math, re, time
import numpy as np

# ---------------------------------------------------------------- gate matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1, -1]).astype(complex); H = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)
S = np.diag([1, 1j]); SDG = S.conj(); T = np.diag([1, np.exp(1j*math.pi/4)]); TDG = T.conj()
SX = 0.5*np.array([[1+1j, 1-1j], [1-1j, 1+1j]]); SXDG = SX.conj().T

def rx(t): return np.array([[math.cos(t/2), -1j*math.sin(t/2)], [-1j*math.sin(t/2), math.cos(t/2)]])
def ry(t): return np.array([[math.cos(t/2), -math.sin(t/2)], [math.sin(t/2), math.cos(t/2)]], complex)
def rz(t): return np.diag([np.exp(-1j*t/2), np.exp(1j*t/2)])
def ph(t): return np.diag([1, np.exp(1j*t)])
def u3(t, p, l):
    return np.array([[math.cos(t/2), -np.exp(1j*l)*math.sin(t/2)],
                     [np.exp(1j*p)*math.sin(t/2), np.exp(1j*(p+l))*math.cos(t/2)]])
def r_gate(t, p): return u3(t, p - math.pi/2, -p + math.pi/2)

ONEQ = {
    'h': lambda: H, 'x': lambda: X, 'y': lambda: Y, 'z': lambda: Z, 's': lambda: S, 'sdg': lambda: SDG,
    't': lambda: T, 'tdg': lambda: TDG, 'sx': lambda: SX, 'sxdg': lambda: SXDG, 'id': lambda: I2,
    'rx': rx, 'ry': ry, 'rz': rz, 'p': ph, 'phase': ph, 'u1': ph,
    'u2': lambda p, l: u3(math.pi/2, p, l), 'u3': u3, 'u': u3, 'U': u3, 'r': r_gate,
}

def ctrl(U):
    M = np.eye(4, dtype=complex); M[2:, 2:] = U; return M
def expm_pauli(P, t):  # exp(-i t/2 P) for P with P^2 = I
    return math.cos(t/2)*np.eye(4) - 1j*math.sin(t/2)*P
XX, YY, ZZ = np.kron(X, X), np.kron(Y, Y), np.kron(Z, Z)
ZX = np.kron(Z, X)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex)
ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], complex)
def xx_pm_yy(t, b, sign):
    # generic 2-qubit rotation in a 2-d subspace; exact form not critical for the probe
    M = np.eye(4, dtype=complex)
    c, s = math.cos(t/2), math.sin(t/2)
    if sign < 0:  # xx_minus_yy acts on |00>,|11>
        M[0, 0] = M[3, 3] = c; M[0, 3] = -1j*s*np.exp(-1j*b); M[3, 0] = -1j*s*np.exp(1j*b)
    else:         # xx_plus_yy acts on |01>,|10>
        M[1, 1] = M[2, 2] = c; M[1, 2] = -1j*s*np.exp(-1j*b); M[2, 1] = -1j*s*np.exp(1j*b)
    return M

TWOQ = {
    'cx': lambda: ctrl(X), 'cy': lambda: ctrl(Y), 'cz': lambda: ctrl(Z), 'ch': lambda: ctrl(H),
    'csx': lambda: ctrl(SX), 'cs': lambda: ctrl(S), 'csdg': lambda: ctrl(SDG),
    'swap': lambda: SWAP, 'iswap': lambda: ISWAP,
    'dcx': lambda: ctrl(X) @ SWAP @ ctrl(X) @ SWAP,  # cx(a,b) cx(b,a)
    'ecr': lambda: np.array([[0,1,0,1j],[1,0,-1j,0],[0,1j,0,1],[-1j,0,1,0]],complex)/math.sqrt(2),
    'rxx': lambda t: expm_pauli(XX, t), 'ryy': lambda t: expm_pauli(YY, t),
    'rzz': lambda t: expm_pauli(ZZ, t), 'rzx': lambda t: expm_pauli(ZX, t),
    'crx': lambda t: ctrl(rx(t)), 'cry': lambda t: ctrl(ry(t)), 'crz': lambda t: ctrl(rz(t)),
    'cp': lambda t: ctrl(ph(t)), 'cu1': lambda t: ctrl(ph(t)), 'cphase': lambda t: ctrl(ph(t)),
    'cu3': lambda t, p, l: ctrl(u3(t, p, l)), 'cu': lambda t, p, l, g=0.0: ctrl(np.exp(1j*g)*u3(t, p, l)),
    'xx_minus_yy': lambda t, b=0.0: xx_pm_yy(t, b, -1), 'xx_plus_yy': lambda t, b=0.0: xx_pm_yy(t, b, +1),
}
# operator-Schmidt log2-rank of each 2q gate type (1 = controlled/rank-2, 2 = generic/swap-like)
RANK2 = {'swap': 2, 'iswap': 2, 'dcx': 2, 'xx_minus_yy': 1, 'xx_plus_yy': 2, 'ecr': 1}

# Built-in multi-qubit decompositions (standard qelib1 forms)
def decomp(name, q, params):
    if name in ('ccx', 'toffoli'):
        a, b, c = q
        return [('h', (c,), ()), ('cx', (b, c), ()), ('tdg', (c,), ()), ('cx', (a, c), ()), ('t', (c,), ()),
                ('cx', (b, c), ()), ('tdg', (c,), ()), ('cx', (a, c), ()), ('t', (b,), ()), ('t', (c,), ()),
                ('h', (c,), ()), ('cx', (a, b), ()), ('t', (a,), ()), ('tdg', (b,), ()), ('cx', (a, b), ())]
    if name == 'ccz':
        a, b, c = q
        return [('h', (c,), ())] + decomp('ccx', q, ()) + [('h', (c,), ())]
    if name in ('cswap', 'fredkin'):
        a, b, c = q
        return [('cx', (c, b), ())] + decomp('ccx', (a, b, c), ()) + [('cx', (c, b), ())]
    if name == 'rccx':
        a, b, c = q
        pi = math.pi
        return [('u2', (c,), (0, pi)), ('u1', (c,), (pi/4,)), ('cx', (b, c), ()), ('u1', (c,), (-pi/4,)),
                ('cx', (a, c), ()), ('u1', (c,), (pi/4,)), ('cx', (b, c), ()), ('u1', (c,), (-pi/4,)),
                ('u2', (c,), (0, pi))]
    return None  # approximate elsewhere

# ---------------------------------------------------------------- parser
_STMT = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|[{};]')
_CALL = re.compile(r'^([A-Za-z_]\w*)\s*(?:\((.*)\))?\s*(.*)$', re.S)
_IDX = re.compile(r'([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$')
_CONSTANTS = {'pi': math.pi, 'tau': math.tau, 'e': math.e}
_FUNCTIONS = {'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
              'exp': math.exp, 'ln': math.log, 'sqrt': math.sqrt,
              'acos': math.acos, 'asin': math.asin, 'atan': math.atan}

def _eval(expr, env):
    """Evaluate a small arithmetic grammar; QASM input never reaches Python eval."""
    expr = expr.strip().replace('^', '**')
    if len(expr) > 128:
        raise ValueError('Angle expression too long')
    tree = ast.parse(expr, mode='eval')
    if len(list(ast.walk(tree))) > 40:
        raise ValueError('Angle expression too complex')
    def visit(node):
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in env and type(env[node.id]) in (int, float): return float(env[node.id])
            if node.id in _CONSTANTS: return _CONSTANTS[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
            if isinstance(node.op, ast.Pow) and abs(b) <= 8 and abs(a) <= 1e6: return a ** b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            f = _FUNCTIONS.get(node.func.id)
            if f is not None and len(node.args) == 1: return f(visit(node.args[0]))
        raise ValueError('Unsupported angle expression')
    value = visit(tree)
    if not math.isfinite(value) or abs(value) > 1e12:
        raise ValueError('Angle out of bounds')
    return float(value)

def _split(s):
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(''.join(cur).strip()); cur = []
        else:
            cur.append(ch)
    if ''.join(cur).strip():
        out.append(''.join(cur).strip())
    return out

class Parsed:
    def __init__(self):
        self.n = 0; self.ops = []; self.unresolved = 0; self.approx_multi = 0; self.unknown = 0
        self.stmt_seen = 0; self.truncated = False

def parse(text: str, max_ops=400_000, deadline=None):
    """Return Parsed with ops = list of (name, qubits, params) using only 1q/2q gates + ('multi', qubits)."""
    P = Parsed(); regs = {}; defs = {}
    pending = []; start = 0; body = None; head = None; in_if = 0
    def angle(expr, env):
        try:
            return _eval(expr, env)
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
            P.unresolved += 1
            return 0.3
    def operand(tok):
        tok = tok.strip()
        m = _IDX.match(tok)
        if m:
            off, size = regs[m.group(1)]; return [off + int(m.group(2))]
        if tok in regs:
            off, size = regs[tok]; return list(range(off, off+size))
        raise KeyError(tok)
    def emit(name, qs, params, depth=0):
        if len(P.ops) >= max_ops:
            P.truncated = True; raise StopIteration
        if name in defs and depth < 20:
            fparams, fq, fbody = defs[name]
            env = dict(zip(fparams, params)); qmap = dict(zip(fq, qs))
            for (gname, gparams, gargs) in fbody:
                vals = tuple(angle(p, env) for p in gparams)
                emit(gname, tuple(qmap[a] for a in gargs), vals, depth+1)
            return
        if name in ONEQ and len(qs) == 1:
            P.ops.append((name, qs, params)); return
        if name in TWOQ and len(qs) == 2:
            P.ops.append((name, qs, params)); return
        d = decomp(name, qs, params)
        if d is not None:
            for g in d:
                emit(*g, depth=depth+1)
            return
        if name in ('barrier', 'measure', 'reset', 'delay'):
            return
        # unknown or many-qubit (mcx, mcphase, c3sx, rcccx, ...): approximate by a controlled-phase chain
        if len(qs) >= 2:
            P.approx_multi += 1
            for a, b in zip(qs[:-1], qs[1:]):
                P.ops.append(('cp', (a, b), (math.pi/2,)))
        else:
            P.unknown += 1
    try:
        for i, tok in enumerate(_STMT.finditer(text)):
            if deadline and i % 2048 == 0 and time.perf_counter() > deadline:
                P.truncated = True; break
            d = tok.group()
            stmt = text[start:tok.start()].strip() if not pending else (''.join(pending) + text[start:tok.start()]).strip()
            pending = []; start = tok.end()
            if d.startswith('/'):
                pending.append(stmt + ' '); continue
            P.stmt_seen += 1
            if d == '{':
                if stmt.startswith('gate '):
                    m = re.match(r'gate\s+([A-Za-z_]\w*)\s*(?:\((.*?)\))?\s*(.*)$', stmt, re.S)
                    head = (m.group(1), [x.strip() for x in (m.group(2) or '').split(',') if x.strip()],
                            [x.strip() for x in m.group(3).split(',') if x.strip()])
                    body = []
                else:
                    in_if += 1  # if (...) { ... } : treat body gates as applied
                    if stmt:  # statement before '{' like "if (c)" - nothing else to do
                        pass
                continue
            if d == '}':
                if body is not None and head is not None and in_if == 0:
                    defs[head[0]] = (head[1], head[2], body); body = None; head = None
                elif in_if:
                    in_if -= 1
                if not stmt:
                    continue
            if not stmt:
                continue
            if body is not None and in_if == 0:
                m = _CALL.match(stmt)
                if m:
                    body.append((m.group(1), _split(m.group(2) or ''), [a.strip() for a in m.group(3).split(',') if a.strip()]))
                continue
            low = stmt.split(None, 1)[0] if stmt else ''
            if low in ('OPENQASM', 'include', 'creg', 'bit', 'barrier', 'measure', 'reset') or low.startswith('bit['):
                continue
            m = re.match(r'qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$', stmt) or None
            m3 = re.match(r'qubit\s*\[\s*(\d+)\s*\]\s*([A-Za-z_]\w*)$', stmt)
            m3b = re.match(r'qubit\s+([A-Za-z_]\w*)$', stmt)
            if m:
                size = int(m.group(2))
                if P.n + size > 4096: raise ValueError('Qubit limit exceeded')
                regs[m.group(1)] = (P.n, size); P.n += size; continue
            if m3:
                size = int(m3.group(1))
                if P.n + size > 4096: raise ValueError('Qubit limit exceeded')
                regs[m3.group(2)] = (P.n, size); P.n += size; continue
            if m3b:
                if P.n + 1 > 4096: raise ValueError('Qubit limit exceeded')
                regs[m3b.group(1)] = (P.n, 1); P.n += 1; continue
            if '=' in stmt and 'measure' in stmt:
                continue
            if stmt.startswith('if'):
                stmt = re.sub(r'^if\s*\(.*?\)\s*', '', stmt)  # QASM2 one-line if
                if not stmt:
                    continue
            m = _CALL.match(stmt)
            if not m:
                continue
            name = m.group(1)
            params = tuple(angle(p, {}) for p in _split(m.group(2) or ''))
            try:
                groups = [operand(a) for a in _split(m.group(3))]
            except Exception:
                P.unknown += 1; continue
            if not groups:
                continue
            w = max(len(g) for g in groups)
            for k in range(w):
                qs = tuple(g[k] if len(g) == w else g[0] for g in groups)
                if len(set(qs)) != len(qs):
                    continue
                emit(name, qs, params)
    except StopIteration:
        pass
    return P

# ---------------------------------------------------------------- structural bond bound
def structural(P: Parsed, thresholds=(16, 64, 512), deadline=None):
    """Upper bound on log2 bond dim per bond (fixed native order), updated gate by gate.
    cost_t ~ sum over 2q gates of (#bonds spanned) * min(2^b, t)^3 using the max bound touched."""
    n = max(P.n, 1)
    cap = np.minimum(np.arange(1, n), n - np.arange(1, n)).astype(float) if n > 1 else np.zeros(0)
    b = np.zeros(max(n-1, 0))
    cost = {t: 0.0 for t in thresholds}
    twoq = 0; spans = []
    for index, (name, qs, params) in enumerate(P.ops):
        if deadline is not None and index % 128 == 0 and time.perf_counter() > deadline:
            raise TimeoutError('Structural probe budget exhausted')
        if len(qs) != 2:
            continue
        i, j = sorted(qs); twoq += 1; spans.append(j - i)
        r = RANK2.get(name, 1)
        seg = slice(i, j)
        b[seg] = np.minimum(b[seg] + r, cap[seg])
        bm = b[seg].max() if j > i else 0.0
        for t in thresholds:
            chi = min(2.0 ** bm, t)
            cost[t] += (j - i) * chi ** 3
    out = {'n_qubits': P.n, 'twoq': twoq,
           'span_mean': float(np.mean(spans)) if spans else 0.0,
           'span_max': float(np.max(spans)) if spans else 0.0,
           'bound_max_log2': float(b.max()) if b.size else 0.0,
           'bound_mean_log2': float(b.mean()) if b.size else 0.0}
    for t in thresholds:
        out[f'struct_logcost_{t}'] = math.log10(cost[t] + 1.0)
        out[f'struct_frac_bonds_ge_{t}'] = float((b >= math.log2(t)).mean()) if b.size else 0.0
    return out

# ---------------------------------------------------------------- truncated MPS probe
class MPS:
    def __init__(self, n, chi):
        self.n, self.chi = n, chi
        self.A = [np.zeros((1, 2, 1), complex) for _ in range(n)]
        for a in self.A:
            a[0, 0, 0] = 1
        self.discarded = 0.0; self.svds = 0; self.flops = 0.0; self.max_entropy = 0.0
        self.trunc_events = 0; self.center = 0

    def one(self, U, i):
        self.A[i] = np.einsum('ab,xby->xay', U, self.A[i])

    def move_center(self, k):
        """Keep mixed-canonical form so local singular values are true Schmidt values."""
        while self.center < k:
            c = self.center; a = self.A[c]; l, _, r = a.shape
            Q, R = np.linalg.qr(a.reshape(l*2, r))
            self.A[c] = Q.reshape(l, 2, Q.shape[1])
            self.A[c+1] = np.einsum('ab,bcd->acd', R, self.A[c+1]); self.center += 1
        while self.center > k:
            c = self.center; a = self.A[c]; l, _, r = a.shape
            Q, R = np.linalg.qr(a.reshape(l, 2*r).T)
            self.A[c] = Q.T.reshape(Q.shape[1], 2, r)
            self.A[c-1] = np.einsum('abc,cd->abd', self.A[c-1], R.T); self.center -= 1

    def two_adj(self, U4, i):
        """Apply U4 on (i, i+1) with i the first index in the kron."""
        self.move_center(i)
        a, b = self.A[i], self.A[i+1]
        l, r = a.shape[0], b.shape[2]
        th = np.einsum('xay,ybz->xabz', a, b)
        th = np.einsum('abcd,xcdz->xabz', U4.reshape(2, 2, 2, 2), th)
        M = th.reshape(l*2, 2*r)
        self.flops += (l*2) * (2*r) * min(l*2, 2*r)
        try:
            Uu, s, Vh = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            return
        self.svds += 1
        nrm = float((s**2).sum()) or 1.0
        keep = int((s > 1e-10 * s[0]).sum()) if s.size else 1
        keep = max(1, keep)
        if keep > self.chi:
            self.discarded += float((s[self.chi:keep]**2).sum()) / nrm
            self.trunc_events += 1
            keep = self.chi
        s = s[:keep]; p = s**2 / (s**2).sum()
        ent = float(-(p * np.log2(p + 1e-300)).sum())
        self.max_entropy = max(self.max_entropy, ent)
        s = s / math.sqrt((s**2).sum())
        self.A[i] = Uu[:, :keep].reshape(l, 2, keep)
        self.A[i+1] = (s[:, None] * Vh[:keep]).reshape(keep, 2, r)
        self.center = i + 1

    def two(self, U4, i, j):
        # U4 defined with qubit i as first (control) factor
        if i > j:
            U4 = SWAP @ U4 @ SWAP; i, j = j, i
        # move j left to i+1 via swaps, apply, move back
        for k in range(j-1, i, -1):
            self.two_adj(SWAP, k)
        self.two_adj(U4, i)
        for k in range(i+1, j):
            self.two_adj(SWAP, k)

    def bonds(self):
        return np.array([self.A[k].shape[2] for k in range(self.n-1)])

def probe(P: Parsed, chi=16, budget_s=3.0, max_qubits=1024):
    t0 = time.perf_counter(); dead = t0 + budget_s
    out = {'probe_chi': chi}
    n = P.n
    if n < 2 or n > max_qubits:
        out.update({'probe_ok': 0}); return out
    m = MPS(n, chi)
    twoq_total = sum(1 for o in P.ops if len(o[1]) == 2)
    done2 = 0; first_sat = None; done = 0
    cache = {}
    for idx, (name, qs, params) in enumerate(P.ops):
        if time.perf_counter() > dead:  # check every gate so the budget is respected
            break
        key = (name, params)
        U = cache.get(key)
        if U is None:
            f = ONEQ.get(name) if len(qs) == 1 else TWOQ.get(name)
            try:
                U = f(*params)
            except TypeError:
                U = f(*(list(params) + [0.0]*4)[:f.__code__.co_argcount]) if f.__code__.co_argcount else f()
            cache[key] = U
        if len(qs) == 1:
            m.one(U, qs[0])
        else:
            m.two(U, qs[0], qs[1]); done2 += 1
            if first_sat is None and m.trunc_events > 0:
                first_sat = done2
        done = idx + 1
    bd = m.bonds()
    out.update({
        'probe_ok': 1,
        'probe_frac_ops_done': done / max(len(P.ops), 1),
        'probe_twoq_done': done2,
        'probe_frac_bonds_saturated': float((bd >= chi).mean()),
        'probe_mean_log2_bond': float(np.log2(bd).mean()),
        'probe_max_entropy': m.max_entropy,
        'probe_discarded': m.discarded,
        'probe_log_discarded': math.log10(m.discarded + 1e-12),
        'probe_trunc_frac': m.trunc_events / max(m.svds, 1),
        'probe_first_sat_frac': (first_sat / max(twoq_total, 1)) if first_sat else 1.0,
        'probe_log_flops': math.log10(m.flops + 1.0),
        'probe_seconds': time.perf_counter() - t0,
    })
    return out

def featurize_file(path, chi=16, parse_budget=4.0, probe_budget=3.0, max_ops=400_000):
    t0 = time.perf_counter()
    with open(path, 'rb') as fh:
        raw = fh.read()
    total_lines = raw.count(b';')
    text = raw[:20_000_000].decode('utf-8', 'replace')  # prefix cap bounds memory/time on huge files
    del raw
    P = parse(text, max_ops=max_ops, deadline=t0 + parse_budget)
    t1 = time.perf_counter()
    feats = {'parse_seconds': t1 - t0, 'ops_parsed': len(P.ops), 'stmt_seen': P.stmt_seen,
             'stmt_total': total_lines, 'parse_frac': min(1.0, P.stmt_seen / max(total_lines, 1)),
             'parse_truncated': int(P.truncated), 'approx_multi': P.approx_multi, 'unknown_ops': P.unknown}
    feats.update(structural(P))
    t2 = time.perf_counter(); feats['struct_seconds'] = t2 - t1
    feats.update(probe(P, chi=chi, budget_s=probe_budget))
    feats['total_seconds'] = time.perf_counter() - t0
    return feats


def featurize_text(text: str, chi=16, parse_budget=4.0, probe_budget=1.0, deadline=None):
    """Extract bounded features for a new circuit or signal an unsafe fallback.

    The original research files were generated from the first 20 MB of QASM.
    Unsupported operations and unresolved angles deliberately disable the probe.
    """
    start = time.perf_counter()
    parse_deadline = min(start + parse_budget, deadline - 1.5) if deadline is not None else start + parse_budget
    if parse_deadline <= start:
        return {'usable': False, 'reason': 'probe budget exhausted'}
    parsed = parse(text[:20_000_000], max_ops=400_000, deadline=parse_deadline)
    if parsed.unknown or parsed.approx_multi or parsed.unresolved or parsed.n < 2:
        return {'usable': False, 'reason': 'unsupported circuit syntax'}
    structural_deadline = deadline - 1.25 if deadline is not None else None
    features = structural(parsed, deadline=structural_deadline)
    if deadline is not None and deadline - time.perf_counter() < 1.1:
        return {'usable': False, 'reason': 'probe budget exhausted'}
    features.update(probe(parsed, chi=chi, budget_s=probe_budget))
    features['usable'] = bool(features.get('probe_ok'))
    return features
