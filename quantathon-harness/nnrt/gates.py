"""Gate matrices and multi-qubit decompositions used by the probe simulator."""
import math

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * math.pi / 4)])
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])


def rx(t): return np.array([[math.cos(t / 2), -1j * math.sin(t / 2)], [-1j * math.sin(t / 2), math.cos(t / 2)]])
def ry(t): return np.array([[math.cos(t / 2), -math.sin(t / 2)], [math.sin(t / 2), math.cos(t / 2)]], complex)
def rz(t): return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])
def phase(t): return np.diag([1, np.exp(1j * t)])


def u3(t, p, l):
    return np.array([[math.cos(t / 2), -np.exp(1j * l) * math.sin(t / 2)],
                     [np.exp(1j * p) * math.sin(t / 2), np.exp(1j * (p + l)) * math.cos(t / 2)]])


ONEQ = {
    'h': lambda: H, 'x': lambda: X, 'y': lambda: Y, 'z': lambda: Z, 's': lambda: S, 'sdg': lambda: S.conj(),
    't': lambda: T, 'tdg': lambda: T.conj(), 'sx': lambda: SX, 'sxdg': lambda: SX.conj().T, 'id': lambda: I2,
    'rx': rx, 'ry': ry, 'rz': rz, 'p': phase, 'phase': phase, 'u1': phase,
    'u2': lambda p, l: u3(math.pi / 2, p, l), 'u3': u3, 'u': u3, 'U': u3,
    'r': lambda t, p: u3(t, p - math.pi / 2, -p + math.pi / 2),
}


def controlled(U):
    M = np.eye(4, dtype=complex)
    M[2:, 2:] = U
    return M


def pauli_rotation(P, t):  # exp(-i t/2 P) for a two-qubit Pauli product P
    return math.cos(t / 2) * np.eye(4) - 1j * math.sin(t / 2) * P


XX, YY, ZZ, ZX = np.kron(X, X), np.kron(Y, Y), np.kron(Z, Z), np.kron(Z, X)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex)
ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], complex)
ECR = np.array([[0, 1, 0, 1j], [1, 0, -1j, 0], [0, 1j, 0, 1], [-1j, 0, 1, 0]], complex) / math.sqrt(2)


def xx_pm_yy(t, b, sign):
    M = np.eye(4, dtype=complex)
    c, s = math.cos(t / 2), math.sin(t / 2)
    i, j = (0, 3) if sign < 0 else (1, 2)  # xx_minus_yy mixes |00>,|11>; xx_plus_yy mixes |01>,|10>
    M[i, i] = M[j, j] = c
    M[i, j] = -1j * s * np.exp(-1j * b)
    M[j, i] = -1j * s * np.exp(1j * b)
    return M


TWOQ = {
    'cx': lambda: controlled(X), 'cy': lambda: controlled(Y), 'cz': lambda: controlled(Z), 'ch': lambda: controlled(H),
    'csx': lambda: controlled(SX), 'cs': lambda: controlled(S), 'csdg': lambda: controlled(S.conj()),
    'swap': lambda: SWAP, 'iswap': lambda: ISWAP, 'ecr': lambda: ECR,
    'dcx': lambda: controlled(X) @ SWAP @ controlled(X) @ SWAP,
    'rxx': lambda t: pauli_rotation(XX, t), 'ryy': lambda t: pauli_rotation(YY, t),
    'rzz': lambda t: pauli_rotation(ZZ, t), 'rzx': lambda t: pauli_rotation(ZX, t),
    'crx': lambda t: controlled(rx(t)), 'cry': lambda t: controlled(ry(t)), 'crz': lambda t: controlled(rz(t)),
    'cp': lambda t: controlled(phase(t)), 'cu1': lambda t: controlled(phase(t)), 'cphase': lambda t: controlled(phase(t)),
    'cu3': lambda t, p, l: controlled(u3(t, p, l)),
    'cu': lambda t, p, l, g=0.0: controlled(np.exp(1j * g) * u3(t, p, l)),
    'xx_minus_yy': lambda t, b=0.0: xx_pm_yy(t, b, -1), 'xx_plus_yy': lambda t, b=0.0: xx_pm_yy(t, b, +1),
}
# log2 of the operator-Schmidt rank: how many bits of entanglement a gate can add across a cut
SCHMIDT_BITS = {'swap': 2, 'iswap': 2, 'dcx': 2, 'xx_plus_yy': 2}


def matrix(name, params):
    """Unitary for a 1- or 2-qubit gate; missing trailing parameters default to 0."""
    f = ONEQ.get(name) or TWOQ[name]
    try:
        return f(*params)
    except TypeError:
        argc = f.__code__.co_argcount
        return f(*(list(params) + [0.0] * argc)[:argc])


def decompose(name, q):
    """Standard qelib1 decompositions of common 3-qubit gates into 1q/2q gates, or None."""
    pi = math.pi
    if name in ('ccx', 'toffoli'):
        a, b, c = q
        return [('h', (c,), ()), ('cx', (b, c), ()), ('tdg', (c,), ()), ('cx', (a, c), ()), ('t', (c,), ()),
                ('cx', (b, c), ()), ('tdg', (c,), ()), ('cx', (a, c), ()), ('t', (b,), ()), ('t', (c,), ()),
                ('h', (c,), ()), ('cx', (a, b), ()), ('t', (a,), ()), ('tdg', (b,), ()), ('cx', (a, b), ())]
    if name == 'ccz':
        return [('h', (q[2],), ())] + decompose('ccx', q) + [('h', (q[2],), ())]
    if name in ('cswap', 'fredkin'):
        a, b, c = q
        return [('cx', (c, b), ())] + decompose('ccx', (a, b, c)) + [('cx', (c, b), ())]
    if name == 'rccx':
        a, b, c = q
        return [('u2', (c,), (0, pi)), ('u1', (c,), (pi / 4,)), ('cx', (b, c), ()), ('u1', (c,), (-pi / 4,)),
                ('cx', (a, c), ()), ('u1', (c,), (pi / 4,)), ('cx', (b, c), ()), ('u1', (c,), (-pi / 4,)),
                ('u2', (c,), (0, pi))]
    return None
