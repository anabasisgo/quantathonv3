"""Entanglement-based probe features.

1. Structural bond bound (no simulation): walking the 2-qubit gates in file order, each cut's possible
   bond dimension grows by at most 1 bit per controlled-type gate (2 for swap-like gates), capped by
   min(qubits left, qubits right). Gives cost estimates at each simulator threshold.
2. Truncated matrix-product-state simulation: the circuit's gates are applied to an MPS whose bond
   dimension is capped (default 16), under a deterministic work budget. How quickly and how widely the
   bonds saturate, and how much weight is truncated, indicates how hard the real simulator will find the
   circuit at larger thresholds. The simulator is exact when no truncation happens.
"""
import math
import time

import numpy as np

from . import config
from .gates import SCHMIDT_BITS, SWAP, matrix

NAN = float('nan')


def bond_bound(C, thresholds=config.THRESHOLDS):
    n = C.n
    out = {k: NAN for k in config.BOUND_FEATURES}
    if n < 2:
        return out
    cap = np.minimum(np.arange(1, n), n - np.arange(1, n)).astype(float)
    b = np.zeros(n - 1)
    cost = dict.fromkeys(thresholds, 0.0)
    spans = []
    for name, qs, _ in C.ops:
        if len(qs) != 2:
            continue
        i, j = sorted(qs)
        spans.append(j - i)
        b[i:j] = np.minimum(b[i:j] + SCHMIDT_BITS.get(name, 1), cap[i:j])
        chi = 2.0 ** b[i:j].max()
        for t in thresholds:
            cost[t] += (j - i) * min(chi, t) ** 3
    out.update(span_mean=float(np.mean(spans)) if spans else 0.0, span_max=float(max(spans, default=0)),
               bound_max_log2=float(b.max()), bound_mean_log2=float(b.mean()))
    for t in thresholds:
        out[f'struct_logcost_{t}'] = math.log10(cost[t] + 1.0)
        out[f'struct_frac_bonds_ge_{t}'] = float((b >= math.log2(t)).mean())
    return out


class MPS:
    """Matrix product state kept in mixed-canonical form, so local singular values are true Schmidt values."""

    def __init__(self, n, chi):
        self.n, self.chi = n, chi
        self.A = [np.zeros((1, 2, 1), complex) for _ in range(n)]
        for a in self.A:
            a[0, 0, 0] = 1
        self.center = 0
        self.discarded = self.flops = self.max_entropy = 0.0
        self.svds = self.trunc_events = 0

    def one(self, U, i):
        self.A[i] = np.einsum('ab,xby->xay', U, self.A[i])

    def _move_center(self, k):
        while self.center < k:
            c = self.center
            l, _, r = self.A[c].shape
            Q, R = np.linalg.qr(self.A[c].reshape(l * 2, r))
            self.A[c] = Q.reshape(l, 2, Q.shape[1])
            self.A[c + 1] = np.einsum('ab,bcd->acd', R, self.A[c + 1])
            self.center += 1
        while self.center > k:
            c = self.center
            l, _, r = self.A[c].shape
            Q, R = np.linalg.qr(self.A[c].reshape(l, 2 * r).T)
            self.A[c] = Q.T.reshape(Q.shape[1], 2, r)
            self.A[c - 1] = np.einsum('abc,cd->abd', self.A[c - 1], R.T)
            self.center -= 1

    def _two_adjacent(self, U4, i):
        self._move_center(i)
        a, b = self.A[i], self.A[i + 1]
        l, r = a.shape[0], b.shape[2]
        theta = np.einsum('abcd,xcdz->xabz', U4.reshape(2, 2, 2, 2), np.einsum('xay,ybz->xabz', a, b))
        M = theta.reshape(l * 2, 2 * r)
        self.flops += (l * 2) * (2 * r) * min(l * 2, 2 * r)
        try:
            Uu, s, Vh = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            return
        self.svds += 1
        norm = float((s ** 2).sum()) or 1.0
        keep = max(1, int((s > 1e-10 * s[0]).sum())) if s.size else 1
        if keep > self.chi:
            self.discarded += float((s[self.chi:keep] ** 2).sum()) / norm
            self.trunc_events += 1
            keep = self.chi
        s = s[:keep]
        p = s ** 2 / (s ** 2).sum()
        self.max_entropy = max(self.max_entropy, float(-(p * np.log2(p + 1e-300)).sum()))
        s = s / math.sqrt((s ** 2).sum())
        self.A[i] = Uu[:, :keep].reshape(l, 2, keep)
        self.A[i + 1] = (s[:, None] * Vh[:keep]).reshape(keep, 2, r)
        self.center = i + 1

    def two(self, U4, i, j):
        """Apply a 2-qubit gate (first tensor factor on qubit i); distant qubits are swapped next to each other and back."""
        if i > j:
            U4, i, j = SWAP @ U4 @ SWAP, j, i
        for k in range(j - 1, i, -1):
            self._two_adjacent(SWAP, k)
        self._two_adjacent(U4, i)
        for k in range(i + 1, j):
            self._two_adjacent(SWAP, k)

    def bonds(self):
        return np.array([self.A[k].shape[2] for k in range(self.n - 1)])


def simulate(C, chi=config.PROBE_BOND_CAP, work_budget=config.PROBE_WORK_BUDGET):
    out = {k: NAN for k in config.PROBE_FEATURES}
    if C.n < 2 or not C.ops:
        return out
    m = MPS(C.n, chi)
    twoq_total = sum(1 for op in C.ops if len(op[1]) == 2)
    wall_stop = time.perf_counter() + config.PROBE_SAFETY_WALL_S
    cache, done, twoq_done, first_sat = {}, 0, 0, None
    for name, qs, params in C.ops:
        work = config.FLOP_COST * m.flops + config.OP_COST * done + config.SVD_COST * m.svds
        if work > work_budget or time.perf_counter() > wall_stop:
            break
        key = (name, params)
        U = cache.get(key)
        if U is None:
            U = cache[key] = matrix(name, params)
        if len(qs) == 1:
            m.one(U, qs[0])
        else:
            m.two(U, qs[0], qs[1])
            twoq_done += 1
            if first_sat is None and m.trunc_events:
                first_sat = twoq_done
        done += 1
    bonds = m.bonds()
    out.update(
        probe_frac_bonds_saturated=float((bonds >= chi).mean()),
        probe_mean_log2_bond=float(np.log2(bonds).mean()),
        probe_max_entropy=m.max_entropy,
        probe_log_discarded=math.log10(m.discarded + 1e-12),
        probe_trunc_frac=m.trunc_events / max(m.svds, 1),
        probe_first_sat_frac=first_sat / max(twoq_total, 1) if first_sat else 1.0,
        probe_log_flops=math.log10(m.flops + 1.0),
        probe_frac_ops_done=done / len(C.ops),
    )
    return out
