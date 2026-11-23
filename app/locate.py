"""Locate the shortest contiguous *playable* shot segment whose signed
contribution reaches a requested fraction of the video-wide positive
contribution.

For every playable shot i define, over the playable set V of the whole
video,

    w_i = softmax(attention score)_i            (weights sum to 1 over V)
    c_i = w_i * dot(feature_i, query)           (signed contribution)

A segment is a contiguous run of the original shot indices in which every
shot is playable (hidden shots split runs).  Its contribution is the sum of
the signed c_i inside it, so negative contributions count against the
segment.  Among all segments whose contribution reaches

    target = positive_total * ratio,   positive_total = sum_{c_i > 0} c_i

we return the shortest one, breaking ties by the earliest start.

Numerics
--------
Window sums are differences of prefix sums.  Plain doubles cannot be used
for them: a decisive tiny term (say 1e-12) added on top of an O(1) prefix
sits below double ulp and vanishes from every later subtraction, however
compensated (verified empirically -- even math.fsum over stored prefixes
cannot recover digits already lost).  We therefore keep every prefix as a
*double-double* pair (hi, lo) (~31 significant decimal digits, a constant
handful of flops per add), in which 1e-14-scale terms survive exactly.
Threshold comparisons then only need a relative 1e-12 tolerance for the
few-ulp rounding of the contributions themselves; no absolute floor is
needed, so genuinely tiny targets are never dwarfed.

Complexity
----------
O(N + K), N = shot count, K = effective orders: coverage diff array
O(N+K), one linear pass for weights/contributions, one monotonic-deque
shortest-subarray scan with constant-work double-double steps.

The snapshot (tree root softmax aggregates + order ranges) is taken under
the service lock; everything here runs on that immutable snapshot, so
orders registered/revoked while a locate call is in flight do not affect
its result and the lock is never held during the O(N) work.
"""

import math
from collections import deque
from dataclasses import dataclass

DIM = 16

# Status codes returned to the API.
OK = "ok"
NO_POSITIVE = "no_positive_contribution"
NO_QUALIFYING = "no_qualifying_interval"


@dataclass(frozen=True)
class LocateSnapshot:
    """Immutable view of one video at the moment the locate call started."""

    n: int
    scores: object          # array('d'), length n (never mutated after build)
    features: object        # array('d'), n * DIM (never mutated after build)
    root_max: float         # max attention score among playable shots
    root_sum: float         # sum of exp(score - root_max) over playable shots
    visible_count: int
    orders: tuple           # tuple of (start, end) effective restriction ranges


@dataclass(frozen=True)
class LocateResult:
    status: str
    start: int | None
    end: int | None
    achieved_ratio: float | None
    positive_total: float
    visible_count: int
    message: str


def locate_segment(snap: LocateSnapshot, query, ratio: float) -> LocateResult:
    """Find the shortest qualifying playable segment on a snapshot."""
    n = snap.n
    visible = snap.visible_count
    if visible == 0:
        return LocateResult(
            NO_POSITIVE, None, None, None, 0.0, 0,
            "全片无可播镜头，不存在正贡献",
        )

    # ---- coverage diff array: shot i is hidden iff cover[i] > 0 ----------
    diff = [0] * (n + 1)
    for s, e in snap.orders:
        diff[s] += 1
        diff[e] -= 1

    # ---- per-shot softmax weight (same numerics as the retrieval vector) --
    m = snap.root_max
    z = snap.root_sum
    scores = snap.scores
    features = snap.features
    # Playability is tracked separately from the weight: a shot with an
    # extremely negative attention score has a softmax weight that underflows
    # to 0.0, yet it is still playable and must NOT split a contiguous run.
    playable = [False] * n
    contrib = [0.0] * n
    pos_hi = pos_lo = 0.0  # double-double accumulation of positive total
    cover = 0
    for i in range(n):
        cover += diff[i]
        if cover > 0:
            continue  # hidden: splits playable runs
        playable[i] = True
        w = math.exp(scores[i] - m) / z  # may legitimately underflow to 0.0
        base = i * DIM
        dot = 0.0
        for k in range(DIM):
            dot += features[base + k] * query[k]
        c = w * dot
        contrib[i] = c
        if c > 0.0:  # inlined double-double add of (c, 0.0)
            s = pos_hi + c
            bb = s - pos_hi
            e = (pos_hi - (s - bb)) + (c - bb) + pos_lo
            tt = s + e
            pos_hi, pos_lo = tt, e - (tt - s)

    positive_total = pos_hi + pos_lo
    if positive_total <= 0.0:
        return LocateResult(
            NO_POSITIVE, None, None, None, 0.0, visible,
            "全片可播镜头对该查询向量均无正贡献，无法按正贡献占比定位",
        )

    target = positive_total * ratio
    # Relative tolerance for the few-ulp rounding of the stored doubles;
    # no absolute floor, so a genuinely tiny target keeps its full meaning.
    eps = 1e-12 * positive_total

    # ---- shortest qualifying subarray per maximal playable run ------------
    # Two structures per run:
    #   * pre_hi/pre_lo: double-double prefix sums as lists (one Knuth
    #     two-sum per shot), giving exact window sums even under severe
    #     cancellation;
    #   * a collections.deque of prefix indices with strictly increasing
    #     double-double prefix values.  Dominance is decided on the exact
    #     dd pair (two distinct dd prefixes can share one plain-double high
    #     part under cancellation), and both ends move in O(1).
    # The front is popped while the double-double window sum reaches target
    # (+ eps).  Each prefix enters and leaves the deque at most once, hence
    # strictly O(N) work with a constant number of flops per shot.
    # Equal-length ties keep the earliest start (right ends scanned in
    # order; best_len replaced only on a strict improvement).
    best_len = n + 1
    best_start = -1
    best_end = -1
    best_window = 0.0

    i = 0
    while i < n:
        if not playable[i]:  # hidden shot: run boundary
            i += 1
            continue
        pre_hi = [0.0]          # double-double prefix per run position
        pre_lo = [0.0]
        dq = deque([0])         # prefix indices into pre_hi/pre_lo
        phi = plo = 0.0         # running double-double prefix
        j = i
        while j < n and playable[j]:
            cj = contrib[j]
            # inlined double-double add of (cj, 0.0) onto (phi, plo)
            s = phi + cj
            bb = s - phi
            e = (phi - (s - bb)) + (cj - bb) + plo
            t = s + e
            phi, plo = t, e - (t - s)
            pre_hi.append(phi)
            pre_lo.append(plo)
            r = j - i + 1  # prefix index of the window ending at shot j
            while dq:
                a_p = dq[0]
                # exact window sum = dd(phi,plo) - dd(ahi,alo): Knuth
                # two-sum on the high parts, then fold in both low parts
                ahi = pre_hi[a_p]
                alo = pre_lo[a_p]
                wh = phi - ahi
                b = wh - phi
                err = (phi - (wh - b)) - ahi - b
                wl = err + plo - alo
                tt = wh + wl
                window = tt + (wl - (tt - wh))
                if window + eps < target:
                    break
                dq.popleft()
                length = r - a_p  # window shots: [i+a_p .. j]
                if length < best_len:
                    best_len = length
                    best_start = i + a_p
                    best_end = j + 1
                    best_window = window
            # strictly increasing double-double prefixes
            while dq:
                tail = dq[-1]
                thi, tlo = pre_hi[tail], pre_lo[tail]
                if phi > thi or (phi == thi and plo > tlo):
                    break
                dq.pop()
            dq.append(r)
            if best_len == 1:
                break  # no segment can be shorter; earliest one already kept
            j += 1
        if best_len == 1:
            break
        i = j + 1

    if best_start < 0:
        return LocateResult(
            NO_QUALIFYING, None, None, None, positive_total, visible,
            "存在正贡献，但没有任何连续可播区间（含段内负贡献相抵后）能达到所填占比",
        )

    segment_sum = best_window
    achieved = segment_sum / positive_total
    return LocateResult(
        OK, best_start, best_end, achieved, positive_total, visible,
        "定位成功",
    )
