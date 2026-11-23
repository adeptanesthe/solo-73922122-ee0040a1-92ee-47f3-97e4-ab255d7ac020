"""Self-tests: hiding/recovery correctness, overlap semantics, idempotency,
conflict handling, withdrawn marker, <=1e-9 agreement with a stable full
recomputation, O(log N) update timing, and the locate API (shortest
playable segment, signed contributions, tie-breaking, failure statuses,
O(N+K) timing)."""

import math
import random
import time

import pytest
from fastapi.testclient import TestClient

from app.locate import (DIM as LDIM, LocateSnapshot, NO_POSITIVE,
                        NO_QUALIFYING, OK, locate_segment)
from app.main import app
from app.segment_tree import DIM, VisibleSoftmaxTree


# ------------------------------------------------------------- reference

def brute_force_vector(features, scores, hidden):
    """Stable full recomputation: max-shift + math.fsum (exact-ish sums)."""
    vis = [i for i in range(len(scores)) if i not in hidden]
    if not vis:
        return None
    m = max(scores[i] for i in vis)
    ws = [(i, math.exp(scores[i] - m)) for i in vis]
    s = math.fsum(w for _, w in ws)
    return [math.fsum(w * features[i][k] for i, w in ws) / s for k in range(DIM)]


def assert_close(tree_vec, ref_vec, tol=1e-9):
    if ref_vec is None:
        assert tree_vec is None
        return
    assert tree_vec is not None
    diffs = [abs(a - b) for a, b in zip(tree_vec, ref_vec)]
    assert max(diffs) <= tol, f"max dim diff {max(diffs):.3e} > {tol}"


# ---------------------------------------------------- tree vs brute force

@pytest.mark.parametrize("seed", range(40))
def test_tree_matches_stable_full_recompute(seed):
    rng = random.Random(seed)
    n = rng.choice([1, 2, 3, 5, 17, 64, 100, 333, 1000])
    features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
    scores = [rng.uniform(-1000, 1000) for _ in range(n)]
    tree = VisibleSoftmaxTree(features, scores)
    hidden = set()
    assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, hidden))

    active = []  # (start, end) of live orders
    for _ in range(30):
        if active and rng.random() < 0.4:
            i = rng.randrange(len(active))
            s, e = active.pop(i)
            tree.add_cover(s, e, -1)
            for j in range(s, e):
                hidden.discard(j)  # rebuilt below anyway
        else:
            s = rng.randrange(0, n)
            e = rng.randrange(s + 1, n + 1)
            tree.add_cover(s, e, +1)
            active.append((s, e))
        # rebuild the hidden set from scratch (reference counting semantics)
        hidden = set()
        for s, e in active:
            hidden.update(range(s, e))
        assert tree.visible_count() == n - len(hidden)
        assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, hidden))


def test_tree_extreme_scores_and_features():
    rng = random.Random(7)
    n = 500
    features = [[rng.choice([-1.0, 1.0]) for _ in range(DIM)] for _ in range(n)]
    scores = [rng.choice([-1000.0, 1000.0]) for _ in range(n)]
    tree = VisibleSoftmaxTree(features, scores)
    hidden = set()
    active = []
    for _ in range(20):
        s = rng.randrange(0, n)
        e = rng.randrange(s + 1, n + 1)
        tree.add_cover(s, e, +1)
        active.append((s, e))
        hidden.update(range(s, e))
        assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, hidden))
    # revoke everything -> must recover the full-set vector exactly-ish
    for s, e in active:
        tree.add_cover(s, e, -1)
    assert tree.visible_count() == n
    assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, set()))


def test_tree_all_hidden_returns_none():
    tree = VisibleSoftmaxTree([[0.5] * DIM, [-0.5] * DIM], [3.0, -3.0])
    tree.add_cover(0, 2, +1)
    assert tree.visible_count() == 0
    assert tree.retrieval_vector() is None
    tree.add_cover(0, 2, -1)
    assert tree.visible_count() == 2
    assert tree.retrieval_vector() is not None


def test_tree_large_n_accuracy():
    rng = random.Random(99)
    n = 100_000
    features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
    scores = [rng.uniform(-1000, 1000) for _ in range(n)]
    tree = VisibleSoftmaxTree(features, scores)
    hidden = set()
    for _ in range(300):
        s = rng.randrange(0, n)
        e = min(n, s + rng.randrange(1, 5000))
        tree.add_cover(s, e, +1)
        hidden.update(range(s, e))
    assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, hidden))


# ------------------------------------------------------------------- API

@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _shots(n, seed=0):
    rng = random.Random(seed)
    return [
        {"feature": [rng.uniform(-1, 1) for _ in range(DIM)],
         "score": rng.uniform(-1000, 1000)}
        for _ in range(n)
    ]


def test_api_hide_overlap_revoke_recover(client):
    shots = _shots(8, seed=1)
    r = client.post("/videos/v1/shots", json={"shots": shots})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["visible_count"] == 8
    initial = body["vector"]

    feats = [s["feature"] for s in shots]
    scores = [s["score"] for s in shots]

    # order A hides [2,6), order B hides [4,8)  (overlap [4,6))
    assert client.post("/videos/v1/restrictions",
                       json={"order_id": "A", "start": 2, "end": 6}).json()["visible_count"] == 4
    assert client.post("/videos/v1/restrictions",
                       json={"order_id": "B", "start": 4, "end": 8}).json()["visible_count"] == 2

    # duplicate registration of A: no state change
    dup = client.post("/videos/v1/restrictions", json={"order_id": "A", "start": 2, "end": 6})
    assert dup.status_code == 200 and dup.json()["visible_count"] == 2

    # same id, different range: conflict
    conflict = client.post("/videos/v1/restrictions", json={"order_id": "A", "start": 2, "end": 7})
    assert conflict.status_code == 409

    # revoke A: [2,4) playable again, [4,8) still hidden by B
    r = client.delete("/videos/v1/restrictions/A")
    body = r.json()
    assert body["visible_count"] == 4
    ref = brute_force_vector(feats, scores, set(range(4, 8)))
    assert max(abs(a - b) for a, b in zip(body["vector"], ref)) <= 1e-9

    # revoke B: fully recovered, vector identical to the import-time one
    r = client.delete("/videos/v1/restrictions/B")
    body = r.json()
    assert body["visible_count"] == 8
    assert max(abs(a - b) for a, b in zip(body["vector"], initial)) <= 1e-9

    # unknown order / unknown video
    assert client.delete("/videos/v1/restrictions/B").status_code == 404
    assert client.get("/videos/nope/retrieval-vector").status_code == 404


def test_api_withdrawn_marker(client):
    client.post("/videos/v2/shots", json={"shots": _shots(3, seed=2)})
    r = client.post("/videos/v2/restrictions", json={"order_id": "ALL", "start": 0, "end": 3})
    body = r.json()
    assert body["status"] == "withdrawn" and body["vector"] is None and body["visible_count"] == 0
    r = client.delete("/videos/v2/restrictions/ALL")
    body = r.json()
    assert body["status"] == "ok" and body["vector"] is not None


def test_api_validation(client):
    client.post("/videos/v3/shots", json={"shots": _shots(4, seed=3)})
    bad_dim = {"feature": [0.0] * 15, "score": 0.0}
    assert client.post("/videos/v4/shots", json={"shots": [bad_dim]}).status_code == 422
    bad_val = {"feature": [1.5] * DIM, "score": 0.0}
    assert client.post("/videos/v4/shots", json={"shots": [bad_val]}).status_code == 422
    bad_score = {"feature": [0.0] * DIM, "score": 1000.1}
    assert client.post("/videos/v4/shots", json={"shots": [bad_score]}).status_code == 422
    assert client.post("/videos/v3/restrictions",
                       json={"order_id": "X", "start": 3, "end": 3}).status_code == 400
    assert client.post("/videos/v3/restrictions",
                       json={"order_id": "X", "start": 0, "end": 5}).status_code == 400
    assert client.post("/videos/v3/shots", json={"shots": []}).status_code == 422


# ------------------------------------------------------------- performance

def test_update_is_log_n():
    """Register/revoke must stay O(log N): per-op cost at N=100k should be
    roughly the same as at N=10k (factor << the 10x linear growth)."""
    timings = {}
    for n in (10_000, 100_000):
        rng = random.Random(5)
        features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
        scores = [rng.uniform(-1000, 1000) for _ in range(n)]
        tree = VisibleSoftmaxTree(features, scores)
        ops = 300
        ranges = []
        for _ in range(ops):
            s = rng.randrange(0, n)
            e = min(n, s + rng.randrange(1, 2000))
            ranges.append((s, e))
        t0 = time.perf_counter()
        for s, e in ranges:
            tree.add_cover(s, e, +1)
        for s, e in ranges:
            tree.add_cover(s, e, -1)
        dt = time.perf_counter() - t0
        timings[n] = dt / (2 * ops)
        print(f"\nN={n}: {ops} register + {ops} revoke in {dt:.3f}s "
              f"-> {dt / (2 * ops) * 1e6:.1f} us/op")
    assert timings[100_000] < 5e-3  # generous absolute bound: 5 ms per op
    assert timings[100_000] < 6 * timings[10_000]  # far below 10x linear growth


# =============================================================== locate ====

def _locate_oracle(features, scores, hidden, query, ratio):
    """O(N^2) exact reference: fsum window sums, shortest then earliest."""
    n = len(scores)
    vis = [i for i in range(n) if i not in hidden]
    if not vis:
        return NO_POSITIVE, None
    m = max(scores[i] for i in vis)
    z = math.fsum(math.exp(scores[i] - m) for i in vis)
    contrib = [0.0] * n
    for i in vis:
        w = math.exp(scores[i] - m) / z
        contrib[i] = w * math.fsum(features[i][k] * query[k] for k in range(DIM))
    pos = math.fsum(c for c in contrib if c > 0.0)
    if pos <= 0.0:
        return NO_POSITIVE, None
    target = pos * ratio
    eps = 1e-12 * pos
    best = None
    for s in range(n):
        if s in hidden:
            continue
        for e in range(s + 1, n + 1):
            if (e - 1) in hidden:
                break
            if math.fsum(contrib[s:e]) + eps >= target:
                cand = (e - s, s, e)
                if best is None or cand < best:
                    best = cand
                break
    if best is None:
        return NO_QUALIFYING, None
    return OK, (best[1], best[2])


def _snapshot(features, scores, orders):
    tree = VisibleSoftmaxTree(features, scores)
    hidden = set()
    for s, e in orders:
        tree.add_cover(s, e, +1)
        hidden.update(range(s, e))
    snap = LocateSnapshot(len(scores), tree.scores, tree.features,
                          *tree.root_softmax(), tree.visible_count(), tuple(orders))
    return snap, hidden


@pytest.mark.parametrize("seed", range(30))
def test_locate_matches_exact_oracle(seed):
    rng = random.Random(1000 + seed)
    n = rng.choice([1, 2, 3, 5, 17, 64, 200])
    features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
    scores = [rng.uniform(-20, 20) for _ in range(n)]
    query = [rng.uniform(-1, 1) for _ in range(DIM)]
    ratio = rng.choice([1e-12, 1e-9, 0.1, 0.5, 0.9, 1.0, rng.random()])
    orders = []
    for _ in range(rng.randint(0, 4)):
        s = rng.randrange(0, n)
        e = rng.randrange(s + 1, n + 1)
        orders.append((s, e))
    snap, hidden = _snapshot(features, scores, orders)
    res = locate_segment(snap, query, ratio)
    exp_status, exp_seg = _locate_oracle(features, scores, hidden, query, ratio)
    assert res.status == exp_status
    if exp_status == OK:
        assert (res.start, res.end) == exp_seg
        # every shot of the segment is playable
        assert all(i not in hidden for i in range(res.start, res.end))
        assert res.achieved_ratio + 1e-11 >= ratio


def test_locate_extreme_cancellation():
    """Tiny positive term (1e-14) next to an O(1) negative must not be lost."""
    rng = random.Random(31337)
    n = 60
    scores = [0.0] * n  # equal weights
    features = [[0.0] * DIM for _ in range(n)]
    for i in range(n):
        mag = 10 ** rng.randint(-14, 0)
        features[i][0] = rng.choice([-1.0, 1.0]) * mag * rng.random()
    query = [1.0] * DIM
    for ratio in (1e-12, 1e-9, 0.5, 1.0):
        snap, hidden = _snapshot(features, scores, [(10, 20), (40, 45)])
        res = locate_segment(snap, query, ratio)
        exp_status, exp_seg = _locate_oracle(features, scores, hidden, query, ratio)
        assert res.status == exp_status
        if exp_status == OK:
            assert (res.start, res.end) == exp_seg


def test_locate_underflow_weight_does_not_split_runs():
    """A softmax weight that underflows to 0.0 still denotes a *playable*
    shot and must not split a contiguous segment (regression: wide score
    ranges make most exp() weights underflow)."""
    rng = random.Random(2026)
    n = 3000
    base = [rng.uniform(-1, 1) for _ in range(DIM)]
    # strong shared signal so most shots contribute positively
    features = [[0.95 * b + 0.05 * rng.uniform(-1, 1) for b in base]
                for _ in range(n)]
    # full legal score range -> the softmax denominator is dominated by a
    # few high scores and most individual weights underflow to 0.0
    scores = [rng.uniform(-1000, 1000) for _ in range(n)]
    snap, hidden = _snapshot(features, scores, [])
    assert not hidden
    res = locate_segment(snap, base, 0.5)
    # without any hidden order a single run spans the whole video, so a
    # qualifying segment must exist and be found
    assert res.status == OK
    assert 0 <= res.start < res.end <= n
    assert res.achieved_ratio + 1e-11 >= 0.5


def test_locate_negative_contribution_changes_segment():
    """A negative shot inside a candidate range forces a longer segment."""
    # equal scores -> equal weight 0.25; query aligned with dimension 0
    features = [[0.8, *([0.0] * (DIM - 1))],
                [-0.8, *([0.0] * (DIM - 1))],
                [0.8, *([0.0] * (DIM - 1))],
                [0.8, *([0.0] * (DIM - 1))]]
    scores = [0.0, 0.0, 0.0, 0.0]
    query = [1.0] * DIM
    snap, _ = _snapshot(features, scores, [])
    # contributions: [0.2, -0.2, 0.2, 0.2]; positives total = 0.6
    r = locate_segment(snap, query, 0.3)  # target 0.18 -> one positive shot
    assert r.status == OK and (r.start, r.end) == (0, 1)
    # target 0.3: length-1 windows give only 0.2; the negative shot rules
    # out [1,3), so the shortest length-2 window is the positive pair [2,4)
    r = locate_segment(snap, query, 0.5)
    assert r.status == OK and (r.start, r.end) == (2, 4)
    assert abs(r.achieved_ratio - (0.4 / 0.6)) <= 1e-12
    # ratio 1.0 needs 0.6 but the whole video nets only 0.4 (the -0.2 shot
    # can never be excluded from a segment spanning all three positives)
    r = locate_segment(snap, query, 1.0)
    assert r.status == NO_QUALIFYING


def test_locate_tie_prefers_earliest():
    """Two equal-length qualifying windows -> the earliest one is returned."""
    features = [[0.0] * DIM for _ in range(6)]
    for i in (1, 4):
        features[i][0] = 0.9
    scores = [0.0] * 6
    query = [1.0] * DIM
    snap, _ = _snapshot(features, scores, [])
    r = locate_segment(snap, query, 0.5)
    assert r.status == OK and (r.start, r.end) == (1, 2)


def test_locate_failure_statuses():
    # no positive contribution: every feature anti-aligned with the query
    features = [[-0.5] * DIM for _ in range(4)]
    snap, _ = _snapshot(features, [0.0] * 4, [])
    r = locate_segment(snap, [1.0] * DIM, 0.5)
    assert r.status == NO_POSITIVE and r.start is None and r.achieved_ratio is None

    # all shots hidden -> no playable shot -> no positive contribution
    snap, _ = _snapshot([[0.5] * DIM for _ in range(3)], [0.0] * 3, [(0, 3)])
    r = locate_segment(snap, [1.0] * DIM, 0.5)
    assert r.status == NO_POSITIVE and r.visible_count == 0

    # positives exist but a hidden wall isolates them below the ratio:
    # one strong positive shot alone is 100% of positives, so ratio 1 is
    # still reachable; instead split positives across the wall with the
    # larger side unable to reach the requested share
    features = [[0.0] * DIM for _ in range(4)]
    features[0][0] = 0.9
    features[3][0] = 0.1
    snap, _ = _snapshot(features, [0.0] * 4, [(1, 3)])  # playable: 0 and 3
    # positive total ~ 0.25*(0.9+0.1)=0.25; shot 0 alone gives 90% -> asking
    # for 95% cannot be met by either isolated playable run
    r = locate_segment(snap, [1.0] * DIM, 0.95)
    assert r.status == NO_QUALIFYING and r.start is None
    assert r.positive_total > 0.0


def test_locate_is_linear_n_plus_k():
    """Locate time grows linearly with N and with the number of orders K."""
    rng = random.Random(909)
    query = [rng.uniform(-1, 1) for _ in range(DIM)]

    def time_locate(n, orders):
        features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
        scores = [rng.uniform(-20, 20) for _ in range(n)]
        os_ = []
        for k in range(orders):
            s = (k * 7) % n
            e = min(n, s + 1 + (k * 13) % 50)
            os_.append((s, e))
        snap, _ = _snapshot(features, scores, os_)
        t0 = time.perf_counter()
        locate_segment(snap, query, 0.5)
        return time.perf_counter() - t0

    t_small = time_locate(10_000, 10)
    t_big = time_locate(100_000, 10)
    print(f"\nlocate N=10k: {t_small * 1e3:.2f} ms, N=100k: {t_big * 1e3:.2f} ms")
    assert t_big < 15 * t_small  # 10x work -> ~10x time, not quadratic
    t_k10 = time_locate(50_000, 10)
    t_k2000 = time_locate(50_000, 2000)
    print(f"locate K=10: {t_k10 * 1e3:.2f} ms, K=2000: {t_k2000 * 1e3:.2f} ms")
    assert t_k2000 < 3 * t_k10 + 0.05


# ------------------------------------------------------------- locate API

def test_api_locate_happy_path_and_semantics(client):
    shots = _shots(12, seed=11)
    client.post("/videos/loc/shots", json={"shots": shots})
    query = [0.1 * ((i % 5) - 2) for i in range(DIM)]
    r = client.post("/videos/loc/locate", json={"query": query, "ratio": 0.5})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert 0 <= body["start"] < body["end"] <= 12
    assert body["achieved_ratio"] >= 0.5 - 1e-11
    assert body["positive_total"] > 0.0
    assert body["visible_count"] == 12

    # ratio=1 returns a segment reaching (nearly) all positive contribution
    r1 = client.post("/videos/loc/locate", json={"query": query, "ratio": 1.0})
    assert r1.status_code == 200 and r1.json()["status"] == "ok"
    assert r1.json()["achieved_ratio"] + 1e-11 >= 1.0

    # shortest-first: a smaller ratio never returns a longer segment
    r_lo = client.post("/videos/loc/locate", json={"query": query, "ratio": 0.01})
    assert (r_lo.json()["end"] - r_lo.json()["start"]) <= \
           (body["end"] - body["start"])


def test_api_locate_respects_hidden_runs(client):
    shots = _shots(10, seed=12)
    client.post("/videos/loc2/shots", json={"shots": shots})
    client.post("/videos/loc2/restrictions",
                json={"order_id": "wall", "start": 3, "end": 7})
    query = [0.2] * DIM
    r = client.post("/videos/loc2/locate", json={"query": query, "ratio": 0.5})
    assert r.status_code == 200
    body = r.json()
    assert body["visible_count"] == 6
    if body["status"] == "ok":
        # the segment never crosses the hidden wall [3,7)
        assert body["end"] <= 3 or body["start"] >= 7


def test_api_locate_no_positive(client):
    # features all anti-aligned with the query
    shots = [{"feature": [-0.5] * DIM, "score": 0.0} for _ in range(4)]
    client.post("/videos/loc3/shots", json={"shots": shots})
    r = client.post("/videos/loc3/locate",
                    json={"query": [1.0] * DIM, "ratio": 0.5})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no_positive_contribution"
    assert body["start"] is None and body["achieved_ratio"] is None


def test_api_locate_no_qualifying(client):
    # two isolated playable shots, weaker one cannot reach the high ratio
    shots = [{"feature": [0.0] * DIM, "score": 0.0} for _ in range(4)]
    shots[0]["feature"][0] = 0.9
    shots[3]["feature"][0] = 0.1
    client.post("/videos/loc4/shots", json={"shots": shots})
    client.post("/videos/loc4/restrictions",
                json={"order_id": "wall", "start": 1, "end": 3})
    r = client.post("/videos/loc4/locate",
                    json={"query": [1.0] * DIM, "ratio": 0.95})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no_qualifying_interval"
    assert body["start"] is None and body["positive_total"] > 0.0


def test_api_locate_validation_and_404(client):
    client.post("/videos/loc5/shots", json={"shots": _shots(4, seed=13)})
    # wrong query dimension
    assert client.post("/videos/loc5/locate",
                       json={"query": [0.0] * 15, "ratio": 0.5}).status_code == 422
    # ratio out of range
    assert client.post("/videos/loc5/locate",
                       json={"query": [0.0] * DIM, "ratio": 0.0}).status_code == 422
    assert client.post("/videos/loc5/locate",
                       json={"query": [0.0] * DIM, "ratio": 1.01}).status_code == 422
    # non-finite query elements are rejected at the schema layer (they
    # cannot even be encoded as strict JSON over HTTP)
    from app.main import LocateRequest
    with pytest.raises(Exception):
        LocateRequest(query=[float("inf")] + [0.0] * (DIM - 1), ratio=0.5)
    with pytest.raises(Exception):
        LocateRequest(query=[float("nan")] + [0.0] * (DIM - 1), ratio=0.5)
    # unknown video
    assert client.post("/videos/nope/locate",
                       json={"query": [0.0] * DIM, "ratio": 0.5}).status_code == 404


def test_api_locate_uses_entry_snapshot(client):
    """Orders changed after the snapshot do not alter the locate result:
    the response visible_count matches the state captured at entry."""
    shots = _shots(20, seed=14)
    client.post("/videos/loc6/shots", json={"shots": shots})
    query = [0.3] * DIM
    before = client.post("/videos/loc6/locate",
                         json={"query": query, "ratio": 0.5}).json()
    client.post("/videos/loc6/restrictions",
                json={"order_id": "r1", "start": 0, "end": 5})
    after = client.post("/videos/loc6/locate",
                        json={"query": query, "ratio": 0.5}).json()
    assert before["visible_count"] == 20
    assert after["visible_count"] == 15
    # revoking restores the original locate answer
    client.delete("/videos/loc6/restrictions/r1")
    again = client.post("/videos/loc6/locate",
                        json={"query": query, "ratio": 0.5}).json()
    assert again["visible_count"] == 20
    assert (again["start"], again["end"]) == (before["start"], before["end"])
