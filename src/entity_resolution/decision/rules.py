import lightgbm as lgb
import numpy as np

from entity_resolution.features.base import BASE
from entity_resolution.core import decide, macro_f05

CHEAP = ["score", "rank", "gap_top1", "margin12", "cos_name", "cos_addr"]
CHEAP_COLS = [BASE.index(c) for c in CHEAP]
CHEAP_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=15, min_data_in_leaf=100, seed=3407, verbose=-1)
FLOORS = np.round(np.arange(0.30, 0.901, 0.02), 2)
FACTORS = np.round(np.arange(0.50, 1.001, 0.05), 2)
BARS = np.round(np.arange(0.30, 0.951, 0.025), 3)


def fit_cheap(X, y, threads):
    """The first model: only the shortlist columns of the base matrix; predict it on X[:, CHEAP_COLS]."""
    return lgb.train({**CHEAP_PARAMS, "num_threads": threads},
                     lgb.Dataset(X[:, CHEAP_COLS], np.asarray(y, dtype=np.float32)), num_boost_round=100)


def trim_mask(prob, cut=0.01):
    return prob >= cut


def best_top(rec, cand, prob, n_rec, n_s1):
    """Per pair: is it its record's single best candidate (ties to the first), and the best score among the records
    whose best candidate is that pair's business (-inf when none)."""
    best = decide(rec, prob, n_rec, -np.inf)
    top = np.full(n_s1, -np.inf)
    np.maximum.at(top, cand[best], prob[best])
    return best, top[cand]


def drawbridge(rec, cand, prob, n_rec, n_s1, floor, factor):
    """Each record goes to its best candidate only; a business answers when its best such record reaches the floor, and
    then keeps every such record scoring at least factor times that best."""
    best, top = best_top(rec, cand, prob, n_rec, n_s1)
    return best & (top >= floor) & (prob >= factor * top)


def accepted_per_s1(accept, n_s1):
    return accept.sum() / n_s1


def floor_curve(rec, cand, prob, true_owner, k, s1_mask, n_rec, factor):
    """Macro F0.5 over s1_mask at every floor in FLOORS, for one factor."""
    best, top = best_top(rec, cand, prob, n_rec, len(k))
    c, p, t, own = cand[best], prob[best], top[best], true_owner[best]
    near = p >= factor * t
    return np.array([macro_f05(c, near & (t >= f), own, k, s1_mask)[0] for f in FLOORS])


def tune_drawbridge(rec, cand, prob, true_owner, k, s1_mask, n_rec):
    """The best (floor, factor) on the FLOORS x FACTORS grid and its score; ties go to the lower floor, then the lower
    factor, as in eda/f4_rules.py."""
    curves = np.array([floor_curve(rec, cand, prob, true_owner, k, s1_mask, n_rec, x) for x in FACTORS]).T
    i, j = np.unravel_index(np.argmax(curves), curves.shape)
    return float(FLOORS[i]), float(FACTORS[j]), float(curves[i, j])


def safe_range(curve, grid=FLOORS, tol=0.001):
    """The run of grid values around the best point of a curve whose scores stay within tol of the best."""
    assert len(curve) == len(grid), (len(curve), len(grid))
    ok = curve >= curve.max() - tol
    lo = hi = int(np.argmax(curve))
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    return float(grid[lo]), float(grid[hi])


def count_floor(rec, cand, prob, n_rec, n_s1, factor, target, lo, hi):
    """The floor at which Drawbridge with this factor accepts closest to `target` records per Source 1 row, and that
    floor clipped to [lo, hi].

    The accepted count only changes where the floor passes a business's best score, so the search runs exactly over
    those scores (sorted, cumulated); a business enters with all its kept records, which bounds how close it gets.
    """
    best, top = best_top(rec, cand, prob, n_rec, n_s1)
    t = top[best & (prob >= factor * top)]
    tops = np.unique(t)[::-1]
    accepted = np.cumsum(np.bincount(np.searchsorted(-tops, -t), minlength=len(tops)))
    raw = float(tops[np.argmin(np.abs(accepted - target * n_s1))])
    return raw, min(max(raw, lo), hi)


def bar_curve(rec, cand, prob, true_owner, k, s1_mask, n_rec, floor, factor):
    """Macro F0.5 over s1_mask at every per-record bar in BARS, on top of Drawbridge (floor, factor)."""
    acc = drawbridge(rec, cand, prob, n_rec, len(k), floor, factor)
    return np.array([macro_f05(cand, acc & (prob >= b), true_owner, k, s1_mask)[0] for b in BARS])


def count_bar(rec, cand, prob, n_rec, n_s1, floor, factor, target):
    """F2's count rule: the per-record bar at which Drawbridge (floor, factor) plus that bar accepts `target` records
    per Source 1 row."""
    p = np.sort(prob[drawbridge(rec, cand, prob, n_rec, n_s1, floor, factor)])[::-1]
    n = min(int(round(target * n_s1)), len(p))
    return float(p[n - 1]) if n > 0 else 1.0
