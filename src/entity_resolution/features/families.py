import math
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from entity_resolution.features.base import pairs

KIND = {**dict.fromkeys("apartment apt unit suite ste flat room rm floor fl lot trailer office bldg building tower "
                        "space spc".split(), "unit"),
        **dict.fromkeys("sector sec phase block blk gali ward part pocket".split(), "area"),
        **dict.fromkeys("box pmb".split(), "box"),
        **dict.fromkeys("highway hwy road rd cr us interstate nh sh route rte mile street st county".split(), "road")}
NO = {"no", "number", "num"}
TOK = re.compile(r"#+|\d+(?:\s*/\s*\d+)*(?:[a-z]{1,2}(?![a-z]))?|[a-z]+")
HONORIFIC = {"shri", "sri", "shree", "smt", "mr", "mrs", "ms", "dr", "kumari"}
LEGAL_FORM = {"pvt": "private", "private": "private", "public": "public", "ltd": "limited", "limited": "limited",
              "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co",
              "company": "co", "llp": "llp", "lp": "lp", "plc": "plc", "pc": "pc", "sa": "sa", "sas": "sas",
              "sarl": "sarl", "eurl": "eurl", "sci": "sci"}
DBA = r"(?i)\bd\s*\.?\s*b\s*\.?\s*a\b|\ba\s*\.?\s*k\s*\.?\s*a\b|trading as|\bt/a\b"
WEB = r"(?i)www\.|\.(com|in|net|org|co|fr)\b"
SCRIPT = r"[\x{0900}-\x{0DFF}]"
EXTRA = ["hn_equal", "hn_rec_missing", "hn_cand_missing", "hn_suffix_only", "hn_one_digit", "hn_transposed", "hn_gap",
         "hn_log_gap", "sec_agree", "sec_one_side", "num_conflicts", "street_sim",
         "grp_share_rec_num", "grp_share_biz_num", "grp_biz_is_majority", "grp_distinct_nums", "grp_rank",
         "freq_name_biz", "freq_name_rec", "freq_core_biz", "freq_rarest_shared",
         "left_rec_n", "left_biz_n", "fuzzy_pairs", "abbrev_pairs", "left_rec_rarest", "left_biz_rarest",
         "dba", "web_name", "honorific_one_side", "legal_conflict", "private_public", "script_rec", "script_biz"]
CHUNK = 50_000


def parse(addr):
    """Numbers of one address as (kind, value, suffix); the kind comes from the word before the number."""
    toks = TOK.findall((addr or "").lower())
    out = []
    for i, t in enumerate(toks):
        if not t[0].isdigit() or t[-2:].isalpha():  # words, and ordinals such as 40th (street names)
            continue
        p1, p2 = (toks[i - 1] if i >= 1 else ""), (toks[i - 2] if i >= 2 else "")
        if p1.startswith("#"):
            kind = KIND.get(p2) if KIND.get(p2) in ("unit", "area", "box") else "hash"
        elif p1 in NO:
            kind = KIND.get(p2, "premise")
        else:
            kind = KIND.get(p1, "premise")
        digits, suffix = re.fullmatch(r"([\d/\s]+?)([a-z]?)", t).groups()
        out.append((kind, "/".join(p.strip().lstrip("0") or "0" for p in digits.split("/")), suffix))
    return out


def summary(addr):
    """Primary number (value, suffix) or None, and the other numbers as {kind: set of values}."""
    nums = parse(addr)
    prem = [(v, s) for k, v, s in nums if k == "premise"]
    hashes = [(v, s) for k, v, s in nums if k == "hash"]
    primary = (prem or hashes or [None])[0]
    kinds = {}
    for k, v, s in nums:
        if k == "hash" and prem:
            k = "unit"
        if k in ("unit", "area", "box", "road"):
            kinds.setdefault(k, set()).add(v)
    return primary, kinds


def lead(v):
    return int(v.split("/")[0][:9])


def is_abbrev(s, w):
    it = iter(w)
    return 2 <= len(s) < len(w) and s[0] == w[0] and all(ch in it for ch in s)


def pair_off(la, lb, pairs):
    """Greedy one-to-one matching in the order of `pairs` (i, j); returns both leftovers and the count."""
    used_a, used_b = set(), set()
    for i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    return [x for i, x in enumerate(la) if i not in used_a], [x for j, x in enumerate(lb) if j not in used_b], len(used_a)


def align(a, b):
    """One-to-one word pairing: exact, then fuzzy, then abbreviation or initials. Returns the leftovers
    of each side, the fuzzy pair count and the abbreviation pair count."""
    ca, cb = Counter(a), Counter(b)
    la, lb = list((ca - cb).elements()), list((cb - ca).elements())
    scored = sorted(((fuzz.ratio(x, z), i, j) for i, x in enumerate(la) for j, z in enumerate(lb)
                     if min(len(x), len(z)) >= 4), reverse=True)
    la, lb, n_fuzzy = pair_off(la, lb, [(i, j) for s, i, j in scored if s >= 80])
    la, lb, n_abbr = pair_off(la, lb, [(i, j) for i, x in enumerate(la) for j, z in enumerate(lb)
                                       if is_abbrev(x, z) or is_abbrev(z, x)])
    for short, long_ in ((la, lb), (lb, la)):
        for w in list(short):
            for s in range(len(long_) - len(w) + 1):
                if len(w) >= 2 and all(long_[s + k][0] == w[k] for k in range(len(w))):
                    short.remove(w)
                    del long_[s:s + len(w)]
                    n_abbr += 1
                    break
    return la, lb, n_fuzzy, n_abbr


def legal(n):
    return {LEGAL_FORM[w] for w in n.replace("l l c", "llc").replace("l l p", "llp").split() if w in LEGAL_FORM}


def init_worker(doc_freq, n_s1):
    global DF, N_S1
    DF, N_S1 = doc_freq, n_s1


def summaries(addrs):
    return [summary(x) for x in addrs]


def pair_rows(A, B, rn, cn):
    """The per-pair Python columns: the 11 house-number checks, freq_rarest_shared, the 6 word-alignment columns,
    then honorific_one_side, legal_conflict, private_public. A, B hold each side's summary(), rn, cn its name."""
    out = np.full((len(rn), 21), -1.0, dtype=np.float32)
    for i, ((pa, ka), (pb, kb), a, b) in enumerate(zip(A, B, rn, cn)):
        out[i, 1], out[i, 2] = pa is None and pb is not None, pb is None and pa is not None
        if pa and pb:
            (va, sa), (vb, sb) = pa, pb
            out[i, 0] = va == vb
            out[i, 3] = va == vb and sa != sb
            out[i, 4] = va != vb and Levenshtein.distance(va, vb) == 1
            out[i, 5] = va != vb and len(va) > 1 and sorted(va) == sorted(vb)
            out[i, 6] = abs(lead(va) - lead(vb))
            out[i, 7] = math.log1p(out[i, 6])
        shared = [k for k in ("unit", "area") if k in ka and k in kb]
        if shared:
            out[i, 8] = all(ka[k] & kb[k] for k in shared)
        out[i, 9] = sum((k in ka) != (k in kb) for k in ("unit", "area"))
        out[i, 10] = (bool(pa and pb) and pa[0] != pb[0]) + sum(
            not (ka[k] & kb[k]) for k in ("unit", "area", "box", "road") if k in ka and k in kb)
        wa, wb = a.split(), b.split()
        common = set(wa) & set(wb)
        if common:
            out[i, 11] = math.log1p(min(DF[w] for w in common))
        la, lb, nf, na = align(wa, wb)
        out[i, 12:16] = len(la), len(lb), nf, na
        if la:
            out[i, 16] = max(math.log(N_S1 / (1 + DF[w])) for w in la)
        if lb:
            out[i, 17] = max(math.log(N_S1 / (1 + DF[w])) for w in lb)
        out[i, 18] = bool(HONORIFIC & set(wa)) != bool(HONORIFIC & set(wb))
        fa, fb = legal(a), legal(b)
        if fa and fb:
            out[i, 19] = not (fa & fb)
        out[i, 20] = ("private" in fa and "public" in fb and "private" not in fb) or \
                     ("private" in fb and "public" in fa and "private" not in fa)
    return out


def group_family(idx, val, rec, cand, score, pv_ot, pv_s1):
    """Each candidate's crowd: the records whose first candidate it is (records without one stay out).
    pv_* are primary house-number codes, -1 for none."""
    top, n_s1 = idx[:, 0], len(pv_s1)
    ok = top >= 0
    m = int(max(pv_ot.max(), pv_s1.max())) + 2
    keys, counts = np.unique(top[ok].astype(np.int64) * m + pv_ot[ok] + 1, return_counts=True)

    def count(c, v):
        q = c.astype(np.int64) * m + v + 1
        pos = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
        return np.where(keys[pos] == q, counts[pos], 0)

    numbered = keys % m > 0
    distinct = np.bincount(keys[numbered] // m, minlength=n_s1)
    most = np.zeros(n_s1, dtype=np.int64)
    np.maximum.at(most, keys[numbered] // m, counts[numbered])
    own = top[rec] == cand
    rest = np.bincount(top[ok], minlength=n_s1)[cand] - own
    rv, cv = pv_ot[rec], pv_s1[cand]
    g_rec, g_biz = count(cand, rv), count(cand, cv)
    share = lambda hits, known: np.where((rest > 0) & known, hits / np.maximum(rest, 1), -1)
    tops = np.sort(top[ok] * 2.0 + val[ok, 0])
    above = np.searchsorted(tops, cand * 2.0 + 1.5) - np.searchsorted(tops, cand * 2.0 + score, side="right")
    return np.column_stack([share(g_rec - own, rv >= 0), share(g_biz - (own & (rv == cv)), cv >= 0),
                            np.where((distinct[cand] > 0) & (cv >= 0), g_biz == most[cand], -1), distinct[cand], above])


def spans(n):
    return [slice(i, i + CHUNK) for i in range(0, n, CHUNK)]


def log1p_table(x):
    """math.log1p of non-negative integer counts, through a table so every value equals the scalar call."""
    return np.array([math.log1p(i) for i in range(int(x.max()) + 1)])[x]


def families(s1, ot, idx, val, rec, cand, workers):
    """The 34 label-free comparison columns (EXTRA) for the pairs of features.pairs(idx, val), rows in its order.

    `s1`, `ot` come from features.prep(). The Python loops run in `workers` processes; rapidfuzz uses as many threads.
    """
    rec2, cand2, _, score = pairs(idx, val)
    assert np.array_equal(rec, rec2) and np.array_equal(cand, cand2), "rec, cand are not pairs(idx, val)"
    rn, cn = ot["n"].to_list(), s1["n"].to_list()
    doc_freq = Counter(w for n in cn for w in set(n.split()))
    rec_l, cand_l = rec.tolist(), cand.tolist()
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(doc_freq, s1.height)) as ex:
        parsed = lambda addrs: [x for part in ex.map(summaries, [addrs[s] for s in spans(len(addrs))]) for x in part]
        A, B = parsed(ot["business_address"].to_list()), parsed(s1["business_address"].to_list())
        pick = lambda table, rows: [[table[j] for j in rows[s]] for s in spans(len(rows))]
        W = np.concatenate(list(ex.map(pair_rows, pick(A, rec_l), pick(B, cand_l), pick(rn, rec_l),
                                       pick(cn, cand_l))))
    assert len(A) == ot.height and len(B) == s1.height and W.shape == (len(rec), 21), (len(A), len(B), W.shape)

    vocab = {}
    code = lambda S: np.array([-1 if p is None else vocab.setdefault(p[0], len(vocab)) for p, _ in S], dtype=np.int64)
    pv_ot, pv_s1 = code(A), code(B)
    street = lambda frame: frame["a"].str.replace_all(r"\d+", " ").str.replace_all(r"\s+", " ").to_numpy()
    sim = process.cpdist(street(ot)[rec], street(s1)[cand], scorer=fuzz.token_set_ratio, workers=workers)
    in_s1 = lambda frame, col: (frame.select(col).join(s1.group_by(col).len(), on=col, how="left",
                                                       maintain_order="left")["len"].fill_null(0).to_numpy())
    core_biz = np.where(s1["core"].to_numpy() != "", log1p_table(in_s1(s1, "core")), -1)
    flag = lambda frame, rx: frame["business_name"].fill_null("").str.contains(rx).to_numpy()
    X = np.column_stack([
        W[:, :11], sim, group_family(idx, val, rec, cand, score, pv_ot, pv_s1),
        log1p_table(in_s1(s1, "n"))[cand], log1p_table(in_s1(ot, "n"))[rec], core_biz[cand], W[:, 11:18],
        flag(ot, DBA)[rec] | flag(s1, DBA)[cand], flag(ot, WEB)[rec] | flag(s1, WEB)[cand], W[:, 18:21],
        flag(ot, SCRIPT)[rec], flag(s1, SCRIPT)[cand]]).astype(np.float32)
    assert X.shape == (len(rec), len(EXTRA)), X.shape
    return EXTRA, X
