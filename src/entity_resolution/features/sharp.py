import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from os.path import commonprefix

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from entity_resolution.features.families import LEGAL_FORM, lead, legal, parse, spans, summary
from entity_resolution.features.base import pairs

SHARP = ["hn_gap_20", "hn_digit_indel", "hn_digit_sub", "hn_biz_in_rec", "hn_rec_in_biz",
         "dw_ratio", "dw_prefix", "dw_plus_letters", "dw_stem_ending", "dw_edit", "dw_rarity_biz", "dw_rarity_rec",
         "code_changed", "legal_same", "legal_one_side",
         "addr_z_ratio", "addr_z_tset", "name_nospace", "web_vs_name", "shop_words"]
WEB_LABEL = r"([a-z0-9][a-z0-9-]{2,})\.(?:com|in|net|org|co|fr)\b"
SHOP_SHARE = 0.00015


def init_worker(doc_freq, n_s1):
    global DF, N_S1
    DF, N_S1 = doc_freq, n_s1


def idf(w):
    return math.log(N_S1 / (1 + DF[w]))


def digests(addrs):
    """summary() of each address and the set of every number value in it."""
    return [(summary(x), {v for _, v, _ in parse(x)}) for x in addrs]


def pair_rows(A, B, rn, cn):
    """The 15 per-pair Python columns: 5 house-number checks, 7 on the best-matching pair of differing name words,
    code_changed, legal_same, legal_one_side. A, B hold each side's digests(), rn, cn its name."""
    out = np.full((len(rn), 15), -1.0, dtype=np.float32)
    for i, (((pa, _), all_a), ((pb, _), all_b), a, b) in enumerate(zip(A, B, rn, cn)):
        if pa and pb:
            va, vb = pa[0], pb[0]
            moved = va != vb
            edit = Levenshtein.distance(va, vb)
            out[i, 0] = moved and abs(lead(va) - lead(vb)) <= 20
            out[i, 1] = moved and abs(len(va) - len(vb)) == 1 and edit == 1
            out[i, 2] = moved and len(va) == len(vb) and edit == 1
            out[i, 3] = moved and vb in all_a
            out[i, 4] = moved and va in all_b
        wa, wb = a.split(), b.split()
        ca = Counter(w for w in wa if w not in LEGAL_FORM)
        cb = Counter(w for w in wb if w not in LEGAL_FORM)
        la, lb = list((ca - cb).elements()), list((cb - ca).elements())
        if la and lb:
            s, x, z = max((fuzz.ratio(x, z), x, z) for x in la for z in lb)
            p, d = len(commonprefix([x, z])), Levenshtein.distance(x, z)
            out[i, 5:12] = (s, p / max(len(x), len(z)), 1 <= d <= 3 and d == abs(len(x) - len(z)),
                            p >= 3 and 1 <= len(x) - p <= 3 and 1 <= len(z) - p <= 3, d, idf(z), idf(x))
            out[i, 12] = any(x.isalpha() and z.isalpha() and len(x) <= 4 and len(z) <= 4 for x in la for z in lb)
        fa, fb = legal(a), legal(b)
        if fa and fb:
            out[i, 13] = fa == fb
        out[i, 14] = bool(fa) != bool(fb)
    return out


def shop_words(s1, ot):
    """Per record, its name words that are in no Source 1 name but in at least SHOP_SHARE of the Source 2/3 names
    (at least 20; 20 of about 130,000 on the state slices where it was measured)."""
    words = lambda frame: frame.select(r=pl.int_range(pl.len()), w=pl.col("n").str.split(" ").list.unique()).explode("w")
    ow = words(ot).filter(pl.col("w") != "")
    shop = ow.group_by("w").len().filter(pl.col("len") >= max(20, SHOP_SHARE * ot.height)).join(words(s1).select("w").unique(), on="w",
                                                                          how="anti")
    hits = ow.join(shop.select("w"), on="w", how="semi").group_by("r").len()
    out = np.zeros(ot.height, dtype=np.float32)
    out[hits["r"].to_numpy()] = hits["len"].to_numpy()
    return out


def sharp(s1, ot, idx, val, rec, cand, workers):
    """The 20 label-free columns (SHARP) aimed at F10's confident errors, rows in the order of features.pairs(idx, val).

    `s1`, `ot` come from features.prep(). The Python loops run in `workers` processes; rapidfuzz uses as many threads.
    """
    assert isinstance(workers, int) and workers > 0, workers
    rec2, cand2, _, _ = pairs(idx, val)
    assert np.array_equal(rec, rec2) and np.array_equal(cand, cand2), "rec, cand are not pairs(idx, val)"
    rn, cn = ot["n"].to_list(), s1["n"].to_list()
    doc_freq = Counter(w for n in cn for w in set(n.split()))
    rec_l, cand_l = rec.tolist(), cand.tolist()
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(doc_freq, s1.height)) as ex:
        parsed = lambda addrs: [x for part in ex.map(digests, [addrs[s] for s in spans(len(addrs))]) for x in part]
        A, B = parsed(ot["business_address"].to_list()), parsed(s1["business_address"].to_list())
        pick = lambda table, rows: [[table[j] for j in rows[s]] for s in spans(len(rows))]
        W = np.concatenate(list(ex.map(pair_rows, pick(A, rec_l), pick(B, cand_l), pick(rn, rec_l),
                                       pick(cn, cand_l))))
    assert len(A) == ot.height and len(B) == s1.height and W.shape == (len(rec), 15), (len(A), len(B), W.shape)

    unpadded = lambda frame: frame["a"].str.replace_all(r"\b0+(\d)", "${1}").to_numpy()
    za, zb = unpadded(ot)[rec], unpadded(s1)[cand]
    glued = lambda frame: frame["n"].str.replace_all(" ", "", literal=True).to_numpy()
    ga, gb = glued(ot)[rec], glued(s1)[cand]
    label = lambda frame: (frame["business_name"].fill_null("").str.to_lowercase().str.extract(WEB_LABEL, 1)
                           .fill_null("").to_numpy())
    wa, wb = label(ot)[rec], label(s1)[cand]
    web = lambda w, other: np.where(w != "", process.cpdist(w, other, scorer=fuzz.partial_ratio, workers=workers), -1)
    X = np.column_stack([
        W, process.cpdist(za, zb, scorer=fuzz.ratio, workers=workers),
        process.cpdist(za, zb, scorer=fuzz.token_set_ratio, workers=workers),
        process.cpdist(ga, gb, scorer=fuzz.ratio, workers=workers), np.maximum(web(wa, gb), web(wb, ga)),
        shop_words(s1, ot)[rec]]).astype(np.float32)
    assert X.shape == (len(rec), len(SHARP)), X.shape
    return SHARP, X
