import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from text import clean, numbers
from translit import romanise

LEGAL = ("llc inc incorporated corp corporation co company ltd limited pvt private llp pc plc lp "
         "sa sas sarl sci eurl the and of shri smt sri mr ms dba").split()
BASE = ["score", "rank", "gap_top1", "margin12", "cos_name", "cos_addr", "name_ratio", "name_tset", "name_partial",
        "name_jw", "addr_ratio", "addr_tset", "core_equal", "in_degree", "best_for_cand", "is_s3", "addr_missing",
        "name_no_latin", "name_len", "num_shared", "num_jaccard", "num_first_equal", "num_rec_none", "num_rec_subset"]


def core(col):
    return (clean(col).str.split(" ").list.eval(pl.element().filter(
        (pl.element() != "") & ~pl.element().is_in(LEGAL))).list.sort().list.join(" "))


def prep(df, table=None):
    """Cleaned name, address, core name and numbers; Indian-script names go through the learned word table first."""
    latin = romanise(df["business_name"], table) if table is not None else df["business_name"]
    return df.with_columns(latin=latin).with_columns(
        n=clean("latin"), a=clean("business_address"), core=core("latin"), nums=numbers("business_address"))


def pairs(idx, val):
    """Flatten a shortlist into (record row, Source 1 row, rank, score), dropping empty slots."""
    k = idx.shape[1]
    rec = np.repeat(np.arange(idx.shape[0]), k)
    keep = idx.ravel() >= 0
    return rec[keep], idx.ravel()[keep], np.tile(np.arange(k), idx.shape[0])[keep], val.ravel()[keep]


def row_dot(q, d, rec, cand, chunk=1_000_000):
    out = np.empty(len(rec), dtype=np.float32)
    for i in range(0, len(rec), chunk):
        out[i:i + chunk] = np.asarray(q[rec[i:i + chunk]].multiply(d[cand[i:i + chunk]]).sum(axis=1)).ravel()
    return out


def number_features(rec_nums, s1_nums):
    r, c = pl.col("r"), pl.col("c")
    shared, union = r.list.set_intersection(c).list.len(), r.list.set_union(c).list.len()
    both = (r.list.len() > 0) & (c.list.len() > 0)
    return pl.DataFrame({"r": rec_nums, "c": s1_nums}).select(
        num_shared=shared,
        num_jaccard=pl.when(union > 0).then(shared / union).otherwise(-1),
        num_first_equal=pl.when(both).then((r.list.first() == c.list.first()).cast(pl.Float64)).otherwise(-1),
        num_rec_none=(r.list.len() == 0).cast(pl.Float64),
        num_rec_subset=pl.when(r.list.len() > 0).then((shared == r.list.unique().list.len()).cast(pl.Float64))
        .otherwise(-1),
    ).to_numpy().astype(np.float32)


def base_features(s1, ot, idx, val, parts, workers=-1):
    """The 24 slice-model features for every shortlisted pair.

    `s1`, `ot` come from prep(); `idx`, `val` from shortlist.top_k; `parts` holds the (records, Source 1) TF-IDF
    matrices of the name and of the address, each unit length per row.
    """
    rec, cand, rank, score = pairs(idx, val)
    top1 = val[rec, 0]
    second = val[rec, 1] if idx.shape[1] > 1 else np.zeros_like(top1)
    rn, cn = ot["n"].to_numpy()[rec], s1["n"].to_numpy()[cand]
    ra, ca = ot["a"].to_numpy()[rec], s1["a"].to_numpy()[cand]
    first = idx[:, 0][idx[:, 0] >= 0]
    best_for_cand = np.zeros(s1.height, dtype=np.float32)
    np.maximum.at(best_for_cand, cand, score)
    raw_name = ot["business_name"].fill_null("")
    cols = {
        "score": score, "rank": rank, "gap_top1": top1 - score, "margin12": top1 - second,
        "cos_name": row_dot(parts[0][0], parts[0][1], rec, cand),
        "cos_addr": row_dot(parts[1][0], parts[1][1], rec, cand),
        "name_ratio": process.cpdist(rn, cn, scorer=fuzz.ratio, workers=workers),
        "name_tset": process.cpdist(rn, cn, scorer=fuzz.token_set_ratio, workers=workers),
        "name_partial": process.cpdist(rn, cn, scorer=fuzz.partial_ratio, workers=workers),
        "name_jw": process.cpdist(rn, cn, scorer=JaroWinkler.normalized_similarity, workers=workers),
        "addr_ratio": process.cpdist(ra, ca, scorer=fuzz.ratio, workers=workers),
        "addr_tset": process.cpdist(ra, ca, scorer=fuzz.token_set_ratio, workers=workers),
        "core_equal": ot["core"].to_numpy()[rec] == s1["core"].to_numpy()[cand],
        "in_degree": np.bincount(first, minlength=s1.height)[cand],
        "best_for_cand": score >= best_for_cand[cand],
        "is_s3": ot["entity_id"].str.starts_with("S3").to_numpy()[rec],
        "addr_missing": ot["business_address"].is_null().to_numpy()[rec],
        "name_no_latin": (~raw_name.str.contains(r"[A-Za-z]")).to_numpy()[rec],
        "name_len": raw_name.str.len_chars().to_numpy()[rec],
    }
    num = number_features(ot["nums"].gather(rec), s1["nums"].gather(cand))
    X = np.column_stack([np.asarray(v, dtype=np.float32) for v in cols.values()] + [num])
    return rec, cand, X
