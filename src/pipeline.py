import lightgbm as lgb
import numpy as np
import polars as pl

from translit import learn

K = 3
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9,
              verbose=-1)
ROUNDS = 300
NO_LATIN = ~pl.col("business_name").str.contains(r"[A-Za-z]")


def truth_pairs(gt):
    """Ground truth as one (source1_entity_id, other_id) row per true match."""
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
            .rename({"matched_entity_ids": "other_id"}))


def script_table(s1, others, pairs):
    """The Indian-script word table, learned from script-name copies and their owner's Latin name."""
    script = (others.filter(NO_LATIN).join(pairs, left_on="entity_id", right_on="other_id")
              .join(s1.select("entity_id", latin_name="business_name"), left_on="source1_entity_id",
                    right_on="entity_id")
              .select(script_name="business_name", latin_name="latin_name"))
    return learn(script)


def owner_rows(s1, ot, pairs):
    """Each record's owner as a row of s1, or -1 when it has none in s1."""
    pos = s1.select("entity_id").with_row_index("row")
    return (ot.select("entity_id").join(pairs, left_on="entity_id", right_on="other_id", how="left", maintain_order="left")
            .join(pos, left_on="source1_entity_id", right_on="entity_id", how="left", maintain_order="left")["row"]
            .fill_null(-1).to_numpy().astype(np.int64))


def fit(X, y, threads, rounds=ROUNDS):
    return lgb.train({**PARAMS, "num_threads": threads}, lgb.Dataset(X, np.asarray(y, dtype=np.float32)),
                     num_boost_round=rounds)


def decide(rec, prob, n_rec, t):
    """Each record keeps its single most likely candidate (ties to the first), and only at or above t."""
    order = np.lexsort((-prob, rec))
    first = np.ones(len(rec), dtype=bool)
    first[1:] = rec[order][1:] != rec[order][:-1]
    keep = np.zeros(len(rec), dtype=bool)
    keep[order[first]] = True
    return keep & (prob >= t)


def macro_f05(cand, accept, true_owner, k, s1_mask):
    """The organisers' metric over the Source 1 rows in s1_mask; k is each row's number of true matches."""
    n = len(k)
    tp = np.bincount(cand[accept & true_owner], minlength=n)
    fp = np.bincount(cand[accept & ~true_owner], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where((k == 0) & (fp == 0), 1.0, 1.25 * tp / (1.25 * tp + 0.25 * (k - tp) + fp))
    return f[s1_mask].mean(), f[s1_mask & (k == 0)].mean(), f[s1_mask & (k > 0)].mean()
