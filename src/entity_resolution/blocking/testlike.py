import numpy as np
import polars as pl


def twins(s1, cap=200):
    """Near-twin Source 1 rows as sorted (row, row_b) pairs, row < row_b: the same non-empty core name, the same
    non-empty street and different first house numbers.

    The street is the cleaned address without its first number, as a sorted word list, compared exactly: a fuzzy
    street score calls "47th Street" and "8th Street" the same. Only core groups of 2 to `cap` rows are joined;
    bigger groups are chain names and are skipped.
    """
    street = (pl.col("a").str.replace(r"\d+", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
              .list.sort().list.join(" "))
    df = (s1.select("core", street=street, num=pl.col("nums").list.first())
          .with_row_index("row").filter(pl.col("core") != "").with_columns(size=pl.len().over("core")))
    big = df.filter(pl.col("size") > cap)
    df = df.filter(pl.col("size").is_between(2, cap) & pl.col("num").is_not_null() & (pl.col("street") != ""))
    out = (df.join(df, on=["core", "street"], suffix="_b")
           .filter((pl.col("row") < pl.col("row_b")) & (pl.col("num") != pl.col("num_b")))
           .sort("row", "row_b").select("row", "row_b").to_numpy().astype(np.int64))
    print(f"near-twin pairs {len(out):,}; skipped {big.height:,} rows in {big['core'].n_unique():,} name groups above {cap}")
    return out


def deleted_mask(s1, n_del, seed=3407):
    """Source 1 rows to delete so train looks like test: one business of each near-twin pair (coin flip; a pair is
    skipped once either side is gone), then random rows outside every twin pair until n_del are gone.

    Returns the mask and each row's reason: 0 kept, 1 twin, 2 random.
    """
    rng = np.random.default_rng(seed)
    tw = twins(s1)
    order = rng.permutation(len(tw))
    heads = rng.random(len(tw)) < 0.5
    gone = set()
    for (a, b), h in zip(tw[order].tolist(), heads.tolist()):
        if a not in gone and b not in gone:
            gone.add(a if h else b)
    reason = np.zeros(s1.height, dtype=np.int8)
    reason[list(gone)] = 1
    free = rng.permutation(np.setdiff1d(np.arange(s1.height), tw.ravel()))
    n_rand = n_del - len(gone)
    assert 0 <= n_rand <= len(free), f"{n_del:,} to delete, {len(gone):,} twins, {len(free):,} rows outside twin pairs"
    reason[free[:n_rand]] = 2
    print(f"deleted {n_del:,} of {s1.height:,} Source 1 rows ({n_del / s1.height:.1%}): {len(gone):,} twins, "
          f"{n_rand:,} random")
    return reason > 0, reason


def n_to_delete(n_s1_train, n_others_train, n_s1_test, n_others_test):
    """Source 1 rows to delete so train has test's records per business."""
    return n_s1_train - round(n_others_train / (n_others_test / n_s1_test))


def compact(idx, val, deleted, k=3):
    """Drop shortlisted Source 1 rows that are deleted and keep each record's first k survivors, best first.

    `idx` rows are best first with -1 for an empty slot; empty output slots are -1 with score 0.
    """
    assert idx.shape == val.shape and k <= idx.shape[1]
    alive = (idx >= 0) & ~deleted[np.maximum(idx, 0)]
    first = np.argsort(~alive, axis=1, kind="stable")[:, :k]
    ok = np.take_along_axis(alive, first, 1)
    return (np.where(ok, np.take_along_axis(idx, first, 1), -1).astype(idx.dtype),
            np.where(ok, np.take_along_axis(val, first, 1), 0).astype(val.dtype))
