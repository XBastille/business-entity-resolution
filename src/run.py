import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from features import base_features, prep
from files import read_source, write_outputs
from pipeline import K, decide, fit, macro_f05, owner_rows, script_table, truth_pairs
from shortlist import top_k, vectors

VAL_SHARE = 0.2
GRID = np.round(np.arange(0.30, 0.91, 0.025), 3)


def log(msg, t0=[time.perf_counter()]):
    print(f"[{time.perf_counter() - t0[0]:7.0f}s] {msg}", flush=True)


def country_pairs(s1, others, table, threads):
    """Shortlist and features for one country; returns the prepped frames and the scored pairs."""
    s1, others = prep(s1, table), prep(others, table)
    parts = vectors([s1["n"].to_numpy(), s1["a"].to_numpy()], [others["n"].to_numpy(), others["a"].to_numpy()])
    idx, val = top_k(parts, K, n_threads=threads)
    rec, cand, X = base_features(s1, others, idx, val, parts)
    return s1, others, rec, cand, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="folder holding train/ and test/")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    read = lambda split, name: read_source(args.data_dir / split / f"{split}_{name}.tsv")
    tr1 = read("train", "source1")
    tro = pl.concat([read("train", "source2"), read("train", "source3")])
    te1 = read("test", "source1")
    teo = pl.concat([read("test", "source2"), read("test", "source3")])
    pairs = truth_pairs(read("train", "ground_truth"))
    log(f"train S1 {tr1.height:,}, S2/S3 {tro.height:,}, true pairs {pairs.height:,}; test S1 {te1.height:,}, S2/S3 {teo.height:,}")

    is_val_id = tr1.select("entity_id", val=pl.int_range(pl.len()).shuffle(seed=0) < int(VAL_SHARE * tr1.height))
    india = tro.filter(pl.col("country") == "India")
    train_fold_s1 = tr1.join(is_val_id.filter(~pl.col("val")), on="entity_id", how="semi")
    tables = {"train_fold": script_table(train_fold_s1, india, pairs), "all": script_table(tr1, india, pairs)}

    data = {}
    for c in sorted(tr1["country"].unique().to_list()):
        table = tables["train_fold"] if c == "India" else None
        s1, ot, rec, cand, X = country_pairs(tr1.filter(pl.col("country") == c), tro.filter(pl.col("country") == c),
                                             table, args.threads)
        owner = owner_rows(s1, ot, pairs)
        is_val = s1.join(is_val_id, on="entity_id", how="left", maintain_order="left")["val"].to_numpy()
        any_val = np.zeros(ot.height, dtype=bool)
        np.logical_or.at(any_val, rec, is_val[cand])
        data[c] = dict(rec=rec, cand=cand, X=X, y=owner[rec] == cand, train=~any_val[rec], is_val=is_val,
                       k=np.bincount(owner[owner >= 0], minlength=s1.height), n_rec=ot.height)
        log(f"train {c}: {len(rec):,} pairs, {len(rec) / s1.height:.1f} per S1")

    train_cs = list(data)
    Xtr = np.concatenate([data[c]["X"][data[c]["train"]] for c in train_cs])
    ytr = np.concatenate([data[c]["y"][data[c]["train"]] for c in train_cs])
    model = fit(Xtr, ytr, args.threads)
    bars = {}
    for c in train_cs:
        d = data[c]
        ev = ~d["train"]
        prob = model.predict(d["X"][ev], num_threads=args.threads)
        scores = [macro_f05(d["cand"][ev], decide(d["rec"][ev], prob, d["n_rec"], t), d["y"][ev], d["k"], d["is_val"])[0]
                  for t in GRID]
        bars[c] = float(GRID[int(np.argmax(scores))])
        log(f"held-out {c}: macro F0.5 {max(scores):.5f} at bar {bars[c]}")
    pooled = float(np.mean(list(bars.values())))

    final = fit(np.concatenate([data[c]["X"] for c in train_cs]), np.concatenate([data[c]["y"] for c in train_cs]),
                args.threads)
    matches, candidates = [], []
    for c in sorted(te1["country"].unique().to_list()):
        table = tables["all"] if c == "India" else None
        s1, ot, rec, cand, X = country_pairs(te1.filter(pl.col("country") == c), teo.filter(pl.col("country") == c),
                                             table, args.threads)
        prob = final.predict(X, num_threads=args.threads)
        t = bars.get(c, pooled)
        acc = decide(rec, prob, ot.height, t)
        frame = pl.DataFrame({"source1_entity_id": s1["entity_id"].to_numpy()[cand], "other_id": ot["entity_id"].to_numpy()[rec]})
        candidates.append(frame)
        matches.append(frame.filter(pl.Series(acc)))
        log(f"test {c}: bar {t:.4f}, {acc.sum() / s1.height:.3f} accepted per S1")
    write_outputs(te1["entity_id"], pl.concat(matches), pl.concat(candidates), args.out_dir)


if __name__ == "__main__":
    main()
