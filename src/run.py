import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from entity_resolution.blocking.shortlist import top_k, vectors
from entity_resolution.blocking.testlike import compact, deleted_mask, n_to_delete
from entity_resolution.core import BIG, K, SEED, fit, macro_f05, owner_rows, script_table, truth_pairs
from entity_resolution.data.io import read_source, write_outputs
from entity_resolution.decision.rules import (BARS, CHEAP_COLS, accepted_per_s1, bar_curve, count_bar, drawbridge,
                                              fit_cheap, safe_range, trim_mask, tune_drawbridge)
from entity_resolution.features.base import base_features, prep
from entity_resolution.features.families import families
from entity_resolution.features.sharp import sharp
from entity_resolution.models import pair_model

VAL_SHARE = 0.2                  # Source 1 businesses held out; half tune the decision rule, half report
SEARCH_K = 10                    # train shortlist depth before test-like deletion; 3 survivors are compared
BAND = (0.05, 0.95)              # tree probabilities that the pair model re-judges
EVAL_SHARE = 0.25                # share of held-out businesses whose pairs fit the tree + pair model stacker


def log(msg, t0=[time.perf_counter()]):
    print(f"[{time.perf_counter() - t0[0]:7.0f}s] {msg}", flush=True)


def cached(path, make):
    """np.savez cache of a dict of arrays, so a long run can restart without recomputing finished stages."""
    if not path.exists():
        np.savez(path, **make())
    return dict(np.load(path))


def shortlist(s1, ot, table, k, threads):
    """Cleaned frames, the TF-IDF name and address parts fitted on this Source 1, and every record's top-k rows."""
    s1, ot = prep(s1, table), prep(ot, table)
    parts = vectors([s1["n"].to_numpy(), s1["a"].to_numpy()], [ot["n"].to_numpy(), ot["a"].to_numpy()])
    idx, val = top_k(parts, k, n_threads=threads)
    return s1, ot, parts, idx, val


def comparisons(s1, ot, idx, val, parts, threads):
    """The 78 comparison columns of every shortlisted pair: 24 base, 34 family and 20 sharp columns."""
    rec, cand, Xb = base_features(s1, ot, idx, val, parts)
    _, Xf = families(s1, ot, idx, val, rec, cand, workers=threads)
    _, Xs = sharp(s1, ot, idx, val, rec, cand, workers=threads)
    return {"rec": rec, "cand": cand, "X": np.hstack([Xb, Xf, Xs]).astype(np.float32)}


def train_country(s1, ot, pairs, table, n_test_s1, n_test_ot, work, threads):
    """A test-like training frame: delete Source 1 businesses until records per business match test (their copies
    become records with no owner, as test's decoys are), re-rank each record's top 10 without them, compare the first
    3 survivors, and mark the held-out businesses."""
    s1, ot, parts, idx10, val10 = shortlist(s1, ot, table, SEARCH_K, threads)
    rng = np.random.default_rng(SEED)
    is_val, tune = rng.random(s1.height) < VAL_SHARE, rng.random(s1.height) < 0.5
    deleted, _ = deleted_mask(s1, n_to_delete(s1.height, ot.height, n_test_s1, n_test_ot), seed=SEED)
    idx, val = compact(idx10, val10, deleted, k=3)
    keep = ~deleted
    new_pos = np.full(s1.height, -1)
    new_pos[keep] = np.arange(keep.sum())
    idx = np.where(idx >= 0, new_pos[np.maximum(idx, 0)], -1).astype(np.int32)
    s1 = s1.with_columns(is_val=pl.Series(is_val), tune=pl.Series(tune)).filter(pl.Series(keep))
    owner = owner_rows(s1, ot, pairs)
    z = cached(work / "pairs.npz", lambda: comparisons(s1, ot, idx, val, [(q, d[keep]) for q, d in parts], threads))
    rec, cand = z["rec"], z["cand"]
    is_val, tune = s1["is_val"].to_numpy(), s1["tune"].to_numpy()
    any_val = np.zeros(ot.height, dtype=bool)
    np.logical_or.at(any_val, rec, is_val[cand])
    log(f"train {work.name}: {int(deleted.sum()):,} businesses deleted, {len(rec):,} pairs")
    return dict(rec=rec, cand=cand, X=z["X"], y=owner[rec] == cand, train=~any_val[rec], n_rec=ot.height,
                k=np.bincount(owner[owner >= 0], minlength=s1.height), v_tune=is_val & tune, v_report=is_val & ~tune,
                s1_text=pair_model.text(s1), ot_text=pair_model.text(ot))


def trim(d, threads):
    """The first model's 1% cut (F5). Training pairs are trimmed out of fold, two folds by record."""
    cheap, tr, fold = np.zeros(len(d["rec"])), d["train"], d["rec"] % 2
    for f in (0, 1):
        m = fit_cheap(d["X"][tr & (fold != f)], d["y"][tr & (fold != f)], threads)
        cheap[tr & (fold == f)] = m.predict(d["X"][tr & (fold == f)][:, CHEAP_COLS])
    cheap[~tr] = fit_cheap(d["X"][tr], d["y"][tr], threads).predict(d["X"][~tr][:, CHEAP_COLS])
    d["keep"] = trim_mask(cheap)
    d["ev"] = ~d["train"] & d["keep"]


def tune_rule(rec, cand, prob, y, k, tune_mask, report_mask, n_rec):
    """Drawbridge floor and factor, then the per-record bar, chosen on tune_mask; the macro F0.5 on report_mask."""
    floor, factor, _ = tune_drawbridge(rec, cand, prob, y, k, tune_mask, n_rec)
    curve = bar_curve(rec, cand, prob, y, k, tune_mask, n_rec, floor, factor)
    bar = float(BARS[int(np.argmax(curve))])
    acc = drawbridge(rec, cand, prob, n_rec, len(k), floor, factor) & (prob >= bar)
    return dict(floor=floor, factor=factor, bar=bar, safe=safe_range(curve, BARS),
                target=float(accepted_per_s1(acc & tune_mask[cand], tune_mask.sum())),
                report=macro_f05(cand, acc, y, k, report_mask))


def decide_test(rec, cand, prob, n_rec, n_s1, t, safe):
    """F2's count rule on test: the bar that accepts as many records per Source 1 row as on tune, kept in the safe
    range, on top of Drawbridge."""
    raw = count_bar(rec, cand, prob, n_rec, n_s1, t["floor"], t["factor"], t["target"])
    bar = min(max(raw, safe[0]), safe[1])
    return drawbridge(rec, cand, prob, n_rec, n_s1, t["floor"], t["factor"]) & (prob >= bar), bar


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def pair_data(data, n_first, n_more):
    """Pair model data, in the order it was drawn: n_first training pairs and 25% of held-out businesses (their
    records' pairs are the stacker's data), then n_more training pairs with no text in the first set."""
    rng = np.random.default_rng(SEED)
    total = sum(int((d["train"] & d["keep"]).sum()) for d in data.values())
    first, pool = [], []
    for c, d in data.items():
        tr = np.flatnonzero(d["train"] & d["keep"])
        pick = rng.choice(tr, size=min(len(tr), round(n_first * len(tr) / total)), replace=False)
        texts = lambda rows: pl.DataFrame({"a": d["ot_text"][d["rec"][rows]], "b": d["s1_text"][d["cand"][rows]],
                                           "y": d["y"][rows].astype(np.int8), "country": c})
        first.append(texts(pick))
        pool.append(texts(tr))
        val_biz = np.flatnonzero(d["v_tune"] | d["v_report"])
        chosen = np.zeros(len(d["k"]), dtype=bool)
        chosen[rng.choice(val_biz, size=round(EVAL_SHARE * len(val_biz)), replace=False)] = True
        touch = np.zeros(d["n_rec"], dtype=bool)
        touch[d["rec"][d["ev"] & chosen[d["cand"]]]] = True
        d["sel"] = np.flatnonzero(d["ev"] & touch[d["rec"]])
        d["chosen"] = np.zeros(len(d["k"]), dtype=bool)
        d["chosen"][d["cand"][d["sel"]][chosen[d["cand"][d["sel"]]]]] = True
    first = pl.concat(first).sample(fraction=1.0, shuffle=True, seed=SEED)
    more = pl.concat(pool).join(first.select("a", "b", "country"), on=["a", "b", "country"], how="anti")
    return pl.concat([first, more.sample(n=n_more, shuffle=True, seed=SEED + 1)])


def main():
    ap = argparse.ArgumentParser(description="Business entity resolution: test-like trees plus a cross-encoder.")
    ap.add_argument("--data-dir", type=Path, required=True, help="folder holding train/ and test/")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True, help="caches of finished stages")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--pair-model", default=pair_model.MODEL)
    ap.add_argument("--pair-first", type=int, default=1_000_000)
    ap.add_argument("--pair-more", type=int, default=3_000_000)
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    read = lambda split, name: read_source(args.data_dir / split / f"{split}_{name}.tsv")
    tr1, tro = read("train", "source1"), pl.concat([read("train", "source2"), read("train", "source3")])
    te1, teo = read("test", "source1"), pl.concat([read("test", "source2"), read("test", "source3")])
    pairs = truth_pairs(read("train", "ground_truth"))
    log(f"train S1 {tr1.height:,}, S2/S3 {tro.height:,}, true pairs {pairs.height:,}; test S1 {te1.height:,}, S2/S3 {teo.height:,}")

    fold_val = tr1.select("entity_id", val=pl.int_range(pl.len()).shuffle(seed=0) < int(VAL_SHARE * tr1.height))
    fold_s1 = tr1.join(fold_val.filter(~pl.col("val")), on="entity_id", how="semi")
    tables = {"train": script_table(fold_s1, tro, pairs), "test": script_table(tr1, tro, pairs)}
    size = lambda f, c: f.filter(pl.col("country") == c).height
    train_cs = sorted(tr1["country"].unique().to_list(), key=lambda c: -size(tr1, c))
    test_cs = sorted(te1["country"].unique().to_list(), key=lambda c: -size(te1, c))

    data = {}
    for c in train_cs:
        work = args.work_dir / f"train_{c}"
        work.mkdir(exist_ok=True)
        data[c] = train_country(tr1.filter(pl.col("country") == c), tro.filter(pl.col("country") == c), pairs,
                                tables["train"], size(te1, c), size(teo, c), work, args.threads)
        trim(data[c], args.threads)
        log(f"trim {c}: {data[c]['keep'].sum() / len(data[c]['k']):.2f} pairs per S1 kept, "
            f"true pairs kept {(data[c]['keep'] & data[c]['y']).sum() / data[c]['y'].sum():.5f}")

    model_path = args.work_dir / "tree.txt"
    if not model_path.exists():
        tr = [d["train"] & d["keep"] for d in data.values()]
        vt = [d["ev"] & d["v_tune"][d["cand"]] for d in data.values()]
        cat = lambda key, masks: np.concatenate([d[key][m] for d, m in zip(data.values(), masks)])
        fit(cat("X", tr), cat("y", tr), args.threads, rounds=3000, params=BIG,
            valid=(cat("X", vt), cat("y", vt))).save_model(str(model_path))
    tree = lgb.Booster(model_file=str(model_path))
    for c, d in data.items():
        ev = d["ev"]
        d["prob"] = tree.predict(d["X"][ev], num_threads=args.threads)
        t = tune_rule(d["rec"][ev], d["cand"][ev], d["prob"], d["y"][ev], d["k"], d["v_tune"], d["v_report"], d["n_rec"])
        log(f"trees alone, V-report {c}: macro F0.5 {t['report'][0]:.5f}")

    cheap_all = fit_cheap(np.concatenate([d["X"][d["train"]] for d in data.values()]),
                          np.concatenate([d["y"][d["train"]] for d in data.values()]), args.threads)
    test = {}
    for c in test_cs:
        work = args.work_dir / f"test_{c}"
        work.mkdir(exist_ok=True)
        s1, ot, parts, idx, val = shortlist(te1.filter(pl.col("country") == c), teo.filter(pl.col("country") == c),
                                            tables["test"], K, args.threads)
        z = cached(work / "pairs.npz", lambda: comparisons(s1, ot, idx, val, parts, args.threads))
        keep = trim_mask(cheap_all.predict(z["X"][:, CHEAP_COLS], num_threads=args.threads))
        rec, cand = z["rec"][keep], z["cand"][keep]
        prob = tree.predict(z["X"][keep], num_threads=args.threads)
        band = (prob > BAND[0]) & (prob < BAND[1])
        test[c] = dict(rec=rec, cand=cand, prob=prob, band=band, n_s1=s1.height, n_rec=ot.height,
                       s1_id=s1["entity_id"].to_numpy(), ot_id=ot["entity_id"].to_numpy(),
                       a=pair_model.text(ot)[rec[band]], b=pair_model.text(s1)[cand[band]])
        log(f"test {c}: {keep.sum() / s1.height:.3f} candidates per S1, {band.sum():,} pairs in the uncertain band")

    train_pairs = pair_data(data, args.pair_first, args.pair_more)
    model, tok = pair_model.train(train_pairs["a"].to_numpy(), train_pairs["b"].to_numpy(), train_pairs["y"].to_numpy(),
                                  args.work_dir / "pair_model", model_name=args.pair_model)
    for c, d in data.items():
        sel = d["sel"]
        d["xl"] = pair_model.logits(model, tok, d["ot_text"][d["rec"][sel]], d["s1_text"][d["cand"][sel]])
        d["tp"] = tree.predict(d["X"][sel], num_threads=args.threads)
    for c, t in test.items():
        t["xl"] = pair_model.logits(model, tok, t["a"], t["b"])

    band_of = lambda p: (p > BAND[0]) & (p < BAND[1])
    fit_rows = [(d, d["v_tune"][d["cand"][d["sel"]]] & d["chosen"][d["cand"][d["sel"]]] & band_of(d["tp"])) for d in data.values()]
    stacker = LogisticRegression(C=1e4, max_iter=1000).fit(
        np.concatenate([np.column_stack([logit(d["tp"][m]), d["xl"][m]]) for d, m in fit_rows]),
        np.concatenate([d["y"][d["sel"]][m] for d, m in fit_rows]))
    log(f"stacker: {stacker.coef_.round(4).tolist()} intercept {stacker.intercept_.round(4).tolist()}")
    blend = lambda tp, xl, m: np.where(m, stacker.predict_proba(np.column_stack([logit(tp), xl]))[:, 1], tp)

    rules = {}
    for c, d in data.items():
        sel = d["sel"]
        p = blend(d["tp"], d["xl"], band_of(d["tp"]))
        tune_mask, report_mask = d["v_tune"] & d["chosen"], d["v_report"] & d["chosen"]
        rules[c] = tune_rule(d["rec"][sel], d["cand"][sel], p, d["y"][sel], d["k"], tune_mask, report_mask, d["n_rec"])
        log(f"trees + pair model, V-report {c} (evaluation businesses): macro F0.5 {rules[c]['report'][0]:.5f}")
    pooled = {k: float(np.mean([rules[c][k] for c in rules])) for k in ("floor", "factor", "target")}
    pooled_safe = (min(r["safe"][0] for r in rules.values()), max(r["safe"][1] for r in rules.values()))

    matches, candidates = [], []
    for c, t in test.items():
        prob = t["prob"].copy()
        prob[t["band"]] = stacker.predict_proba(np.column_stack([logit(prob[t["band"]]), t["xl"]]))[:, 1]
        rule, safe = rules.get(c, pooled), rules[c]["safe"] if c in rules else pooled_safe
        acc, bar = decide_test(t["rec"], t["cand"], prob, t["n_rec"], t["n_s1"], rule, safe)
        frame = pl.DataFrame({"source1_entity_id": t["s1_id"][t["cand"]], "other_id": t["ot_id"][t["rec"]]})
        candidates.append(frame)
        matches.append(frame.filter(pl.Series(acc)))
        log(f"test {c}: bar {bar:.4f}, {acc.sum() / t['n_s1']:.3f} accepted per S1")
    write_outputs(te1["entity_id"], pl.concat(matches), pl.concat(candidates), args.out_dir)


if __name__ == "__main__":
    main()
