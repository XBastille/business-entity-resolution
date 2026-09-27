import hashlib

import polars as pl

KEY = ["source1_entity_id", "other_id"]


def read_source(path):
    """An organiser TSV, every column a string and quoting off; the row count must match the file's lines."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    with open(path, "rb") as f:
        data_lines = sum(1 for _ in f) - 1
    assert df.height == data_lines, f"{path}: read {df.height} rows, file has {data_lines} data lines"
    return df


def write_outputs(s1_ids, matches, candidates, out_dir):
    """Both submission files: one row per test Source 1 id, comma-separated Source 2/3 ids, empty when none.

    `matches` and `candidates` are frames of (source1_entity_id, other_id) pairs; every match must be a candidate.
    """
    assert matches.join(candidates, on=KEY, how="anti").height == 0, "a match outside the candidate set"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = pl.DataFrame({"source1_entity_id": s1_ids})
    for name, col, pairs in (("matching_results.tsv", "matched_entity_ids", matches),
                             ("candidate_pairs.tsv", "candidate_entity_ids", candidates)):
        assert not pairs.select(KEY).is_duplicated().any(), f"{name}: duplicate pairs"
        assert pairs["other_id"].str.contains(r"^S[23]-").all(), f"{name}: an id that is not Source 2/3"
        assert pairs.join(ids, on="source1_entity_id", how="anti").height == 0, f"{name}: unknown Source 1 id"
        lists = pairs.group_by("source1_entity_id").agg(pl.col("other_id").sort().str.join(",").alias(col))
        rows = ids.join(lists, on="source1_entity_id", how="left").with_columns(pl.col(col).fill_null(""))
        path = out_dir / name
        # newline="\n": the validator strips only "\n", so Windows "\r\n" would break the header
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"source1_entity_id\t{col}\n")
            f.writelines(f"{a}\t{b}\n" for a, b in rows.iter_rows())
        filled = (rows[col] != "").sum()
        print(f"{name}: {rows.height:,} rows, {filled:,} non-empty, {pairs.height:,} ids, "
              f"sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
