import unicodedata

import polars as pl
from anyascii import anyascii

from text import clean

INDIC = ("ऀ", "෿")


def words(name):
    """The words common.clean would give, in plain Python: letters, digits and combining marks, Latin accents dropped."""
    name = unicodedata.normalize("NFKD", name.lower())
    kept = ("" if "̀" <= c <= "ͯ" else c if c.isalnum() or unicodedata.category(c).startswith("M") else " "
            for c in name)
    return "".join(kept).split()


def learn(pairs):
    """Script word to Latin word, from (script_name, latin_name) pairs of true matches.

    Names with the same word count are aligned position by position; a script word keeps its most frequent
    Latin reading when seen at least twice and in at least half its alignments.
    """
    tok = pairs.with_columns(s=clean("script_name").str.split(" "), l=clean("latin_name").str.split(" "))
    aligned = tok.filter(pl.col("s").list.len() == pl.col("l").list.len())
    counts = aligned.explode("s", "l").group_by("s", "l").len()
    best = (counts.sort("len", descending=True).group_by("s", maintain_order=True)
            .agg(latin=pl.col("l").first(), n=pl.col("len").first(), total=pl.col("len").sum()))
    keep = best.filter((pl.col("n") >= 2) & (pl.col("n") / pl.col("total") >= 0.5))
    print(f"script-name pairs {pairs.height:,}, aligned {aligned.height:,}, script words kept {keep.height:,} of {best.height:,}")
    return dict(zip(keep["s"].to_list(), keep["latin"].to_list()))


def translate(name, table):
    return " ".join(table.get(w) or " ".join(anyascii(w).lower().split()) for w in words(name))


def romanise(names, table):
    """Rewrite names that contain an Indian-script letter word by word; leave every other name as it is."""
    lo, hi = INDIC
    return pl.Series([translate(x, table) if any(lo <= c <= hi for c in x) else x for x in names.fill_null("")])
