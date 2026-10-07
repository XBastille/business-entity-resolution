import polars as pl


def clean(col):
    """Lowercase, strip Latin accents, turn every run of other non-letters/digits into one space.

    Combining marks (\\p{M}) are kept: in Indian scripts the vowel signs are marks, and dropping
    them splits one word into fragments.
    """
    return (pl.col(col).fill_null("").str.to_lowercase().str.normalize("NFKD")
            .str.replace_all(r"[̀-ͯ]", "").str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
            .str.strip_chars())


def numbers(col):
    """Every digit run in the column, leading zeros stripped: copies pad numbers ("0110" for "110")."""
    return (pl.col(col).fill_null("").str.extract_all(r"\d+")
            .list.eval(pl.element().str.strip_chars_start("0").replace("", "0")))
