import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

FIT_SAMPLE = 1_000_000


def vectors(s1_parts, other_parts, seed=0):
    """Character-trigram TF-IDF per text part (name, address), fitted on Source 1 plus a sample of Source 2/3.

    Returns one (records matrix, Source 1 matrix) pair per part; every row is unit length.
    """
    rng = np.random.default_rng(seed)
    out = []
    for s1_text, other_text in zip(s1_parts, other_parts):
        fit = other_text[rng.choice(len(other_text), size=min(FIT_SAMPLE, len(other_text)), replace=False)]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, dtype=np.float32)
        vec.fit(np.concatenate([s1_text, fit]))
        out.append((vec.transform(other_text), vec.transform(s1_text)))
    return out


def top_k(parts, k, n_threads, chunk=100_000):
    """Each record's k best Source 1 rows by the mean of the part cosines, best first; -1 where fewer than k
    share a trigram.

    Records go through in chunks so no single call runs long enough to stall a worker's heartbeat.
    """
    scale = np.float32((1 / len(parts)) ** 0.5)
    q = sp.hstack([p[0] for p in parts], format="csr") * scale
    d = (sp.hstack([p[1] for p in parts], format="csr") * scale).T
    idx = np.full((q.shape[0], k), -1, dtype=np.int32)
    val = np.zeros((q.shape[0], k), dtype=np.float32)
    for lo in range(0, q.shape[0], chunk):
        c = sp_matmul_topn(q[lo:lo + chunk], d, top_n=k, sort=True, n_threads=n_threads)
        counts = np.diff(c.indptr)
        rows = lo + np.repeat(np.arange(c.shape[0]), counts)
        pos = np.arange(c.nnz) - np.repeat(c.indptr[:-1], counts)
        idx[rows, pos] = c.indices
        val[rows, pos] = c.data
    return idx, val
