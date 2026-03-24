import os
import json
import numpy as np
import torch
import nltk
from nltk.corpus import brown
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import gensim.downloader as api
from numpy.linalg import norm

nltk.download('brown', quiet=True)

# ─── Hyperparameters ─────────────────────────────────────────────
WINDOW_SIZE = 2       # context window on each side
VOCAB_SIZE  = 15000   # top-k words by frequency
EMBED_DIM   = 100     # SVD output dimension


# ─── 1. Build Vocabulary ─────────────────────────────────────────
print("Loading Brown corpus …")
sentences = brown.sents()
words = [w.lower() for sent in sentences for w in sent if w.isalpha()]
word_freq = Counter(words)
vocab = [w for w, _ in word_freq.most_common(VOCAB_SIZE)]
vocab_size = len(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

# Index-ify sentences (drop OOV)
indexed_sents = []
for sent in sentences:
    ids = [word2idx[w.lower()] for w in sent
           if w.isalpha() and w.lower() in word2idx]
    if ids:
        indexed_sents.append(ids)


# ─── 2. Co-occurrence Matrix ─────────────────────────────────────
print("Building co-occurrence matrix …")
rows, cols, vals = [], [], []
for sent in indexed_sents:
    for i, wid in enumerate(sent):
        lo = max(0, i - WINDOW_SIZE)
        hi = min(len(sent), i + WINDOW_SIZE + 1)
        for j in range(lo, hi):
            if j != i:
                rows.append(wid)
                cols.append(sent[j])
                vals.append(1.0)

co = csr_matrix((vals, (rows, cols)),
                shape=(vocab_size, vocab_size), dtype=np.float64)
print(f"Co-occurrence matrix shape: {co.shape}")


# ─── 3. PPMI weighting ───────────────────────────────────────────
print("Computing PPMI …")
total = float(co.sum())
row_sum = np.asarray(co.sum(axis=1)).ravel()
col_sum = np.asarray(co.sum(axis=0)).ravel()

coo = co.tocoo()
denom = row_sum[coo.row] * col_sum[coo.col]
valid = denom > 0

with np.errstate(divide='ignore', invalid='ignore'):
    pmi_vals = np.log2((coo.data[valid] * total) / denom[valid])

positive = np.isfinite(pmi_vals) & (pmi_vals > 0)
ppmi_sparse = csr_matrix(
    (pmi_vals[positive].astype(np.float32), (coo.row[valid][positive], coo.col[valid][positive])),
    shape=co.shape,
)
del co, coo, denom, pmi_vals


# ─── 4. SVD ───────────────────────────────────────────────────────
print(f"Running SVD (k={EMBED_DIM}) …")
u, s, _ = svds(ppmi_sparse, k=EMBED_DIM)
# Weight by sqrt(s) for balanced representation
embeddings = u * np.sqrt(s)
embeddings = embeddings.astype(np.float32)
print(f"Embeddings shape: {embeddings.shape}")


# ─── 5. Save ──────────────────────────────────────────────────────
os.makedirs("embeddings", exist_ok=True)
torch.save(torch.tensor(embeddings), "embeddings/svd.pt")
with open("embeddings/svd_word2idx.json", "w") as f:
    json.dump(word2idx, f)
print("Saved  embeddings/svd.pt  and  embeddings/svd_word2idx.json")


# ─── 6. Analogy evaluation ───────────────────────────────────────
def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-10)

def analogy(a, b, c, emb, w2i, i2w, top_k=5):
    """A : B :: C : ?   →  argmax cos(x, B − A + C)"""
    for w in (a, b, c):
        if w not in w2i:
            print(f"  '{w}' not in vocabulary!")
            return []
    vec = emb[w2i[b]] - emb[w2i[a]] + emb[w2i[c]]
    norms = norm(emb, axis=1)
    norms[norms == 0] = 1e-10
    sims = emb @ vec / (norms * (norm(vec) + 1e-10))
    for w in (a, b, c):
        sims[w2i[w]] = -np.inf
    top = np.argsort(sims)[-top_k:][::-1]
    return [(i2w[i], float(sims[i])) for i in top]

ANALOGY_SETS = [
    ("paris", "france", "delhi"),
    ("king",  "man",    "queen"),
    ("swim",  "swimming", "run"),
]

print("\n" + "=" * 50)
print("ANALOGY TEST – SVD embeddings")
print("=" * 50)
for a, b, c in ANALOGY_SETS:
    print(f"\n  {a} : {b} :: {c} : ?")
    for w, s in analogy(a, b, c, embeddings, word2idx, idx2word):
        print(f"    {w:15s}  {s:.4f}")


# ─── 7. Pre-trained GloVe analogy + bias ─────────────────────────
print("\nLoading GloVe (glove-wiki-gigaword-100) …")
glove = api.load("glove-wiki-gigaword-100")
print("GloVe loaded.")

print("\n" + "=" * 50)
print("ANALOGY TEST – GloVe")
print("=" * 50)
for a, b, c in ANALOGY_SETS:
    print(f"\n  {a} : {b} :: {c} : ?")
    try:
        res = glove.most_similar(positive=[b, c], negative=[a], topn=5)
        for w, s in res:
            print(f"    {w:15s}  {s:.4f}")
    except KeyError as e:
        print(f"    word missing: {e}")


# ─── Bias Audit ───────────────────────────────────────────────────
def bias_check(label, sim_fn):
    pairs = [
        ("doctor", "man"), ("doctor", "woman"),
        ("nurse",  "man"), ("nurse",  "woman"),
        ("homemaker", "man"), ("homemaker", "woman"),
    ]
    print(f"\n--- Gender-Bias Audit: {label} ---")
    for w1, w2 in pairs:
        try:
            print(f"  cos({w1}, {w2}) = {sim_fn(w1, w2):.4f}")
        except KeyError:
            print(f"  ({w1}, {w2}) – word missing")

def svd_sim(w1, w2):
    return cosine(embeddings[word2idx[w1]], embeddings[word2idx[w2]])

bias_check("SVD", svd_sim)
bias_check("GloVe", lambda w1, w2: float(glove.similarity(w1, w2)))

print("\nDone.")