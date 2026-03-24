"""
Word2Vec – Skip-Gram with Negative Sampling
Trained on the Brown Corpus
"""

import os, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import brown
from collections import Counter
from numpy.linalg import norm
import gensim.downloader as api

nltk.download('brown', quiet=True)

# ─── Hyperparameters ─────────────────────────────────────────────
VOCAB_SIZE   = int(os.getenv("W2V_VOCAB_SIZE", "15000"))
EMBED_DIM    = int(os.getenv("W2V_EMBED_DIM", "100"))
WINDOW_SIZE  = int(os.getenv("W2V_WINDOW_SIZE", "2"))
NEG_SAMPLES  = int(os.getenv("W2V_NEG_SAMPLES", "5"))
EPOCHS       = int(os.getenv("W2V_EPOCHS", "4"))
BATCH_SIZE   = int(os.getenv("W2V_BATCH_SIZE", "1024"))
LR           = float(os.getenv("W2V_LR", "0.001"))
SUBSAMPLE_T  = float(os.getenv("W2V_SUBSAMPLE_T", "1e-4"))
MAX_PAIRS    = int(os.getenv("W2V_MAX_PAIRS", "0"))
device_name  = os.getenv("W2V_DEVICE", "auto")
if device_name == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(device_name)


# ─── 1. Vocabulary ───────────────────────────────────────────────
print("Loading Brown corpus …")
sentences = brown.sents()
words = [w.lower() for sent in sentences for w in sent if w.isalpha()]
word_freq = Counter(words)
total_words = sum(word_freq.values())

vocab = [w for w, _ in word_freq.most_common(VOCAB_SIZE)]
vocab_size = len(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

# Frequencies for negative-sampling distribution (unigram^0.75)
freqs = np.array([word_freq[vocab[i]] for i in range(vocab_size)], dtype=np.float64)
freqs_pow = freqs ** 0.75
neg_dist = freqs_pow / freqs_pow.sum()

# Sub-sampling probabilities 
sub_probs = {}
for w in vocab:
    f = word_freq[w] / total_words
    sub_probs[w] = 1.0 - math.sqrt(SUBSAMPLE_T / f)


# ─── 2. Build training pairs ─────────────────────────────────────
print("Generating skip-gram pairs …")
pairs = []   # (target_idx, context_idx)
for sent in sentences:
    ids = [word2idx[w.lower()] for w in sent
           if w.isalpha() and w.lower() in word2idx]
    for i, tid in enumerate(ids):
        # Sub-sample frequent words
        if np.random.random() < sub_probs.get(idx2word[tid], 0):
            continue
        lo = max(0, i - WINDOW_SIZE)
        hi = min(len(ids), i + WINDOW_SIZE + 1)
        for j in range(lo, hi):
            if j != i:
                pairs.append((tid, ids[j]))

pairs = np.array(pairs, dtype=np.int64)
if MAX_PAIRS > 0 and len(pairs) > MAX_PAIRS:
    choice = np.random.choice(len(pairs), size=MAX_PAIRS, replace=False)
    pairs = pairs[choice]
    print(f"  Capped training pairs to {len(pairs):,} using W2V_MAX_PAIRS.")
print(f"  {len(pairs):,} training pairs generated.")


# ─── 3. Dataset / DataLoader ─────────────────────────────────────
class SkipGramDataset(Dataset):
    def __init__(self, pairs, neg_dist, k):
        self.pairs = pairs
        self.neg_dist = neg_dist
        self.k = k
        self.vocab_size = len(neg_dist)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target, context = self.pairs[idx]
        negs = np.random.choice(self.vocab_size, size=self.k,
                                replace=False, p=self.neg_dist)
        return (torch.tensor(target, dtype=torch.long),
                torch.tensor(context, dtype=torch.long),
                torch.tensor(negs, dtype=torch.long))

dataset = SkipGramDataset(pairs, neg_dist, NEG_SAMPLES)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=0, drop_last=True)


# ─── 4. Model ────────────────────────────────────────────────────
class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.target_emb  = nn.Embedding(vocab_size, embed_dim)
        self.context_emb = nn.Embedding(vocab_size, embed_dim)
        # Xavier init
        nn.init.xavier_uniform_(self.target_emb.weight)
        nn.init.xavier_uniform_(self.context_emb.weight)

    def forward(self, target, context, negatives):
        # target:    (B,)
        # context:   (B,)
        # negatives: (B, k)
        t = self.target_emb(target)           # (B, D)
        c = self.context_emb(context)         # (B, D)
        n = self.context_emb(negatives)       # (B, k, D)

        # Positive score
        pos_score = torch.sum(t * c, dim=1)                     # (B,)
        pos_loss  = torch.nn.functional.logsigmoid(pos_score)   # (B,)

        # Negative scores
        neg_score = torch.bmm(n, t.unsqueeze(2)).squeeze(2)     # (B, k)
        neg_loss  = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)

        return -(pos_loss + neg_loss).mean()

model = SkipGramNS(vocab_size, EMBED_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)


# ─── 5. Train ────────────────────────────────────────────────────
print(f"Training on {DEVICE} for {EPOCHS} epochs …")
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for batch_i, (t, c, n) in enumerate(loader):
        t, c, n = t.to(DEVICE), c.to(DEVICE), n.to(DEVICE)
        loss = model(t, c, n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / (batch_i + 1)
    print(f"  Epoch {epoch}/{EPOCHS}  loss={avg:.4f}")


# ─── 6. Extract & save embeddings ────────────────────────────────
embeddings = model.target_emb.weight.detach().cpu().numpy()
os.makedirs("embeddings", exist_ok=True)
torch.save(torch.tensor(embeddings), "embeddings/skipgram.pt")
with open("embeddings/skipgram_word2idx.json", "w") as f:
    json.dump(word2idx, f)
print("Saved  embeddings/skipgram.pt  and  embeddings/skipgram_word2idx.json")


# ─── 7. Analogy evaluation ───────────────────────────────────────
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
print("ANALOGY TEST – Skip-Gram")
print("=" * 50)
for a, b, c in ANALOGY_SETS:
    print(f"\n  {a} : {b} :: {c} : ?")
    for w, s in analogy(a, b, c, embeddings, word2idx, idx2word):
        print(f"    {w:15s}  {s:.4f}")


# ─── 8. GloVe analogy + bias ─────────────────────────────────────
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


# ─── Bias Check ───────────────────────────────────────────────────
def cosine(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-10))

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

def sg_sim(w1, w2):
    return cosine(embeddings[word2idx[w1]], embeddings[word2idx[w2]])

bias_check("Skip-Gram", sg_sim)
bias_check("GloVe", lambda w1, w2: float(glove.similarity(w1, w2)))

print("\nDone.")
