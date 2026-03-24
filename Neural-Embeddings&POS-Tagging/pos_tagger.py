"""
POS Tagger – MLP with sliding-window context
Evaluates with SVD, Skip-Gram, and GloVe embeddings on the Brown Corpus
"""

import os, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import brown
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gensim.downloader as api

nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)

# ─── Hyperparameters ─────────────────────────────────────────────
CONTEXT_WINDOW = 2          # C  (total window = 2C+1)
HIDDEN1        = 256
HIDDEN2        = 128
DROPOUT        = 0.3
EPOCHS         = int(os.getenv("POS_EPOCHS", "15"))
BATCH_SIZE     = 256
LR             = 1e-3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════
# 1.  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════
print("Loading Brown corpus with universal tagset …")
tagged_sents = brown.tagged_sents(tagset='universal')

# Build tag set
all_tags = sorted({tag for sent in tagged_sents for _, tag in sent})
tag2idx  = {t: i for i, t in enumerate(all_tags)}
idx2tag  = {i: t for t, i in tag2idx.items()}
NUM_TAGS = len(all_tags)
print(f"  {NUM_TAGS} tags: {all_tags}")

# 80 / 10 / 10 split (sentence-level)
np.random.seed(42)
n = len(tagged_sents)
idx = np.random.permutation(n)
tr_end = int(0.8 * n)
va_end = int(0.9 * n)
train_sents = [tagged_sents[i] for i in idx[:tr_end]]
val_sents   = [tagged_sents[i] for i in idx[tr_end:va_end]]
test_sents  = [tagged_sents[i] for i in idx[va_end:]]
print(f"  Split: train={len(train_sents)}, val={len(val_sents)}, test={len(test_sents)}")


# ═══════════════════════════════════════════════════════════════════
# 2.  EMBEDDING LOADERS
# ═══════════════════════════════════════════════════════════════════
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def load_custom_embeddings(pt_path, w2i_path):
    """Load SVD or Skip-Gram embeddings + vocab."""
    emb = torch.load(pt_path, map_location="cpu").numpy()
    with open(w2i_path) as f:
        w2i = json.load(f)
    dim = emb.shape[1]
    # Add PAD (zeros) and UNK (random)
    pad_vec = np.zeros((1, dim), dtype=np.float32)
    unk_vec = np.random.randn(1, dim).astype(np.float32) * 0.01
    emb_full = np.vstack([emb, pad_vec, unk_vec])
    w2i[PAD_TOKEN] = len(emb_full) - 2
    w2i[UNK_TOKEN] = len(emb_full) - 1
    return emb_full, w2i, dim

def load_glove_embeddings():
    """Load GloVe and build matrix."""
    print("  Loading GloVe (glove-wiki-gigaword-100) …")
    glove = api.load("glove-wiki-gigaword-100")
    dim = glove.vector_size
    words = list(glove.key_to_index.keys())
    w2i = {w: i for i, w in enumerate(words)}
    emb = np.array([glove[w] for w in words], dtype=np.float32)
    # PAD + UNK
    pad_vec = np.zeros((1, dim), dtype=np.float32)
    unk_vec = np.random.randn(1, dim).astype(np.float32) * 0.01
    emb_full = np.vstack([emb, pad_vec, unk_vec])
    w2i[PAD_TOKEN] = len(emb_full) - 2
    w2i[UNK_TOKEN] = len(emb_full) - 1
    return emb_full, w2i, dim


# ═══════════════════════════════════════════════════════════════════
# 3.  DATASET
# ═══════════════════════════════════════════════════════════════════
class POSDataset(Dataset):
    def __init__(self, sents, w2i, tag2idx, C):
        self.samples = []
        pad_id = w2i[PAD_TOKEN]
        unk_id = w2i[UNK_TOKEN]
        for sent in sents:
            words = [w.lower() for w, _ in sent]
            tags  = [tag2idx[t] for _, t in sent]
            ids   = [w2i.get(w, unk_id) for w in words]
            # Pad sentence
            padded = [pad_id] * C + ids + [pad_id] * C
            for i, tag in enumerate(tags):
                window = padded[i : i + 2 * C + 1]
                self.samples.append((window, tag))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        w, t = self.samples[idx]
        return torch.tensor(w, dtype=torch.long), torch.tensor(t, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════
# 4.  MLP MODEL
# ═══════════════════════════════════════════════════════════════════
class POSTaggerMLP(nn.Module):
    def __init__(self, pretrained_emb, embed_dim, window_size, num_tags,
                 h1=HIDDEN1, h2=HIDDEN2, drop=DROPOUT, freeze_emb=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(pretrained_emb, dtype=torch.float32),
            freeze=freeze_emb
        )
        input_dim = embed_dim * (2 * window_size + 1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(h2, num_tags),
            # Softmax is applied after — we use CrossEntropyLoss which
            # applies LogSoftmax internally.  For inference we add it
            # explicitly in predict().
        )

    def forward(self, x):
        # x: (B, 2C+1) word indices
        e = self.embedding(x)          # (B, 2C+1, D)
        e = e.view(e.size(0), -1)      # (B, (2C+1)*D)
        return self.net(e)             # (B, num_tags)  — raw logits

    def predict(self, x):
        logits = self.forward(x)
        probs  = torch.softmax(logits, dim=-1)
        return probs.argmax(dim=-1)


# ═══════════════════════════════════════════════════════════════════
# 5.  TRAIN / EVALUATE HELPERS
# ═══════════════════════════════════════════════════════════════════
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_f1 = 0.0
    best_state  = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        val_acc, val_f1 = evaluate(model, val_loader)
        print(f"  Epoch {ep:2d}/{epochs}  loss={avg_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    return model

def evaluate(model, loader):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pred = model.predict(x).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(y.numpy())
    acc = accuracy_score(all_true, all_pred)
    f1  = f1_score(all_true, all_pred, average='macro', zero_division=0)
    return acc, f1

def full_evaluate(model, loader, label, all_tags, save_cm=False):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pred = model.predict(x).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(y.numpy())
    acc = accuracy_score(all_true, all_pred)
    f1  = f1_score(all_true, all_pred, average='macro', zero_division=0)
    print(f"\n  [{label}]  Test Accuracy = {acc:.4f}   Macro-F1 = {f1:.4f}")

    if save_cm:
        cm = confusion_matrix(all_true, all_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=all_tags)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix – {label}")
        plt.tight_layout()
        cm_path = os.path.join(BASE_DIR, f"confusion_matrix_{label.lower().replace(' ', '_')}.png")
        plt.savefig(cm_path, dpi=150)
        print(f"  Saved confusion matrix → {cm_path}")
        plt.close()

    return acc, f1


def collect_error_examples(model, sents, w2i, idx2tag, max_examples=5):
    """Collect misclassified token examples with short sentence context."""
    model.eval()
    errors = []
    pad_id = w2i[PAD_TOKEN]
    unk_id = w2i[UNK_TOKEN]

    with torch.no_grad():
        for sent in sents:
            words = [w for w, _ in sent]
            tags = [t for _, t in sent]
            ids = [w2i.get(w.lower(), unk_id) for w in words]
            padded = [pad_id] * CONTEXT_WINDOW + ids + [pad_id] * CONTEXT_WINDOW

            for i, (word, gold_tag) in enumerate(zip(words, tags)):
                window = padded[i : i + 2 * CONTEXT_WINDOW + 1]
                x = torch.tensor([window], dtype=torch.long, device=DEVICE)
                pred_idx = model.predict(x).item()
                pred_tag = idx2tag[pred_idx]

                if pred_tag != gold_tag:
                    left = words[max(0, i - 3):i]
                    right = words[i + 1:i + 4]
                    context = " ".join(left + [f"[{word}]"] + right)
                    errors.append({
                        "word": word,
                        "gold": gold_tag,
                        "pred": pred_tag,
                        "context": context,
                    })
                    if len(errors) >= max_examples:
                        return errors
    return errors


# ═══════════════════════════════════════════════════════════════════
# 6.  MAIN: run experiments for each embedding type
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = {}

    embedding_configs = []
    requested_variants = {
        x.strip().lower() for x in os.getenv("EMBEDDING_VARIANTS", "svd,skip-gram,glove").split(",")
        if x.strip()
    }

    # --- SVD ---
    svd_pt   = os.path.join(BASE_DIR, "embeddings", "svd.pt")
    svd_w2i  = os.path.join(BASE_DIR, "embeddings", "svd_word2idx.json")
    if "svd" in requested_variants and os.path.exists(svd_pt) and os.path.exists(svd_w2i):
        emb, w2i, dim = load_custom_embeddings(svd_pt, svd_w2i)
        embedding_configs.append(("SVD", emb, w2i, dim))
    elif "svd" in requested_variants:
        print("⚠  SVD embeddings not found – run svd_embeddings.py first.")

    # --- Skip-Gram ---
    sg_pt   = os.path.join(BASE_DIR, "embeddings", "skipgram.pt")
    sg_w2i  = os.path.join(BASE_DIR, "embeddings", "skipgram_word2idx.json")
    if "skip-gram" in requested_variants and os.path.exists(sg_pt) and os.path.exists(sg_w2i):
        emb, w2i, dim = load_custom_embeddings(sg_pt, sg_w2i)
        embedding_configs.append(("Skip-Gram", emb, w2i, dim))
    elif "skip-gram" in requested_variants:
        print("⚠  Skip-Gram embeddings not found – run word2vec.py first.")

    # --- GloVe ---
    if "glove" in requested_variants:
        emb_g, w2i_g, dim_g = load_glove_embeddings()
        embedding_configs.append(("GloVe", emb_g, w2i_g, dim_g))

    best_label = None
    best_f1    = -1.0
    best_w2i   = None

    for label, emb, w2i, dim in embedding_configs:
        print(f"\n{'═' * 60}")
        print(f"  Experiment: {label}  (dim={dim})")
        print(f"{'═' * 60}")

        train_ds = POSDataset(train_sents, w2i, tag2idx, CONTEXT_WINDOW)
        val_ds   = POSDataset(val_sents,   w2i, tag2idx, CONTEXT_WINDOW)
        test_ds  = POSDataset(test_sents,  w2i, tag2idx, CONTEXT_WINDOW)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0)

        model = POSTaggerMLP(emb, dim, CONTEXT_WINDOW, NUM_TAGS)
        model = train_model(model, train_loader, val_loader)
        acc, f1 = full_evaluate(model, test_loader, label, all_tags)
        results[label] = {"accuracy": acc, "macro_f1": f1}

        if f1 > best_f1:
            best_f1    = f1
            best_label = label
            best_model = model
            best_test_loader = test_loader
            best_w2i = w2i

    # ─── Confusion matrix for best model ──────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Best model: {best_label}  (F1={best_f1:.4f})")
    print(f"  Generating confusion matrix …")
    print(f"{'═' * 60}")
    full_evaluate(best_model, best_test_loader, f"Best ({best_label})",
                  all_tags, save_cm=True)

    if os.getenv("POS_DUMP_ERRORS", "0") == "1":
        examples = collect_error_examples(best_model, test_sents, best_w2i, idx2tag, max_examples=5)
        print(f"\n{'═' * 60}")
        print("  ERROR EXAMPLES (first 5 misclassified test tokens)")
        print(f"{'═' * 60}")
        if not examples:
            print("  No misclassified tokens found.")
        else:
            for i, ex in enumerate(examples, 1):
                print(f"  {i}. word='{ex['word']}'  gold={ex['gold']}  pred={ex['pred']}")
                print(f"     context: {ex['context']}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for lbl, m in results.items():
        print(f"  {lbl:12s}  Acc={m['accuracy']:.4f}   F1={m['macro_f1']:.4f}")

    print("\nDone.")
