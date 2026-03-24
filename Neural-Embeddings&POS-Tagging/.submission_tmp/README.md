# Neural Embeddings & POS Tagging

## 1) Required Dependencies

Install the required Python packages:

```bash
pip install numpy torch scipy scikit-learn gensim nltk matplotlib
```

## 2) Required Conditions

- Use Python 3.10+ (project tested in a Python virtual environment).
- Run commands from this directory: `Neural-Embeddings&POS-Tagging/`.
- Ensure internet is available on first run for:
	- NLTK corpus download (`brown`, `universal_tagset`)
	- Gensim GloVe download (`glove-wiki-gigaword-100`)
- Keep script execution order as documented below because `pos_tagger.py` depends on generated embedding files.
- If your machine has limited resources, use the runtime-control environment variables listed in Section 4.

## 3) Required Execution Steps (In Order)

### Step 1: SVD Embeddings (Task 1.1 + Task 2)

```bash
python svd_embeddings.py
```

This step builds a co-occurrence matrix from the Brown corpus, applies PPMI + truncated SVD, saves `embeddings/svd.pt` and `embeddings/svd_word2idx.json`, and prints analogy/bias outputs.

### Step 2: Word2Vec Skip-Gram (Task 1.2 + Task 2)

```bash
python word2vec.py
```

This step trains Skip-Gram with negative sampling, saves `embeddings/skipgram.pt` and `embeddings/skipgram_word2idx.json`, and prints analogy/bias outputs.

### Step 3: POS Tagger MLP (Task 3)

```bash
python pos_tagger.py
```

This step loads SVD/Skip-Gram/GloVe embeddings, trains and evaluates the MLP POS tagger, reports accuracy and macro-F1, and saves confusion matrix images.

### Step 4: Build Submission Zip

```bash
bash package_and_validate.sh
```

This step creates and validates `2024114009_A2.zip` with the required submission structure.

## 4) Optional Runtime Controls (for slower systems)

### Word2Vec controls

- `W2V_DEVICE` (e.g., `cpu`)
- `W2V_VOCAB_SIZE` (e.g., `15000`)
- `W2V_EPOCHS` (e.g., `1`)
- `W2V_BATCH_SIZE` (e.g., `4096`)
- `W2V_MAX_PAIRS` (e.g., `50000`)

Example:

```bash
W2V_DEVICE=cpu W2V_VOCAB_SIZE=15000 W2V_EPOCHS=1 W2V_BATCH_SIZE=4096 W2V_MAX_PAIRS=50000 python word2vec.py
```

### POS controls

- `POS_EPOCHS` (e.g., `1`)
- `EMBEDDING_VARIANTS` (e.g., `svd`, `skip-gram`, `glove`)

Example:

```bash
POS_EPOCHS=1 EMBEDDING_VARIANTS=glove python pos_tagger.py
```

## 5) Expected Output Files

| File | Description |
|------|-------------|
| `embeddings/svd.pt` | SVD word embeddings |
| `embeddings/svd_word2idx.json` | SVD word-index mapping |
| `embeddings/skipgram.pt` | Skip-Gram word embeddings |
| `embeddings/skipgram_word2idx.json` | Skip-Gram word-index mapping |
| `confusion_matrix_*.png` | POS confusion matrix image(s) |
| `report.pdf` | Final report |
| `2024114009_A2.zip` | Validated submission archive |