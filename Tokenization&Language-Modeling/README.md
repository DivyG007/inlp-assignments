# Assignment 1 - Tokenization & Language Modelling

**Name:** Divyanshu Giri  
**Roll Number:** 2024114009  
**Course:** Introduction to NLP – Spring ’26  

---

## Overview

This assignment implements manual tokenization (Whitespace, Regex, BPE) and n-gram language modelling (4-gram with MLE, Witten-Bell, and Kneser-Ney smoothing) for English and Mongolian texts.

## Files

- `tokenizers.py`: Contains implementations for data loading, cleaning, splitting, and the three tokenizers (Whitespace, Regex, BPE).
- `language_models.py`: Contains the 4-gram language model implementation, smoothing techniques, and evaluation scripts.
- `bpe_merges_en.json`: Learned BPE merge rules for English.
- `bpe_merges_mn.json`: Learned BPE merge rules for Mongolian.
- `report.pdf`: Detailed project report containing analysis, examples, and perplexity results.

## How to Run

### Prerequisities
- Python 3
- Standard libraries only (`re`, `json`, `math`, `collections`, `random`, `unicodedata`).

### 1. Train Tokenizers (Optional)
The BPE merge files are already included. However, if you wish to retrain the tokenizers and regenerate the splits:

```bash
python3 tokenizers.py
```

This will:
- Load the CC-100 datasets.
- Clean and split them (80/10/10).
- Train the BPE model (5000 merges).
- Save `bpe_merges_en.json` and `bpe_merges_mn.json`.

### 2. Run Language Models
To execute the language modeling pipeline, calculate perplexity, and see sentence generations:

```bash
python3 language_models.py
```

This will:
- Load the English corpus and the BPE merges.
- Build 4-gram models for all 3 tokenizers (Whitespace, Regex, BPE).
- Apply 3 smoothing techniques (None, Witten-Bell, Kneser-Ney).
- Report Perplexity on the test set.
- Generate sample sentence completions for qualitative analysis.

## Assumptions
- **Reproducibility:** `random.seed(42)` is used to ensure the train/val/test splits are identical across runs.
- **Vocab:** Frequency threshold of `min_freq=2` is used; tokens appearing only once are mapped to `<unk>`.
- **BPE:** `</w>` is used as a word-boundary suffix during training and stripped for final output generation.
- **dataset:** should be included to run the tokenizer and models as dataset/cc100_en.jsonl and dataset/cc100_mn.jsonl
