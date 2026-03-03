import json
import math
import random
from collections import Counter

from tokenizers import (
    corpus_selection,
    corpus_cleaning,
    text_splitting,
    whitespace_tokenizer,
    regex_tokenizer_en,
    apply_bpe,
    load_bpe_merges,
)

SPECIAL_TOKENS = ["<s>", "</s>", "<unk>"]


def tokenize_line(line, tokenizer, merges=None, bpe_cache=None):
    if tokenizer == "whitespace":
        return whitespace_tokenizer(line)
    if tokenizer == "regex":
        return regex_tokenizer_en(line)
    if tokenizer == "bpe":
        if merges is None:
            raise ValueError("BPE merges are required for tokenizer='bpe'.")
        return apply_bpe(line, merges, cache=bpe_cache)
    raise ValueError(f"Unknown tokenizer: {tokenizer}")


def build_vocab(tokenized_corpus, min_freq=1):
    counter = Counter(token for sent in tokenized_corpus for token in sent)
    vocab = {tok: freq for tok, freq in counter.items() if freq >= min_freq}
    for tok in SPECIAL_TOKENS:
        vocab.setdefault(tok, 0)
    token2id = {tok: i for i, tok in enumerate(vocab.keys())}
    id2token = {i: tok for tok, i in token2id.items()}
    return vocab, token2id, id2token

def add_sentence_markers(tokens, n=4):
    start = ["<s>"] * (n - 1)
    return start + tokens + ["</s>"]


def clean_bpe_token(token):
    return token.replace("</w>", "") if token else token


def clean_bpe_suggestions(suggestions):
    return [(clean_bpe_token(tok), prob) for tok, prob in suggestions]


def detokenize_bpe(tokens):
    words = []
    current = ""
    for tok in tokens:
        if tok in {"<s>", "</s>"}:
            continue
        if tok == "<unk>":
            if current:
                words.append(current)
                current = ""
            words.append(tok)
            continue
        if tok.endswith("</w>"):
            current += tok.replace("</w>", "")
            words.append(current)
            current = ""
        else:
            current += tok
    if current:
        words.append(current)
    return " ".join(words)


def next_token_candidates_from_context(context, vocab, trigram_counts, fourgram_counts, exclude_tokens=None):
    denom = trigram_counts.get(context, 0)
    exclude = set(exclude_tokens or [])
    vocab_prob = {}
    for word in vocab:
        if word in exclude:
            continue
        test_fourgram = (context[0], context[1], context[2], word)
        test_fourgram_count = fourgram_counts.get(test_fourgram, 0)
        prob = (test_fourgram_count / denom) if denom > 0 else 0.0
        vocab_prob[word] = prob
    return sorted(vocab_prob.items(), key=lambda x: x[1], reverse=True)[:4]


def build_ngram_counts(tokenized_corpus, n, label=None, progress_every=50000):
    counts = Counter()
    processed = 0
    for sent in tokenized_corpus:
        if len(sent) < n:
            continue
        for i in range(len(sent) - n + 1):
            counts[tuple(sent[i:i + n])] += 1
            processed += 1
            if progress_every and processed % progress_every == 0:
                tag = f"{label}:" if label else ""
                print(f"  [ngram{n}{tag}] processed {processed} n-grams")
    return counts


def build_context_stats(ngram_counts, n):
    context_counts = Counter()
    continuation_types = Counter()
    seen = {}
    for ngram, count in ngram_counts[n].items():
        context = ngram[:-1]
        context_counts[context] += count
        if context not in seen:
            seen[context] = set()
        seen[context].add(ngram[-1])
    for context, cont_set in seen.items():
        continuation_types[context] = len(cont_set)
    return context_counts, continuation_types

def build_continuation_unigram(bigram_counts):
    cont_unigram = Counter()
    seen = {}
    for (w1, w2) in bigram_counts.keys():
        if w2 not in seen:
            seen[w2] = set()
        seen[w2].add(w1)
    for w2, preds in seen.items():
        cont_unigram[w2] = len(preds)
    return cont_unigram

def encode_corpus(tokenized_corpus, token2id):
    unk_id = token2id.get("<unk>")
    return [[token2id.get(tok, unk_id) for tok in sent] for sent in tokenized_corpus]


def replace_oov(tokenized_corpus, vocab):
    vocab_set = set(vocab.keys())
    return [[tok if tok in vocab_set else "<unk>" for tok in sent] for sent in tokenized_corpus]


def mle_prob(word, context, trigram_counts, fourgram_counts):
    denom = trigram_counts.get(context, 0)
    if denom == 0:
        return 0.0
    return fourgram_counts.get(context + (word,), 0) / denom


def wb_prob(word, context, ngram_counts, context_counts, continuation_types):
    if len(context) == 0:
        total = sum(ngram_counts[1].values())
        return (ngram_counts[1].get((word,), 0) / total) if total > 0 else 0.0

    n = len(context) + 1
    c_context = context_counts[n].get(context, 0)
    t_context = continuation_types[n].get(context, 0)
    if c_context == 0:
        return wb_prob(word, context[1:], ngram_counts, context_counts, continuation_types)

    c_ngram = ngram_counts[n].get(context + (word,), 0)
    if c_ngram > 0:
        return c_ngram / (c_context + t_context)
    backoff = t_context / (c_context + t_context)
    return backoff * wb_prob(word, context[1:], ngram_counts, context_counts, continuation_types)


def kn_prob(word, context, ngram_counts, context_counts, continuation_types, cont_unigram, total_continuations, discount=0.75):
    if len(context) == 0:
        return (cont_unigram.get(word, 0) / total_continuations) if total_continuations > 0 else 0.0

    n = len(context) + 1
    c_context = context_counts[n].get(context, 0)
    if c_context == 0:
        return kn_prob(word, context[1:], ngram_counts, context_counts, continuation_types, cont_unigram, total_continuations, discount)

    c_ngram = ngram_counts[n].get(context + (word,), 0)
    t_context = continuation_types[n].get(context, 0)
    lambda_c = (discount * t_context) / c_context

    lower_prob = kn_prob(word, context[1:], ngram_counts, context_counts, continuation_types, cont_unigram, total_continuations, discount)
    return max(c_ngram - discount, 0) / c_context + lambda_c * lower_prob


def perplexity(tokenized_corpus, ngram_counts, context_counts, continuation_types, cont_unigram, total_continuations, smoothing="none"):
    total_log = 0.0
    total_tokens = 0
    progress_every = 50000
    processed = 0

    for sent in tokenized_corpus:
        if len(sent) < 4:
            continue
        for i in range(3, len(sent)):
            context = tuple(sent[i - 3:i])
            word = sent[i]
            if smoothing == "none":
                prob = mle_prob(word, context, ngram_counts[3], ngram_counts[4])
            elif smoothing == "witten_bell":
                prob = wb_prob(word, context, ngram_counts, context_counts, continuation_types)
            elif smoothing == "kneser_ney":
                prob = kn_prob(word, context, ngram_counts, context_counts, continuation_types, cont_unigram, total_continuations)
            else:
                raise ValueError(f"Unknown smoothing: {smoothing}")

            if prob <= 0:
                return float("inf")
            total_log += math.log(prob)
            total_tokens += 1
            processed += 1
            if processed % progress_every == 0:
                print(f"  [perplexity:{smoothing}] processed {processed} tokens")

    if total_tokens == 0:
        return float("inf")
    return math.exp(-total_log / total_tokens)

def four_gram_model(input_text, tokenizer, merges, vocab, trigram_counts, fourgram_counts, bpe_cache=None):
    tokenized_input = tokenize_line(input_text.lower(), tokenizer, merges, bpe_cache)
    vocab_set = set(vocab.keys())
    tokenized_input = [tok if tok in vocab_set else "<unk>" for tok in tokenized_input]
    context = ("<s>", "<s>", "<s>")
    if len(tokenized_input) >= 3:
        context = tuple(tokenized_input[-3:])
    elif len(tokenized_input) == 2:
        context = ("<s>", tokenized_input[0], tokenized_input[1])
    elif len(tokenized_input) == 1:
        context = ("<s>", "<s>", tokenized_input[0])

    denom = trigram_counts.get(context, 0)
    vocab_prob = {}
    for word in vocab:
        test_fourgram = (context[0], context[1], context[2], word)
        test_fourgram_count = fourgram_counts.get(test_fourgram, 0)
        prob = (test_fourgram_count / denom) if denom > 0 else 0.0
        vocab_prob[word] = prob

    top_order = sorted(vocab_prob.items(), key=lambda x: x[1], reverse=True)[:4]
    return top_order


def interactive_loop(tokenizer, merges, vocab, trigram_counts, fourgram_counts, bpe_cache=None):
    print("\nInteractive mode (type 'quit' to exit)")
    print("Constraints:")
    print("- English input only (same as training).")
    print("- Do not include <s>, </s>, or <unk> tokens.")
    print("- Provide at least one word; the model uses the last 3 tokens as context.")

    while True:
        user_text = input("Input text: ").strip()
        if not user_text:
            print("Please enter at least one word or type 'quit'.")
            continue
        if user_text.lower() == "quit":
            break
        suggestions = four_gram_model(user_text, tokenizer, merges, vocab, trigram_counts, fourgram_counts, bpe_cache)
        if tokenizer == "bpe":
            suggestions = clean_bpe_suggestions(suggestions)
        print("Top next-token suggestions:", suggestions)


def generate_sentence(prefix_text, tokenizer, merges, vocab, trigram_counts, fourgram_counts, max_len=30, bpe_cache=None):
    tokens = tokenize_line(prefix_text.lower(), tokenizer, merges, bpe_cache)
    vocab_set = set(vocab.keys())
    tokens = [tok if tok in vocab_set else "<unk>" for tok in tokens]

    if len(tokens) == 0:
        tokens = ["<s>"]

    generated = tokens[:]
    while len(generated) < max_len:
        if len(generated) >= 3:
            context = tuple(generated[-3:])
        elif len(generated) == 2:
            context = ("<s>", generated[0], generated[1])
        else:
            context = ("<s>", "<s>", generated[0])

        next_candidates = next_token_candidates_from_context(
            context,
            vocab,
            trigram_counts,
            fourgram_counts,
            exclude_tokens={"<s>"},
        )
        next_token = next_candidates[0][0] if next_candidates else "</s>"
        generated.append(next_token)
        if next_token == "</s>":
            break

    if tokenizer == "bpe":
        return detokenize_bpe(generated)
    output_tokens = [t for t in generated if t not in {"<s>", "</s>"}]
    return " ".join(output_tokens)


def main():
    random.seed(42)  # must match tokenizers.py for identical splits

    # English corpus path
    path_corpus_en = "dataset/cc100_en.jsonl"
    merges_path_en = "bpe_merges_en.json"

    # 1) Load + clean + split
    text_en = corpus_selection(path_corpus_en)
    text_en = corpus_cleaning(text_en)
    text_en = [t for t in text_en if t]
    train_en, val_en, test_en = text_splitting(text_en)

    # 2) Run for each tokenizer: "whitespace" | "regex" | "bpe"
    # Assignment requires all 3 for 9 total models (3 tokenizers x 3 smoothings)
    for tokenizer in ("whitespace", "regex", "bpe"):
        merges = load_bpe_merges(merges_path_en) if tokenizer == "bpe" else None
        bpe_cache = {} if tokenizer == "bpe" else None

        # 3) Tokenize corpora with start/end markers (for 4-gram)
        progress_every = 50000
        train_tokens = []
        for i, t in enumerate(train_en, 1):
            train_tokens.append(add_sentence_markers(tokenize_line(t, tokenizer, merges, bpe_cache), n=4))
            if i % progress_every == 0:
                print(f"  [tokenize:{tokenizer}] train {i}/{len(train_en)}")
        val_tokens = []
        for i, t in enumerate(val_en, 1):
            val_tokens.append(add_sentence_markers(tokenize_line(t, tokenizer, merges, bpe_cache), n=4))
            if i % progress_every == 0:
                print(f"  [tokenize:{tokenizer}] val {i}/{len(val_en)}")
        test_tokens = []
        for i, t in enumerate(test_en, 1):
            test_tokens.append(add_sentence_markers(tokenize_line(t, tokenizer, merges, bpe_cache), n=4))
            if i % progress_every == 0:
                print(f"  [tokenize:{tokenizer}] test {i}/{len(test_en)}")

        # 4) Build vocabulary + encode train corpus
        vocab, token2id, id2token = build_vocab(train_tokens, min_freq=2)
        train_tokens = replace_oov(train_tokens, vocab)
        val_tokens = replace_oov(val_tokens, vocab)
        test_tokens = replace_oov(test_tokens, vocab)
        train_ids = encode_corpus(train_tokens, token2id)

        print("\nTokenizer:", tokenizer)
        print("Vocab size:", len(vocab))
        print("Sample encoded sentence:", train_ids[0][:20])

        # 5) Build n-gram counts for the probabilistic model
        ngram_counts = {
            1: build_ngram_counts(train_tokens, n=1, label=tokenizer),
            2: build_ngram_counts(train_tokens, n=2, label=tokenizer),
            3: build_ngram_counts(train_tokens, n=3, label=tokenizer),
            4: build_ngram_counts(train_tokens, n=4, label=tokenizer),
        }
        context_counts = {}
        continuation_types = {}
        for n in (2, 3, 4):
            context_counts[n], continuation_types[n] = build_context_stats(ngram_counts, n)

        cont_unigram = build_continuation_unigram(ngram_counts[2])
        total_continuations = sum(cont_unigram.values())

        # Example usage: predict next token candidates
        sample_suggestions = four_gram_model("the home has", tokenizer, merges, vocab, ngram_counts[3], ngram_counts[4], bpe_cache)
        if tokenizer == "bpe":
            sample_suggestions = clean_bpe_suggestions(sample_suggestions)
        print(sample_suggestions)

        # 6) Perplexity on test set for Task 2.3
        for smoothing in ("none", "witten_bell", "kneser_ney"):
            ppl = perplexity(test_tokens, ngram_counts, context_counts, continuation_types, cont_unigram, total_continuations, smoothing=smoothing)
            print(f"Perplexity ({smoothing}):", ppl)

        # 7) Batch sentence completions for report
        test_prefixes = [
            "the home has",
            "I am going to",
            "black",
            "where the",
            "it is working",
            "United States",
            "she said that",
            "how to make",
            "death",
            "the best way to",
        ]
        print(f"\n--- Sentence Completions ({tokenizer}) ---")
        for prefix in test_prefixes:
            completed = generate_sentence(prefix, tokenizer, merges, vocab, ngram_counts[3], ngram_counts[4], bpe_cache=bpe_cache)
            print(f'  "{prefix}" → {completed}')

        # 8) Next-token predictions for report
        test_inputs = [
            "the home has",
            "I want to",
            "she said",
            "it is a",
            "where",
            "how to",
        ]
        print(f"\n--- Next-Token Predictions ({tokenizer}) ---")
        for inp in test_inputs:
            suggestions = four_gram_model(inp, tokenizer, merges, vocab, ngram_counts[3], ngram_counts[4], bpe_cache)
            if tokenizer == "bpe":
                suggestions = clean_bpe_suggestions(suggestions)
            print(f'  "{inp}" → {suggestions}')

        print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()