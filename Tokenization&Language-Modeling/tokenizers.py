import re
import json
import random
import unicodedata
from collections import Counter
from collections import defaultdict

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Task 1.0
def corpus_selection(path):
    text = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj:
                text.append(obj["text"])

    return text

# Task 1.1

def normalizing_regex(line):

    line = unicodedata.normalize("NFKC", line) # unicode normalization
    line = re.sub(r'[\x00-\x1F\x7F]', '', line) # remove control characters    
    line = re.sub(r'’','\'',line) # smart quotes -> standard quotes
    line = re.sub(r'—','-',line)  # em dash -> standard dash
    line = re.sub(r'<br\s*/?>','',line) # to remove html tags
    line = re.sub(r'&amp;','&',line) # normalizing &amp
    # line = re.sub(r'@[A-Za-z0-9_]+','',line) #to remove x handle usernames
    line = re.sub(r'\s+',' ',line) # removes excessive spacing
    line = line.strip() # removes leading and trailing spaces
    line = re.sub(r'^[A-Za-z0-9_]:+\s*:\s*','',line) #to remove speaker's prompt in a dialogue

    return line

def corpus_cleaning(text):

    text = [normalizing_regex(t) for t in text]
    return text
    

def text_splitting(text):

    random.shuffle(text)

    n = len(text)

    train_len = int(TRAIN_RATIO*n)
    val_len = train_len + int(VAL_RATIO*n)

    train_text = text[:train_len]
    val_text = text[train_len:val_len]
    test_text = text[val_len:]

    return train_text, val_text, test_text


# Task 1.2

# whitespace-based tokenizer
def whitespace_tokenizer(text):
    text = re.sub(r'([^\w\s])', r' \1 ', text) # separate punctuation
    return text.split()

# regex based tokenizer for english
def regex_tokenizer_en(text):
    text =  re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]", text)
    return text

# bpe based tokenizer for english
def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def train_bpe(corpus, num_merges):
    vocab = defaultdict(int)
    # Pre-tokenize simply by whitespace to preserve word boundaries for BPE
    # Adding </w> at the end of each word to handle word boundaries
    for word in whitespace_tokenizer(corpus):
        vocab[' '.join(list(word)) + ' </w>'] += 1

    merges = []
    print(f"Starting BPE training ({num_merges} merges)...")
    for i in range(num_merges):
        if (i + 1) % 50 == 0:
            print(f"  > Progress: {i + 1}/{num_merges} merges...")
            
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    
    return merges

def apply_bpe(text, merges, cache=None):
    tokens = []
    if cache is None:
        cache = {}
    for word in whitespace_tokenizer(text):
        if word in cache:
            tokens.extend(cache[word])
            continue
        word_split = " ".join(list(word)) + " </w>"
        for pair in merges:
            bigram = " ".join(pair)
            replacement = "".join(pair)
            word_split = word_split.replace(bigram, replacement)
        word_tokens = [tok for tok in word_split.split() if tok != "</w>"]
        cache[word] = word_tokens
        tokens.extend(word_tokens)
    return tokens

def save_bpe_merges(merges, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([list(pair) for pair in merges], f, ensure_ascii=False)

def load_bpe_merges(path):
    with open(path, "r", encoding="utf-8") as f:
        return [tuple(pair) for pair in json.load(f)]

# regex based tokenizer for mongolian
def regex_tokenizer_mn(text):
    text =  re.findall(r"[а-яА-ЯёЁөӨүҮ]+|\d+(?:[.,]\d+)*|[^\w\s]", text)
    return text

def main():

    random.seed(42)  # reproducibility

    path_en = "dataset/cc100_en.jsonl"
    path_mn = "dataset/cc100_mn.jsonl"

    text_en = corpus_selection(path_en)
    text_mn = corpus_selection(path_mn)

    # Task 1.1 
    text_en = corpus_cleaning(text_en)
    text_mn = corpus_cleaning(text_mn)

    # Remove empty lines after cleaning
    text_en = [t for t in text_en if t]
    text_mn = [t for t in text_mn if t]

    # text splitting
    train_en, val_en, test_en = text_splitting(text_en)
    train_mn, val_mn, test_mn = text_splitting(text_mn)

    print("English split sizes:", len(train_en), len(val_en), len(test_en))
    print("Mongolian split sizes:", len(train_mn), len(val_mn), len(test_mn))

    # Task 1.2 — Train tokenizers

    # whitespace tokenizer
    ws_tokens_en = [whitespace_tokenizer(t) for t in train_en]
    ws_tokens_mn = [whitespace_tokenizer(t) for t in train_mn]

    # regex-based tokenizer
    regex_tokens_en = [regex_tokenizer_en(t) for t in train_en]
    regex_tokens_mn = [regex_tokenizer_mn(t) for t in train_mn]

    # BPE tokenizer
    # Train BPE ONLY on training data
    train_en_concat = " ".join(train_en)
    train_mn_concat = " ".join(train_mn)

    bpe_merges_en = train_bpe(train_en_concat, num_merges=5000)
    bpe_merges_mn = train_bpe(train_mn_concat, num_merges=5000)

    print("BPE (EN) merges learned:", len(bpe_merges_en))
    print("BPE (MN) merges learned:", len(bpe_merges_mn))

    # Save merges for later use in language modeling
    save_bpe_merges(bpe_merges_en, "bpe_merges_en.json")
    save_bpe_merges(bpe_merges_mn, "bpe_merges_mn.json")
    print("Saved BPE merges to bpe_merges_en.json and bpe_merges_mn.json")

    # Note:
    # - merge rules define the BPE tokenizer
    # - application of BPE to text is a separate step (not required yet)


if __name__ == "__main__":
    main()



