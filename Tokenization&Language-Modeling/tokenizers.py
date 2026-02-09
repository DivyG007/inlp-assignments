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
    return text.split()

# regex based tokenizer for english
def regex_tokenizer_en(text):
    text =  re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]", text)
    return text

# bpe based tokenizer for english
def merge(token_ids, target_pair, merged_token_id):
    """
    Replace all non-overlapping occurrences of target_pair
    with merged_token_id in the token sequence.
    """
    merged_sequence = []
    index = 0

    while index < len(token_ids):
        # Check if the current and next token form the target pair
        if (
            index < len(token_ids) - 1 and
            token_ids[index] == target_pair[0] and
            token_ids[index + 1] == target_pair[1]
        ):
            merged_sequence.append(merged_token_id)
            index += 2  # Skip both tokens in the merged pair
        else:
            merged_sequence.append(token_ids[index])
            index += 1

    # the bpe returns merg-rules and not a merged-corpus as these rules will help to tokenize any given new data rather than a merged-corpus
    return merged_sequence

# stops after num_merges iterations
def byte_pair_encoding(corpus, num_merges):
    """
    BYTE-PAIR ENCODING(C, k) without explicit vocab V.
    Vocabulary is implicit in the token sequence and merge rules.
    """

    # initial tokens
    token_seq = list(corpus)

    merge_rules = {}

    next_token_id = 256 

    for i in range(num_merges):
        pair_counts = Counter(zip(token_seq, token_seq[1:]))
        if not pair_counts:
            break
        most_freq_pair = max(pair_counts, key=pair_counts.get)
        new_token = next_token_id
        token_seq = merge(token_seq, most_freq_pair, new_token)
        merge_rules[most_freq_pair] = new_token
        next_token_id += 1

    return merge_rules

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

    bpe_merges_en = byte_pair_encoding(train_en_concat, num_merges=500)
    bpe_merges_mn = byte_pair_encoding(train_mn_concat, num_merges=500)

    print("BPE (EN) merges learned:", len(bpe_merges_en))
    print("BPE (MN) merges learned:", len(bpe_merges_mn))

    # Note:
    # - merge rules define the BPE tokenizer
    # - application of BPE to text is a separate step (not required yet)


if __name__ == "__main__":
    main()



