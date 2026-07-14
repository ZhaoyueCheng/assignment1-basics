"""
Byte-level BPE tokenizer training.
"""

import regex
from collections import defaultdict
from tqdm.contrib.concurrent import process_map

# GPT-2 pre-tokenization regex pattern (pre-compiled)
GPT2_PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def read_input(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def split_chunks(text, special_tokens):
    if not special_tokens:
        return [text]

    # Sort by descending length to prioritize longer tokens
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    special_pattern = "|".join(regex.escape(t) for t in special_tokens)
    special_pattern = regex.compile(special_pattern)
    text_chunks = special_pattern.split(text)

    return [chunk for chunk in text_chunks if chunk != ""]

def convert_to_byte(word):
    utf_8 = list(word.encode("utf-8"))
    return tuple(bytes([i]) for i in utf_8)

def count_word(text):
    word_dict, pair_dict = defaultdict(int), defaultdict(int)
    
    for m in regex.finditer(GPT2_PAT, text):
        word = m.group(0)
        word_bytes = convert_to_byte(word)
        if len(word_bytes) >= 2:
            # update word dict
            word_dict[word_bytes] += 1

            # update pair dict
            for b1, b2 in zip(word_bytes[:-1], word_bytes[1:]):
                pair_dict[(b1,b2)] += 1
    
    return word_dict, pair_dict

def get_vocab(special_tokens):
    vocab = {i:bytes([i]) for i in range(256)}
    for i,special_token in enumerate(special_tokens):
        vocab[i+256] = special_token.encode("utf-8")
    return vocab

def get_max_pair(pair_dicts):
    pair_counts = [(count, pair) for pair, count in pair_dicts.items()]
    pair_counts.sort()
    return pair_counts[-1]

def merge_dict(dicts):
    merged_d = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            merged_d[k] += v
    return merged_d

def apply_merge(max_pair, word):
    merged_word = max_pair[0] + max_pair[1]
    i = 0
    new_word_bytes = []

    while i < len(word):
        if i < len(word)-1 and (word[i], word[i+1]) == max_pair:
            new_word_bytes.append(merged_word)
            i += 2
        else:
            new_word_bytes.append(word[i])
            i += 1
        
    return new_word_bytes

def update_cnt(max_pair, word_dict, pair_dict):
    new_pair_dict = defaultdict(int, pair_dict)
    new_word_dict = defaultdict(int)

    for word, count in word_dict.items():
        old_pairs = list([(x,y) for x,y in zip(word[:-1], word[1:])])
        old_pairs_set = set(old_pairs)

        if max_pair not in old_pairs_set:
            new_word_dict[word] += count
            continue

        new_word_bytes = apply_merge(max_pair, word)
        new_word_dict[tuple(new_word_bytes)] += count

        for pair in old_pairs:
            new_pair_dict[pair] -= count
            if new_pair_dict[pair] == 0:
                new_pair_dict.pop(pair)
        
        for b1, b2 in zip(new_word_bytes[:-1], new_word_bytes[1:]):
            new_pair_dict[(b1,b2)] += count

    return new_word_dict, new_pair_dict

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练一个 byte-level BPE tokenizer。

    Args:
        input_path: 训练语料文件路径
        vocab_size: 最终词表大小 (包含 byte vocab + special tokens + merge 产生的 token)
        special_tokens: 特殊 token 列表

    Returns:
        (vocab, merges)
        - vocab: {token_id: token_bytes} 映射
        - merges: [(token1_bytes, token2_bytes), ...] 按创建顺序排列
    """

    text = read_input(input_path)
    chunks = split_chunks(text, special_tokens)

    if len(chunks) < 10:
        results = list(map(count_word, chunks))
    else:
        results = process_map(count_word, chunks, chunksize=1)
    word_dicts, pair_dicts = zip(*results) if results else ([], [])

    word_dict, pair_dict = merge_dict(word_dicts), merge_dict(pair_dicts)

    vocab = get_vocab(special_tokens)
    n_merges = vocab_size - len(vocab)

    merges = []
    for i in range(n_merges):
        _, max_pair = get_max_pair(pair_dict)
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        word_dict, pair_dict = update_cnt(max_pair, word_dict, pair_dict)
    
    return vocab, merges
