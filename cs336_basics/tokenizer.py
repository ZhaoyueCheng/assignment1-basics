"""Starter interface for the byte-level BPE tokenizer assignment."""

from collections.abc import Iterable, Iterator
from pathlib import Path

import regex


GPT2_PAT = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def convert_to_byte(word):
    utf_8 = list(word.encode("utf-8"))
    return tuple(bytes([i]) for i in utf_8)


def split_chunks(text, special_tokens):
    if not special_tokens:
        return [text]

    # Sort by descending length to prioritize longer tokens
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    special_pattern = "|".join(regex.escape(t) for t in special_tokens)

    # Keep special tokens in the split
    special_pattern = regex.compile(f"({special_pattern})")
    text_chunks = special_pattern.split(text)

    return [chunk for chunk in text_chunks if chunk != ""]


class Tokenizer:
    """Encode and decode text with a byte-level BPE vocabulary."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """Construct a tokenizer from a vocabulary, merges, and special tokens."""
        self.vocab = dict(vocab)
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.merges = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = special_tokens or []

        # Add special tokens that are not already in the vocabulary.
        next_token_id = max(self.vocab, default=-1) + 1
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab_inv:
                self.vocab[next_token_id] = token_bytes
                self.vocab_inv[token_bytes] = next_token_id
                next_token_id += 1

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Construct a tokenizer from serialized vocabulary and merge files."""
        raise NotImplementedError

    def apply_merge(self, word_bytes):
        word_bytes = list(word_bytes)
        while True:
            merge_ind = float("inf")
            word_index = -1

            for i in range(len(word_bytes) - 1):
                pair = (word_bytes[i], word_bytes[i + 1])

                if pair in self.merges and self.merges[pair] < merge_ind:
                    merge_ind = self.merges[pair]
                    word_index = i

            if merge_ind == float("inf"):
                break

            word_bytes = (
                word_bytes[:word_index]
                + [word_bytes[word_index] + word_bytes[word_index + 1]]
                + word_bytes[word_index + 2 :]
            )

        return word_bytes

    def tokenize(self, chunk):
        res = []
        if self.special_tokens and chunk in self.special_tokens:
            return [self.vocab_inv[chunk.encode("utf-8")]]
        for m in regex.finditer(GPT2_PAT, chunk):
            word = m.group(0)
            word_bytes = convert_to_byte(word)
            word_merged = self.apply_merge(word_bytes)
            word_ids = [self.vocab_inv[x] for x in word_merged]
            res.extend(word_ids)

        return res

    def encode(self, text: str) -> list[int]:
        """Encode an input string into token ids."""
        chunks = split_chunks(text, self.special_tokens)
        results = map(self.tokenize, chunks)
        return [token_id for chunk_ids in results for token_id in chunk_ids]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode token ids into text."""
        return b"".join([self.vocab[t] for t in ids]).decode("utf-8", errors="replace")
