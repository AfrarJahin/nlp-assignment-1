"""Data loading and preprocessing for WMT translation dataset."""

import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=50):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]

        src_ids = self.src_vocab.encode(src, add_eos=True)[:self.max_len]
        tgt_ids = self.tgt_vocab.encode(tgt, add_bos=True, add_eos=True)[:self.max_len]

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_fn(batch, src_pad_idx, tgt_pad_idx):
    src_batch, tgt_batch = zip(*batch)

    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=tgt_pad_idx)

    # Encoder padding mask: True for real tokens, False for <pad>
    # Shape: (batch, src_len)
    src_mask = (src_padded != src_pad_idx)

    return src_padded, tgt_padded, src_mask


def load_data(src_file, tgt_file):
    """Load pre-BPE-tokenised files (one sentence per line, space-separated pieces)."""
    with open(src_file, encoding='utf-8') as f:
        src_lines = [line.strip().split() for line in f]
    with open(tgt_file, encoding='utf-8') as f:
        tgt_lines = [line.strip().split() for line in f]
    assert len(src_lines) == len(tgt_lines), \
        f'Source/target line count mismatch: {len(src_lines)} vs {len(tgt_lines)}'
    return src_lines, tgt_lines


def get_dataloader(src_sentences, tgt_sentences, src_vocab, tgt_vocab,
                   batch_size=32, shuffle=True, max_len=50):
    dataset = TranslationDataset(
        src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len)
    collate = partial(
        collate_fn, src_pad_idx=src_vocab.pad_idx, tgt_pad_idx=tgt_vocab.pad_idx)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
