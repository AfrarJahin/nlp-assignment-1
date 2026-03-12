"""Vocabulary backed by a trained SentencePiece BPE model."""

import sentencepiece as spm


class Vocabulary:
    PAD = '<pad>'
    UNK = '<unk>'
    BOS = '<bos>'
    EOS = '<eos>'

    # These IDs match the --pad_id/bos_id/eos_id/unk_id flags used during
    # SentencePiece training in prepare_data.py.
    pad_idx = 0
    unk_idx = 1
    bos_idx = 2
    eos_idx = 3

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def __len__(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, tokens: list[str], add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Convert a list of BPE piece strings to integer IDs."""
        ids = [self.sp.piece_to_id(t) for t in tokens]
        if add_bos:
            ids = [self.bos_idx] + ids
        if add_eos:
            ids = ids + [self.eos_idx]
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> list[str]:
        """Convert IDs back to a list of word strings (via sentencepiece detokenisation)."""
        specials = {self.pad_idx, self.bos_idx, self.eos_idx}
        filtered = []
        for idx in ids:
            if idx == self.eos_idx:
                break
            if strip_special and idx in specials:
                continue
            filtered.append(idx)
        text = self.sp.decode(filtered)
        return text.split()
