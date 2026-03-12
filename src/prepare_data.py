"""
Dataset preparation for WMT14 (toy subset).

Steps:
  1. Download WMT14 de-en via HuggingFace datasets
  2. NLTK-tokenize both sides
  3. Filter pairs where either side > max_len tokens
  4. Randomly sample 10k train / 1k val / 1k test
  5. Train separate BPE (sentencepiece) models on training data
  6. Encode all splits and write to data/
  7. Report statistics

Usage:
  python -m src.prepare_data
  python -m src.prepare_data --src_lang de --tgt_lang en --vocab_size 8000
"""

import argparse
import os
import random

import nltk
import sentencepiece as spm
from datasets import load_dataset


def fast_split(text: str) -> list[str]:
    """Whitespace split used only for fast length filtering."""
    return text.lower().split()


def nltk_tokenize(text: str) -> list[str]:
    """Full NLTK tokenization for sampled pairs only."""
    return nltk.word_tokenize(text.lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", default="de")
    parser.add_argument("--tgt_lang", default="en")
    parser.add_argument("--train_size", type=int, default=10_000)
    parser.add_argument("--val_size", type=int, default=1_000)
    parser.add_argument("--test_size", type=int, default=1_000)
    parser.add_argument(
        "--max_len",
        type=int,
        default=50,
        help="Filter sentences longer than this many tokens",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=16_000,
        help="BPE vocabulary size (8000-16000 recommended)",
    )
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    # 1. Load raw WMT14
    print(f"Loading WMT14 {args.src_lang}-{args.tgt_lang} ...")
    dataset = load_dataset(
        "wmt14",
        f"{args.src_lang}-{args.tgt_lang}",
        split="train",
        trust_remote_code=True,
    )

    # 2. Fast filter over full dataset using whitespace split
    print(f"Filtering pairs with >{args.max_len} tokens (fast whitespace split) ...")
    raw_pairs: list[tuple[str, str]] = []
    for item in dataset:
        trans = item["translation"]
        src_raw = trans[args.src_lang]
        tgt_raw = trans[args.tgt_lang]
        if len(fast_split(src_raw)) <= args.max_len and len(fast_split(tgt_raw)) <= args.max_len:
            raw_pairs.append((src_raw, tgt_raw))

    print(f"Pairs after filtering (>{args.max_len} tokens removed): {len(raw_pairs):,}")

    # 3. Sample splits
    total = args.train_size + args.val_size + args.test_size
    if len(raw_pairs) < total:
        raise ValueError(
            f"Not enough sentence pairs after filtering: {len(raw_pairs):,} < {total:,}"
        )

    random.shuffle(raw_pairs)
    sampled_train = raw_pairs[: args.train_size]
    sampled_val = raw_pairs[args.train_size : args.train_size + args.val_size]
    sampled_test = raw_pairs[args.train_size + args.val_size : total]

    # 4. NLTK tokenize sampled pairs only
    print(f"NLTK-tokenizing {total:,} sampled pairs ...")
    train_pairs = [(nltk_tokenize(s), nltk_tokenize(t)) for s, t in sampled_train]
    val_pairs = [(nltk_tokenize(s), nltk_tokenize(t)) for s, t in sampled_val]
    test_pairs = [(nltk_tokenize(s), nltk_tokenize(t)) for s, t in sampled_test]

    # 5. Train BPE models
    raw_src = os.path.join(args.out_dir, "_raw_train.src")
    raw_tgt = os.path.join(args.out_dir, "_raw_train.tgt")
    with open(raw_src, "w", encoding="utf-8") as f:
        for src, _ in train_pairs:
            f.write(" ".join(src) + "\n")
    with open(raw_tgt, "w", encoding="utf-8") as f:
        for _, tgt in train_pairs:
            f.write(" ".join(tgt) + "\n")

    print(f"Training BPE models (vocab_size={args.vocab_size}) ...")
    for raw_file, prefix in [
        (raw_src, os.path.join(args.out_dir, "spm.src")),
        (raw_tgt, os.path.join(args.out_dir, "spm.tgt")),
    ]:
        spm.SentencePieceTrainer.train(
            input=raw_file,
            model_prefix=prefix,
            vocab_size=args.vocab_size,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            character_coverage=0.9995,
            model_type="bpe",
        )

    src_sp = spm.SentencePieceProcessor(model_file=os.path.join(args.out_dir, "spm.src.model"))
    tgt_sp = spm.SentencePieceProcessor(model_file=os.path.join(args.out_dir, "spm.tgt.model"))

    # 6. Encode and write splits
    for split, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        src_out = os.path.join(args.out_dir, f"{split}.src")
        tgt_out = os.path.join(args.out_dir, f"{split}.tgt")
        with open(src_out, "w", encoding="utf-8") as fs, open(tgt_out, "w", encoding="utf-8") as ft:
            for src, tgt in split_pairs:
                src_bpe = src_sp.encode(" ".join(src), out_type=str)
                tgt_bpe = tgt_sp.encode(" ".join(tgt), out_type=str)
                fs.write(" ".join(src_bpe) + "\n")
                ft.write(" ".join(tgt_bpe) + "\n")
        print(f"  Wrote {len(split_pairs):,} pairs -> {src_out}, {tgt_out}")

    os.remove(raw_src)
    os.remove(raw_tgt)

    # 7. Report statistics
    with open(os.path.join(args.out_dir, "train.src"), encoding="utf-8") as f:
        bpe_src_lens = [len(line.strip().split()) for line in f]
    with open(os.path.join(args.out_dir, "train.tgt"), encoding="utf-8") as f:
        bpe_tgt_lens = [len(line.strip().split()) for line in f]

    avg_src_word = sum(len(s) for s, _ in train_pairs) / len(train_pairs)
    avg_tgt_word = sum(len(t) for _, t in train_pairs) / len(train_pairs)
    avg_src_bpe = sum(bpe_src_lens) / len(bpe_src_lens)
    avg_tgt_bpe = sum(bpe_tgt_lens) / len(bpe_tgt_lens)

    print("\n-- Dataset Statistics ------------------------------")
    print(f"  Pairs after filtering (>{args.max_len} word tokens) : {len(raw_pairs):,}")
    print(f"  Train pairs : {len(train_pairs):,}")
    print(f"  Val   pairs : {len(val_pairs):,}")
    print(f"  Test  pairs : {len(test_pairs):,}")
    print(f"  Avg src length : {avg_src_word:.1f} words  /  {avg_src_bpe:.1f} BPE tokens")
    print(f"  Avg tgt length : {avg_tgt_word:.1f} words  /  {avg_tgt_bpe:.1f} BPE tokens")
    print(f"  Src vocab size : {src_sp.get_piece_size():,} subword pieces")
    print(f"  Tgt vocab size : {tgt_sp.get_piece_size():,} subword pieces")
    print("----------------------------------------------------")
    print(f"\nDone. BPE models saved to {args.out_dir}/spm.{{src,tgt}}.model")

    report_path = os.path.join(args.out_dir, "prep_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Dataset Preparation Report\n")
        f.write("==========================\n")
        f.write(f"Number of sentence pairs after filtering: {len(raw_pairs):,}\n")
        f.write(
            "Average sentence length (train split): "
            f"src={avg_src_word:.1f} words ({avg_src_bpe:.1f} BPE), "
            f"tgt={avg_tgt_word:.1f} words ({avg_tgt_bpe:.1f} BPE)\n"
        )
        f.write(
            "Vocabulary size: "
            f"src={src_sp.get_piece_size():,}, tgt={tgt_sp.get_piece_size():,}\n"
        )
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
