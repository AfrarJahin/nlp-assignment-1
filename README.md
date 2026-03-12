# NLP Assignment 1 — Neural Machine Translation

CSCI-7150-A | Spring 2026 | Problem 2 (70 pts)

Sequence-to-sequence NMT system with a Bidirectional LSTM encoder, Unidirectional LSTM decoder, and Multiplicative (Luong-style) attention, trained on a WMT14 de→en toy subset.

---

## Architecture

| Component | Details |
|---|---|
| **Encoder** | Bidirectional LSTM — outputs `h_i^enc = [←h_i; →h_i] ∈ R^{2h}` |
| **Decoder init** | `h_0^dec = W_h[←h_1^enc; →h_m^enc]`, `c_0^dec = W_c[←c_1^enc; →c_m^enc]` |
| **Decoder input** | `ȳ_t = [y_t; o_{t-1}] ∈ R^{e+h}` (embedding + previous output) |
| **Attention** | Multiplicative: `e_{t,i} = (h_t^dec)ᵀ W_attProj h_i^enc` |
| **Output** | `u_t = [a_t; h_t^dec] → W_u → tanh → dropout → W_vocab → softmax` |
| **Loss** | Cross-entropy NLL, averaged over batch (eqs 36–38) |
| **Optimiser** | Adam + global gradient norm clipping τ (eq 39) |
| **Decoding** | Greedy + Beam search (B=3), score = Σ log p (eq 40) |

---

## Setup

```bash
conda env create -f environment.yml
conda activate nlpass
```

**Dependencies:** PyTorch ≥ 2.2, sentencepiece, nltk, HuggingFace datasets, sacrebleu, matplotlib

---

## Quickstart

### Step 1 — Prepare data

Downloads WMT14 de-en, NLTK-tokenises, filters >50 word tokens, samples 10k/1k/1k, trains BPE models, encodes and saves all splits.

```bash
python -m src.prepare_data
```

Options:

```bash
python -m src.prepare_data \
  --src_lang   de \          # source language
  --tgt_lang   en \          # target language
  --train_size 10000 \       # training pairs
  --val_size   1000 \        # validation pairs
  --test_size  1000 \        # test pairs
  --max_len    50 \          # max word tokens (filter threshold)
  --vocab_size 8000 \        # BPE vocab size (8000–16000)
  --out_dir    data
```

Prints the required report:
- Pairs remaining after filtering
- Average sentence length (word-level and BPE-level)
- Source and target vocabulary sizes

Outputs to `data/`:
```
data/
├── train.src / train.tgt    # BPE-encoded training pairs
├── val.src   / val.tgt
├── test.src  / test.tgt
├── spm.src.model            # SentencePiece BPE model (source)
└── spm.tgt.model            # SentencePiece BPE model (target)
```

### Step 2 — Train and evaluate

```bash
python main.py
```

All outputs are saved to `results/` by default:
```
results/
├── best_model.pt      # best checkpoint (lowest val loss)
├── loss_curves.png    # training loss per iteration + train/val per epoch
└── results.txt        # BLEU scores + 5 translation examples
```

---

## Arguments

### `prepare_data.py`

| Argument | Default | Description |
|---|---|---|
| `--src_lang` | `de` | Source language code |
| `--tgt_lang` | `en` | Target language code |
| `--train_size` | 10000 | Training sentence pairs |
| `--val_size` | 1000 | Validation sentence pairs |
| `--test_size` | 1000 | Test sentence pairs |
| `--max_len` | 50 | Max **word-level** token filter |
| `--vocab_size` | 8000 | BPE vocabulary size (8k–16k) |
| `--out_dir` | `data` | Output directory |
| `--seed` | 42 | Random seed |

### `main.py`

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data` | Directory with BPE models and split files |
| `--embed_dim` | 256 | Embedding dimension `e` |
| `--hidden_dim` | 512 | LSTM hidden size `h` |
| `--num_layers` | 2 | LSTM layers (encoder and decoder) |
| `--dropout` | 0.3 | Dropout probability |
| `--batch_size` | 64 | Mini-batch size `B` |
| `--num_epochs` | 10 | Training epochs |
| `--lr` | 1e-3 | Adam learning rate |
| `--clip` | 1.0 | Gradient clipping threshold `τ` |
| `--beam_size` | 3 | Beam search width `B` (eq 40) |
| `--max_len` | 100 | Max **BPE-level** sequence length |
| `--save_dir` | `results` | Output directory for all results |

---

## Project Structure

```
├── data/
│   ├── train.src / train.tgt        # BPE-encoded splits
│   ├── val.src   / val.tgt
│   ├── test.src  / test.tgt
│   ├── spm.src.model                # SentencePiece model (source)
│   └── spm.tgt.model                # SentencePiece model (target)
├── src/
│   ├── prepare_data.py              # Download, filter, BPE, encode WMT14
│   ├── dataset.py                   # TranslationDataset, padding masks
│   ├── vocab.py                     # SentencePiece BPE vocabulary wrapper
│   ├── model.py                     # Encoder, MultiplicativeAttention, Decoder, Seq2Seq
│   ├── train.py                     # Training loop, evaluation, loss tracking
│   ├── decode.py                    # Greedy decoding + beam search (eq 40)
│   └── utils.py                     # sacrebleu BLEU, loss plotting, checkpointing
├── results/
│   ├── best_model.pt
│   ├── loss_curves.png
│   └── results.txt
├── main.py                          # Entry point
├── environment.yml
└── report.md
```

---

## Platform Notes

Works on **Mac** and **Windows**.

- **Mac Apple Silicon (M1/M2/M3):** change the device line in `main.py` to use MPS:
  ```python
  device = torch.device(
      'cuda' if torch.cuda.is_available()
      else 'mps' if torch.backends.mps.is_available()
      else 'cpu'
  )
  ```
- **Windows with Nvidia GPU:** uncomment `pytorch-cuda=12.1` in `environment.yml`.
- **CPU only:** works out of the box; reduce `--hidden_dim 256 --num_layers 1 --batch_size 32` for faster iteration.
