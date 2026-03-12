"""Entry point: load BPE vocabs, train model, evaluate with BLEU."""

import os
import torch
import argparse

from src.vocab import Vocabulary
from src.dataset import load_data, get_dataloader
from src.model import Encoder, Decoder, Seq2Seq
from src.train import train
from src.decode import greedy_decode, beam_search
from src.utils import compute_bleu, count_parameters, plot_losses


def parse_args():
    p = argparse.ArgumentParser(description='NLP Assignment 1 - Neural Machine Translation')
    # Data
    p.add_argument('--data_dir',   default='data')
    p.add_argument('--train_src',  default='data/train.src')
    p.add_argument('--train_tgt',  default='data/train.tgt')
    p.add_argument('--val_src',    default='data/val.src')
    p.add_argument('--val_tgt',    default='data/val.tgt')
    p.add_argument('--test_src',   default='data/test.src')
    p.add_argument('--test_tgt',   default='data/test.tgt')
    # Model
    p.add_argument('--embed_dim',  type=int,   default=256)
    p.add_argument('--hidden_dim', type=int,   default=512)
    p.add_argument('--num_layers', type=int,   default=2)
    p.add_argument('--dropout',    type=float, default=0.3)
    # Training
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--num_epochs', type=int,   default=10)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--clip',       type=float, default=1.0)
    # Inference
    p.add_argument('--beam_size',  type=int,   default=3)
    p.add_argument('--max_len',    type=int,   default=100,
                   help='Max BPE token length for model input/output. '
                        'Word-level filter (50) is applied separately in prepare_data.py')
    # Output
    p.add_argument('--save_dir',   default='results',
                   help='Directory to save model, plot, and evaluation results')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Create output directory ───────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    model_path  = os.path.join(args.save_dir, 'best_model.pt')
    plot_path   = os.path.join(args.save_dir, 'loss_curves.png')
    results_path = os.path.join(args.save_dir, 'results.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Saving all outputs to: {args.save_dir}/')

    # ── Load BPE vocabularies ─────────────────────────────────────────────────
    src_vocab = Vocabulary(os.path.join(args.data_dir, 'spm.src.model'))
    tgt_vocab = Vocabulary(os.path.join(args.data_dir, 'spm.tgt.model'))
    print(f'Src vocab: {len(src_vocab):,} | Tgt vocab: {len(tgt_vocab):,}')

    # ── Load pre-tokenised data ───────────────────────────────────────────────
    train_src, train_tgt = load_data(args.train_src, args.train_tgt)
    val_src,   val_tgt   = load_data(args.val_src,   args.val_tgt)
    test_src,  test_tgt  = load_data(args.test_src,  args.test_tgt)

    train_loader = get_dataloader(train_src, train_tgt, src_vocab, tgt_vocab,
                                  args.batch_size, shuffle=True,  max_len=args.max_len)
    val_loader   = get_dataloader(val_src,   val_tgt,   src_vocab, tgt_vocab,
                                  args.batch_size, shuffle=False, max_len=args.max_len)

    # ── Build model ───────────────────────────────────────────────────────────
    encoder = Encoder(len(src_vocab), args.embed_dim, args.hidden_dim,
                      args.num_layers, args.dropout, src_vocab.pad_idx)
    decoder = Decoder(len(tgt_vocab), args.embed_dim, args.hidden_dim,
                      args.num_layers, args.dropout, tgt_vocab.pad_idx)
    model = Seq2Seq(encoder, decoder, device).to(device)
    print(f'Model parameters: {count_parameters(model):,}')

    # ── Train ─────────────────────────────────────────────────────────────────
    train_losses, val_losses, batch_losses = train(
        model, train_loader, val_loader, args.num_epochs, args.lr,
        args.clip, tgt_vocab.pad_idx, device, model_path, args.save_dir)

    plot_losses(train_losses, val_losses, batch_losses, save_path=plot_path)

    # ── Evaluate BLEU on test set ─────────────────────────────────────────────
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    greedy_hyps, beam_hyps, references = [], [], []
    decoded_sources = []
    for src_sent, ref_sent in zip(test_src, test_tgt):
        src_ids = torch.tensor(
            [src_vocab.encode(src_sent, add_eos=True)], dtype=torch.long)
        greedy_hyps.append(greedy_decode(model, src_ids, tgt_vocab, args.max_len, device))
        beam_hyps.append(beam_search(model, src_ids, tgt_vocab, args.max_len, args.beam_size, device))
        references.append(tgt_vocab.decode(tgt_vocab.encode(ref_sent, add_eos=True)))
        decoded_sources.append(src_vocab.decode(src_vocab.encode(src_sent, add_eos=True)))

    greedy_bleu = compute_bleu(references, greedy_hyps)
    beam_bleu   = compute_bleu(references, beam_hyps)

    # ── Build results text ────────────────────────────────────────────────────
    lines = []
    lines.append('── Test BLEU ───────────────────────────────────────')
    lines.append(f'  Greedy : {greedy_bleu:.2f}')
    lines.append(f'  Beam-{args.beam_size} : {beam_bleu:.2f}')
    lines.append('────────────────────────────────────────────────────')
    lines.append('')
    lines.append('── Translation Examples ────────────────────────────')
    for i in range(min(5, len(test_src))):
        lines.append(f'\n[{i + 1}]')
        lines.append(f'  Source    : {" ".join(decoded_sources[i])}')
        lines.append(f'  Reference : {" ".join(references[i])}')
        lines.append(f'  Greedy    : {" ".join(greedy_hyps[i])}')
        lines.append(f'  Beam      : {" ".join(beam_hyps[i])}')
    lines.append('\n────────────────────────────────────────────────────')

    output = '\n'.join(lines)

    # Print to console
    print('\n' + output)

    # Save to file
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(output + '\n')
    print(f'Results saved to {results_path}')


if __name__ == '__main__':
    main()
