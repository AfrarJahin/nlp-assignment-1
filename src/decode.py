"""Greedy and beam-search decoding for the BiLSTM / multiplicative-attention model."""

import torch


def greedy_decode(model, src: torch.Tensor, tgt_vocab, max_len: int,
                  device: torch.device, src_pad_idx: int = 0):
    """Single best-path decoding."""
    model.eval()
    with torch.no_grad():
        src = src.to(device)
        src_mask = (src != src_pad_idx)                        # (1, src_len)

        enc_outputs, enc_hidden, enc_cell = model.encoder(src, src_mask)
        hidden, cell = model._init_decoder_state(enc_hidden, enc_cell)
        o_prev = torch.zeros(1, model.decoder.hidden_dim, device=device)

        input_token = torch.tensor([tgt_vocab.bos_idx], device=device)
        generated   = []

        for _ in range(max_len):
            p_t, hidden, cell, o_prev, _ = model.decoder(
                input_token, hidden, cell, enc_outputs, o_prev, src_mask)
            pred = p_t.argmax(dim=1)
            if pred.item() == tgt_vocab.eos_idx:
                break
            generated.append(pred.item())
            input_token = pred

    return tgt_vocab.decode(generated)


def beam_search(model, src: torch.Tensor, tgt_vocab, max_len: int,
                beam_size: int, device: torch.device, src_pad_idx: int = 0):
    """
    Beam search decoding (eq 40).

    Score: s(ŷ_{1:t}) = Σ_{k=1}^{t} log p(ŷ_k | ŷ_{<k}, x)  — raw sum, no length norm.

    Each beam entry: (cumulative_score, token_ids, hidden, cell, o_prev)

    Stops when all beams have generated <eos> or max_len is reached.
    """
    model.eval()
    with torch.no_grad():
        src = src.to(device)
        src_mask = (src != src_pad_idx)                        # (1, src_len)

        enc_outputs, enc_hidden, enc_cell = model.encoder(src, src_mask)
        hidden, cell = model._init_decoder_state(enc_hidden, enc_cell)
        o_prev = torch.zeros(1, model.decoder.hidden_dim, device=device)

        # Beam: (score, token_list, hidden, cell, o_prev)
        beams     = [(0.0, [tgt_vocab.bos_idx], hidden, cell, o_prev)]
        completed = []

        for _ in range(max_len):
            candidates = []
            for score, tokens, h, c, o in beams:
                if tokens[-1] == tgt_vocab.eos_idx:
                    # This hypothesis is done — move to completed
                    completed.append((score, tokens))
                    continue

                in_tok = torch.tensor([tokens[-1]], device=device)
                p_t, new_h, new_c, new_o, _ = model.decoder(
                    in_tok, h, c, enc_outputs, o, src_mask)

                log_probs = torch.log_softmax(p_t, dim=1).squeeze(0)  # (vocab,)
                top_scores, top_ids = log_probs.topk(beam_size)

                for s, idx in zip(top_scores.tolist(), top_ids.tolist()):
                    candidates.append(
                        (score + s, tokens + [idx], new_h, new_c, new_o))

            if not candidates:
                # All beams completed
                break

            # Keep top-B by raw cumulative log-prob (eq 40 — no length normalization)
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

        # Drain any remaining live beams
        completed += [(s, t) for s, t, *_ in beams]

        # Select best by raw score (eq 40)
        best_score, best_tokens = max(completed, key=lambda x: x[0])

        return tgt_vocab.decode(best_tokens[1:])   # strip <bos>
