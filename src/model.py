"""
Encoder-Decoder with Multiplicative Attention for sequence-to-sequence translation.

Architecture (per assignment spec):
  Encoder : Bidirectional LSTM
  Decoder : Unidirectional LSTM  (input = [y_t ; o_{t-1}])
  Attention: Multiplicative  e_{t,i} = h_t^{dec T} W_attProj h_i^{enc}
  Output  : u_t=[a_t;h_t^dec] -> W_u -> tanh -> dropout -> W_vocab -> softmax

Tensor dimension key
  e  = embed_dim
  h  = hidden_dim   (decoder / single-direction encoder size)
  2h = encoder output size (bidirectional)
"""

import torch
import torch.nn as nn


# ── Encoder ──────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Bidirectional LSTM encoder (equations 16-20)."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None):
        """
        Args:
            src      : (batch, src_len)  token ids
            src_mask : (batch, src_len)  True = real token, False = <pad>
        Returns:
            outputs  : (batch, src_len, 2h)  all encoder hidden states h_i^enc
            hidden   : (num_layers*2, batch, h)
            cell     : (num_layers*2, batch, h)
        """
        embedded = self.dropout(self.embedding(src))           # (batch, src_len, e)

        if src_mask is not None:
            lengths = src_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False)
            packed_out, (hidden, cell) = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=src.size(1))
        else:
            outputs, (hidden, cell) = self.rnn(embedded)

        return outputs, hidden, cell                           # outputs: (batch, src_len, 2h)


# ── Multiplicative Attention ──────────────────────────────────────────────────

class MultiplicativeAttention(nn.Module):
    """
    Luong-style multiplicative attention (equations 27-31).

    e_{t,i} = (h_t^dec)^T  W_attProj  h_i^enc
    W_attProj ∈ R^{h × 2h}
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Maps decoder hidden (h) -> encoder space (2h)  so dot product works
        self.W_attProj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)

    def forward(self, h_dec: torch.Tensor, enc_outputs: torch.Tensor,
                src_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_dec      : (batch, h)           decoder hidden state at step t
            enc_outputs: (batch, src_len, 2h) all encoder states
            src_mask   : (batch, src_len)     True = real token
        Returns:
            alpha  : (batch, src_len)  attention weights
            a_t    : (batch, 2h)       context vector
        """
        # proj: (batch, 2h)  =  h_t^dec @ W_attProj  (row-vector convention)
        proj = self.W_attProj(h_dec)                           # (batch, 2h)

        # scores: (batch, src_len)  =  enc_outputs · proj^T
        scores = torch.bmm(enc_outputs, proj.unsqueeze(2)).squeeze(2)  # (batch, src_len)

        # Apply padding mask: set pad positions to -inf before softmax (eq 29)
        if src_mask is not None:
            scores = scores.masked_fill(~src_mask, float('-inf'))

        alpha = torch.softmax(scores, dim=1)                   # (batch, src_len)  eq 30
        a_t = torch.bmm(alpha.unsqueeze(1), enc_outputs).squeeze(1)   # (batch, 2h)  eq 31
        return alpha, a_t


# ── Decoder ──────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Unidirectional LSTM decoder (equations 23-35).

    Input to LSTM at step t:  ȳ_t = [y_t ; o_{t-1}]   ∈ R^{e+h}
    Output pipeline:
        u_t = [a_t ; h_t^dec]  ∈ R^{3h}
        v_t = W_u u_t           ∈ R^h
        o_t = dropout(tanh(v_t))∈ R^h
        p_t = softmax(W_vocab o_t)
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = MultiplicativeAttention(hidden_dim)
        # LSTM input: [y_t ; o_{t-1}] ∈ R^{e+h}   (eq 25)
        self.rnn = nn.LSTM(
            embed_dim + hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.W_u     = nn.Linear(hidden_dim * 3, hidden_dim, bias=False)  # eq 33  (3h->h)
        self.W_vocab = nn.Linear(hidden_dim, vocab_size)                   # eq 35
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                tgt_token:   torch.Tensor,
                hidden:      torch.Tensor,
                cell:        torch.Tensor,
                enc_outputs: torch.Tensor,
                o_prev:      torch.Tensor,
                src_mask:    torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tgt_token  : (batch,)          current target token id
            hidden     : (num_layers, batch, h)
            cell       : (num_layers, batch, h)
            enc_outputs: (batch, src_len, 2h)
            o_prev     : (batch, h)        previous combined output vector o_{t-1}
            src_mask   : (batch, src_len)
        Returns:
            p_t    : (batch, vocab_size)   unnormalised logits
            hidden : (num_layers, batch, h)
            cell   : (num_layers, batch, h)
            o_t    : (batch, h)            combined output vector for next step
            alpha  : (batch, src_len)      attention weights
        """
        y_t = self.dropout(self.embedding(tgt_token))          # (batch, e)   eq 23

        # Concatenate with previous output: ȳ_t = [y_t ; o_{t-1}]  eq 25
        rnn_in = torch.cat([y_t, o_prev], dim=1).unsqueeze(1)  # (batch, 1, e+h)

        rnn_out, (hidden, cell) = self.rnn(rnn_in, (hidden, cell))
        h_t = rnn_out.squeeze(1)                               # (batch, h)   eq 26

        # Multiplicative attention
        alpha, a_t = self.attention(h_t, enc_outputs, src_mask)  # (batch,src_len), (batch,2h)

        # Combined output  eq 32-34
        u_t = torch.cat([a_t, h_t], dim=1)                    # (batch, 3h)  eq 32
        v_t = self.W_u(u_t)                                    # (batch, h)   eq 33
        o_t = self.dropout(torch.tanh(v_t))                    # (batch, h)   eq 34

        p_t = self.W_vocab(o_t)                                # (batch, V)   eq 35  (logits)
        return p_t, hidden, cell, o_t, alpha


# ── Seq2Seq ───────────────────────────────────────────────────────────────────

class Seq2Seq(nn.Module):
    """
    Full encoder-decoder model.

    Decoder init (equations 21-22):
        h_0^dec = W_h [ ←h_1^enc ; →h_m^enc ]   W_h ∈ R^{h×2h}
        c_0^dec = W_c [ ←c_1^enc ; →c_m^enc ]   W_c ∈ R^{h×2h}
    where ←h_1 is the backward final hidden and →h_m is the forward final hidden.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device
        h  = decoder.hidden_dim
        # Project concatenated encoder finals to decoder init size  (eqs 21-22)
        self.W_h = nn.Linear(h * 2, h, bias=False)
        self.W_c = nn.Linear(h * 2, h, bias=False)

    def _init_decoder_state(self, enc_hidden: torch.Tensor, enc_cell: torch.Tensor):
        """
        Build initial decoder hidden / cell from encoder final states.

        enc_hidden : (num_layers*2, batch, h)
        Returns    : each (dec_num_layers, batch, h)
        """
        # Last layer forward: enc_hidden[-2], backward: enc_hidden[-1]
        fwd_h, bwd_h = enc_hidden[-2], enc_hidden[-1]   # each (batch, h)
        fwd_c, bwd_c = enc_cell[-2],   enc_cell[-1]

        h0 = torch.tanh(self.W_h(torch.cat([bwd_h, fwd_h], dim=1)))  # (batch, h)
        c0 = torch.tanh(self.W_c(torch.cat([bwd_c, fwd_c], dim=1)))  # (batch, h)

        # Repeat across decoder layers
        n = self.decoder.rnn.num_layers
        return h0.unsqueeze(0).repeat(n, 1, 1), c0.unsqueeze(0).repeat(n, 1, 1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor | None = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Args:
            src      : (batch, src_len)
            tgt      : (batch, tgt_len)   includes <bos> at position 0
            src_mask : (batch, src_len)
        Returns:
            outputs  : (batch, tgt_len, vocab_size)  logits (position 0 is zeros)
        """
        batch_size, tgt_len = tgt.size()
        vocab_size = self.decoder.W_vocab.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=self.device)

        enc_outputs, enc_hidden, enc_cell = self.encoder(src, src_mask)
        hidden, cell = self._init_decoder_state(enc_hidden, enc_cell)

        # o_0 = 0  (eq 24)
        o_prev = torch.zeros(batch_size, self.decoder.hidden_dim, device=self.device)

        input_token = tgt[:, 0]   # <bos>
        for t in range(1, tgt_len):
            p_t, hidden, cell, o_prev, _ = self.decoder(
                input_token, hidden, cell, enc_outputs, o_prev, src_mask)
            outputs[:, t] = p_t
            use_teacher  = torch.rand(1).item() < teacher_forcing_ratio
            input_token  = tgt[:, t] if use_teacher else p_t.argmax(dim=1)

        return outputs
