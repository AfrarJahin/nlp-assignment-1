"""Training loop for the seq2seq translation model.

Loss design (per assignment spec, eqs 36-38):
  J_t(θ)     = -log p_t[y_t^gold]          per-step NLL        (eq 36)
  J(θ)       = Σ_t J_t(θ)                  sentence loss       (eq 37)
  J_batch(θ) = (1/B) Σ_b J^(b)(θ)          minibatch average   (eq 38)

nn.CrossEntropyLoss(reduction='mean', ignore_index=pad) computes exactly
J_batch averaged over all non-pad target positions across the batch.

Gradient clipping (eq 39):  g ← g · min(1, τ / ‖g‖₂)
Applied via nn.utils.clip_grad_norm_(params, τ).
"""

import json
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    """One training epoch. Returns (avg_loss, per_batch_losses)."""
    model.train()
    total_loss   = 0.0
    batch_losses = []

    for src, tgt, src_mask in dataloader:
        src, tgt, src_mask = src.to(device), tgt.to(device), src_mask.to(device)
        optimizer.zero_grad()

        output = model(src, tgt, src_mask=src_mask)
        # Skip position 0 (<bos>); reshape for CrossEntropyLoss
        output = output[:, 1:].reshape(-1, output.size(-1))
        target = tgt[:, 1:].reshape(-1)

        # J_batch(θ) — eq 38: mean NLL over non-pad tokens in the batch
        loss = criterion(output, target)
        loss.backward()

        # Gradient clipping — eq 39: g ← g · min(1, τ / ‖g‖₂)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        batch_losses.append(loss.item())
        total_loss  += loss.item()

    return total_loss / len(dataloader), batch_losses


def evaluate(model, dataloader, criterion, device):
    """Validation pass. Returns avg loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt, src_mask in dataloader:
            src, tgt, src_mask = src.to(device), tgt.to(device), src_mask.to(device)
            output = model(src, tgt, src_mask=src_mask, teacher_forcing_ratio=0.0)
            output = output[:, 1:].reshape(-1, output.size(-1))
            target = tgt[:, 1:].reshape(-1)
            loss   = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train(
    model,
    train_loader,
    val_loader,
    num_epochs,
    lr,
    clip,
    tgt_pad_idx,
    device,
    save_path='best_model.pt',
    epoch_results_dir=None,
):
    """
    Full training loop.

    Hyperparameters reported (per assignment):
      Optimiser  : Adam
      LR         : lr
      Batch size : inferred from train_loader
      Dropout    : set in model definition
      Clip (τ)   : clip

    Returns:
        train_losses : list of per-epoch average train losses
        val_losses   : list of per-epoch average val losses
        all_batch_losses : flat list of every batch loss (for iteration-level plot)
    """
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)   # eq 36-38
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_loss    = float('inf')
    train_losses     = []
    val_losses       = []
    all_batch_losses = []

    print(f'Optimiser : Adam  |  LR : {lr}  |  Clip τ : {clip}  '
          f'|  Batch size : {train_loader.batch_size}')

    if epoch_results_dir is None:
        epoch_results_dir = os.path.dirname(save_path) or '.'
    os.makedirs(epoch_results_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        epoch_loss, batch_losses = train_epoch(
            model, train_loader, optimizer, criterion, clip, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        all_batch_losses.extend(batch_losses)

        is_best = val_loss < best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        epoch_metrics = {
            'epoch': epoch,
            'train_loss': float(epoch_loss),
            'val_loss': float(val_loss),
            'learning_rate': float(optimizer.param_groups[0]['lr']),
            'is_best': is_best,
            'best_val_loss_so_far': float(best_val_loss),
        }
        epoch_file = os.path.join(epoch_results_dir, f'epoch_{epoch:03d}.json')
        with open(epoch_file, 'w', encoding='utf-8') as f:
            json.dump(epoch_metrics, f, indent=2)

        print(f'Epoch {epoch:02d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}')

    print(f'Best Val Loss: {best_val_loss:.4f}')
    return train_losses, val_losses, all_batch_losses
