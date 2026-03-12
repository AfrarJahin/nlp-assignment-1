"""Helper functions: BLEU scoring, checkpoint I/O, logging."""

import torch
import sacrebleu


def compute_bleu(references: list[list[str]], hypotheses: list[list[str]]) -> float:
    """
    Corpus-level BLEU using sacrebleu (standard, comparable to published results).

    Args:
        references : list of reference token lists
        hypotheses : list of hypothesis token lists
    Returns:
        BLEU score as a float in [0, 100]
    """
    hyp_strs = [' '.join(h) for h in hypotheses]
    ref_strs = [' '.join(r) for r in references]
    result = sacrebleu.corpus_bleu(hyp_strs, [ref_strs])
    return result.score   # already in [0, 100]


def plot_losses(train_losses: list[float], val_losses: list[float],
                batch_losses: list[float], save_path: str = 'loss_curves.png'):
    """
    Save two side-by-side plots required by the assignment:
      Left  — training loss per iteration (batch)
      Right — training & validation loss per epoch
    """
    import matplotlib
    matplotlib.use('Agg')          # no display needed
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: per-iteration training loss
    ax1.plot(batch_losses, linewidth=0.8, alpha=0.85)
    ax1.set_xlabel('Iteration (batch)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss per Iteration')
    ax1.grid(True, alpha=0.3)

    # Right: per-epoch train vs val loss
    epochs = range(1, len(train_losses) + 1)
    ax2.plot(epochs, train_losses, marker='o', label='Train')
    ax2.plot(epochs, val_losses,   marker='s', label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Train / Validation Loss per Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Loss curves saved to {save_path}')


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch']


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
