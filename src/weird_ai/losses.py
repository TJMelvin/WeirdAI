"""
Loss utilities for Weird AI.

These functions calculate how well the model predicts the next token.
"""

import torch
import torch.nn.functional as F


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate cross-entropy loss for one batch.

    Args:
        input_batch: Tensor of shape (batch_size, num_tokens).
        target_batch: Tensor of shape (batch_size, num_tokens).
        model: The Weird AI model.
        device: CPU or CUDA device.

    Returns:
        A scalar loss tensor.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)
    
    logits = logits.reshape(-1, logits.size(-1))
    target_batch = target_batch.reshape(-1)
   
    loss = F.cross_entropy(logits, target_batch)
   
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate the average loss across a data loader.

    Args:
        data_loader: A PyTorch DataLoader.
        model: The Weird AI model.
        device: CPU or CUDA device.
        num_batches: Optional limit on number of batches to evaluate.

    Returns:
        Average loss as a float.
    """
    if len(data_loader) == 0:
        return float("nan")

    total_loss = 0.0
    batch_count = 0

    for input_batch, target_batch in data_loader:

        loss = calc_loss_batch(
            input_batch,
            target_batch,
            model,
            device
        )

        total_loss += loss.item()
        batch_count += 1

        if num_batches is not None and batch_count >= num_batches:
            break

    return total_loss / batch_count


def calculate_perplexity(loss):
    """
    Convert cross-entropy loss into perplexity.

    Args:
        loss: A scalar loss value or tensor.

    Returns:
        Perplexity value.
    """
    return torch.exp(torch.as_tensor(loss))