"""
pack.py – Ternarize weight tensors and pack them into dual-bitmask format.

Encoding (2 bits per weight):
  nonzero_mask  bit = 1  →  value is ±1
  sign_mask     bit = 1  →  value is +1  (only meaningful when nonzero_mask bit is 1)

Groups of 32 consecutive values along the last dimension are packed into
a single pair of uint32 words (bit 0 = first element, bit 31 = 32nd).
"""

from typing import Tuple

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────
# Padding
# ──────────────────────────────────────────────────────────────────────

def pad_to_multiple(tensor: torch.Tensor, multiple: int = 32) -> torch.Tensor:
    """Pad the last dimension of *tensor* with zeros so its size is a
    multiple of *multiple*.  Returns the tensor unchanged if it already
    satisfies the constraint.
    """
    last = tensor.shape[-1]
    remainder = last % multiple
    if remainder == 0:
        return tensor
    pad_size = multiple - remainder
    return torch.nn.functional.pad(tensor, (0, pad_size), mode="constant", value=0)


# ──────────────────────────────────────────────────────────────────────
# Ternarization
# ──────────────────────────────────────────────────────────────────────

def ternarize_tensor(
    weight: torch.Tensor,
    group_size: int = 128,
    threshold_factor: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantise a floating-point weight tensor to ternary {-1, 0, +1}.

    Parameters
    ----------
    weight : torch.Tensor
        Arbitrary-shape float tensor.
    group_size : int
        Number of consecutive elements per quantization group.
    threshold_factor : float
        Fraction of per-group mean(|w|) used as the dead-zone threshold.

    Returns
    -------
    ternary : torch.Tensor   (int8, same shape as *weight*)
        Values in {-1, 0, +1}.
    scales : torch.Tensor     (float16, flat)
        One scale per group = mean(|w_i|) for the non-zero entries of that
        group.  Shape: (numel // group_size,).
    """
    original_shape = weight.shape
    flat = weight.reshape(-1)
    numel = flat.numel()

    # If numel is not divisible by group_size, pad temporarily
    if numel % group_size != 0:
        pad_len = group_size - (numel % group_size)
        flat = torch.nn.functional.pad(flat, (0, pad_len), value=0.0)
    else:
        pad_len = 0

    groups = flat.reshape(-1, group_size)                    # (n_groups, group_size)
    abs_groups = groups.abs()

    # Per-group threshold
    mean_abs = abs_groups.mean(dim=1, keepdim=True)          # (n_groups, 1)
    threshold = threshold_factor * mean_abs                  # (n_groups, 1)

    # Assign ternary values
    ternary = torch.zeros_like(groups, dtype=torch.int8)
    ternary[groups > threshold] = 1
    ternary[groups < -threshold] = -1

    # Per-group scales: mean |w| over non-zero entries
    nonzero_mask = ternary != 0
    # Avoid division by zero for all-zero groups
    nonzero_count = nonzero_mask.sum(dim=1).clamp(min=1)     # (n_groups,)
    nonzero_sum = (abs_groups * nonzero_mask.float()).sum(dim=1)
    scales = (nonzero_sum / nonzero_count.float()).to(torch.float16)  # (n_groups,)

    # Remove padding that was added for grouping
    if pad_len > 0:
        ternary = ternary.reshape(-1)[:numel].reshape(original_shape)
    else:
        ternary = ternary.reshape(original_shape)

    return ternary, scales


# ──────────────────────────────────────────────────────────────────────
# Bitmask packing
# ──────────────────────────────────────────────────────────────────────

def pack_ternary_to_bitmasks(
    ternary: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack a 2-D int8 ternary tensor into dual uint32 bitmasks.

    Parameters
    ----------
    ternary : torch.Tensor
        Shape (out_features, in_features), dtype int8, values in {-1, 0, +1}.
        *in_features* **must** be divisible by 32.

    Returns
    -------
    nonzero_masks : np.ndarray, dtype uint32, shape (out_features, in_features // 32)
        Bit *k* is 1 iff ternary[row, 32*col + k] != 0.
    sign_masks : np.ndarray, dtype uint32, shape (out_features, in_features // 32)
        Bit *k* is 1 iff ternary[row, 32*col + k] > 0.
    """
    assert ternary.ndim == 2, "pack_ternary_to_bitmasks expects a 2-D tensor"
    out_features, in_features = ternary.shape
    assert in_features % 32 == 0, (
        f"in_features ({in_features}) must be divisible by 32"
    )

    n_words = in_features // 32
    t = ternary.numpy().astype(np.int8)                       # ensure numpy

    nonzero_masks = np.zeros((out_features, n_words), dtype=np.uint32)
    sign_masks = np.zeros((out_features, n_words), dtype=np.uint32)

    for row in range(out_features):
        for col in range(n_words):
            nz_word: np.uint32 = np.uint32(0)
            s_word: np.uint32 = np.uint32(0)
            base = col * 32
            for bit in range(32):
                val = t[row, base + bit]
                if val != 0:
                    nz_word |= np.uint32(1 << bit)
                if val > 0:
                    s_word |= np.uint32(1 << bit)
            nonzero_masks[row, col] = nz_word
            sign_masks[row, col] = s_word

    return nonzero_masks, sign_masks
