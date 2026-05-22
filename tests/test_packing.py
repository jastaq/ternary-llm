#!/usr/bin/env python3
"""
test_packing.py – Unit tests for pack.py (no pytest required).

Run:
    python test_packing.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Ensure the ternarize package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ternarize"))

from pack import pack_ternary_to_bitmasks, pad_to_multiple, ternarize_tensor


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def unpack_bitmasks(nonzero_masks: np.ndarray, sign_masks: np.ndarray) -> np.ndarray:
    """Manually unpack dual bitmasks back to int8 {-1, 0, +1}."""
    out_features, n_words = nonzero_masks.shape
    in_features = n_words * 32
    result = np.zeros((out_features, in_features), dtype=np.int8)
    for row in range(out_features):
        for col in range(n_words):
            nz = int(nonzero_masks[row, col])
            sg = int(sign_masks[row, col])
            base = col * 32
            for bit in range(32):
                if nz & (1 << bit):
                    if sg & (1 << bit):
                        result[row, base + bit] = 1
                    else:
                        result[row, base + bit] = -1
    return result


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

def test_pack_unpack_known():
    """Pack a hand-crafted tensor and verify round-trip via manual unpack."""
    print("test_pack_unpack_known ... ", end="")

    # 2 rows, 64 columns (divisible by 32)
    t = torch.zeros(2, 64, dtype=torch.int8)

    # Row 0: set specific positions
    t[0, 0] = 1
    t[0, 1] = -1
    t[0, 31] = 1
    t[0, 32] = -1
    t[0, 63] = -1

    # Row 1: alternating +1 / -1 for the first 32 entries
    for i in range(32):
        t[1, i] = 1 if i % 2 == 0 else -1

    nonzero, sign = pack_ternary_to_bitmasks(t)

    assert nonzero.shape == (2, 2), f"Expected (2,2), got {nonzero.shape}"
    assert sign.shape == (2, 2), f"Expected (2,2), got {sign.shape}"

    # Manually verify row 0 / word 0
    # Positions 0,1,31 are nonzero  →  bits 0,1,31  →  0x80000003
    expected_nz_0_0 = (1 << 0) | (1 << 1) | (1 << 31)
    assert nonzero[0, 0] == expected_nz_0_0, (
        f"nonzero[0,0]: expected {expected_nz_0_0:#010x}, got {nonzero[0, 0]:#010x}"
    )
    # Only position 0 and 31 are positive  →  bits 0,31
    expected_sg_0_0 = (1 << 0) | (1 << 31)
    assert sign[0, 0] == expected_sg_0_0, (
        f"sign[0,0]: expected {expected_sg_0_0:#010x}, got {sign[0, 0]:#010x}"
    )

    # Full round-trip check
    recovered = unpack_bitmasks(nonzero, sign)
    np.testing.assert_array_equal(
        recovered, t.numpy(), err_msg="Round-trip mismatch on known tensor"
    )

    print("OK")


def test_ternarize_output_values():
    """ternarize_tensor output must be in {-1, 0, +1}."""
    print("test_ternarize_output_values ... ", end="")

    torch.manual_seed(42)
    w = torch.randn(256, 128)
    ternary, scales = ternarize_tensor(w, group_size=128, threshold_factor=0.7)

    assert ternary.shape == w.shape, f"Shape mismatch: {ternary.shape} vs {w.shape}"
    unique = set(ternary.unique().tolist())
    assert unique.issubset({-1, 0, 1}), f"Unexpected values: {unique}"
    assert scales.dtype == torch.float16
    assert scales.ndim == 1, f"scales should be 1-D, got ndim={scales.ndim}"

    expected_n_groups = (256 * 128) // 128
    assert scales.shape[0] == expected_n_groups, (
        f"Expected {expected_n_groups} scale groups, got {scales.shape[0]}"
    )

    print("OK")


def test_pad_to_multiple():
    """pad_to_multiple should pad only when necessary."""
    print("test_pad_to_multiple ... ", end="")

    # Already aligned
    t1 = torch.randn(4, 64)
    p1 = pad_to_multiple(t1, 32)
    assert p1.shape == (4, 64), f"Unexpected shape: {p1.shape}"
    assert torch.equal(t1, p1)

    # Needs padding: 50 → 64
    t2 = torch.randn(4, 50)
    p2 = pad_to_multiple(t2, 32)
    assert p2.shape == (4, 64), f"Expected (4,64), got {p2.shape}"
    # Original data preserved
    assert torch.equal(p2[:, :50], t2)
    # Padded region is zero
    assert (p2[:, 50:] == 0).all()

    # 1-D tensor
    t3 = torch.randn(100)
    p3 = pad_to_multiple(t3, 32)
    assert p3.shape == (128,), f"Expected (128,), got {p3.shape}"

    print("OK")


def test_round_trip_shapes():
    """Full pipeline: float → ternarize → pad → pack → verify shapes."""
    print("test_round_trip_shapes ... ", end="")

    torch.manual_seed(123)
    out_features, in_features = 64, 100
    w = torch.randn(out_features, in_features)

    # Pad
    w_padded = pad_to_multiple(w, 32)
    assert w_padded.shape[1] % 32 == 0
    padded_in = w_padded.shape[1]  # should be 128

    # Ternarize
    ternary, scales = ternarize_tensor(w_padded, group_size=128, threshold_factor=0.7)
    assert ternary.shape == (out_features, padded_in)

    # Pack
    nonzero, sign = pack_ternary_to_bitmasks(ternary)
    expected_words = padded_in // 32
    assert nonzero.shape == (out_features, expected_words), (
        f"nonzero shape: {nonzero.shape}"
    )
    assert sign.shape == (out_features, expected_words), f"sign shape: {sign.shape}"
    assert nonzero.dtype == np.uint32
    assert sign.dtype == np.uint32

    # Unpack and verify against ternary
    recovered = unpack_bitmasks(nonzero, sign)
    np.testing.assert_array_equal(
        recovered, ternary.numpy(), err_msg="Round-trip shape test failed"
    )

    print("OK")


def test_all_zero_group():
    """A group of all zeros should produce scale = 0 and all-zero ternary."""
    print("test_all_zero_group ... ", end="")

    w = torch.zeros(1, 128)
    ternary, scales = ternarize_tensor(w, group_size=128, threshold_factor=0.7)
    assert (ternary == 0).all(), "All-zero input should produce all-zero ternary"
    assert scales[0].item() == 0.0, f"Scale for all-zero group should be 0, got {scales[0]}"

    print("OK")


def test_large_values():
    """Extreme values should all be quantised to ±1."""
    print("test_large_values ... ", end="")

    w = torch.ones(1, 128) * 100.0
    w[0, ::2] *= -1  # alternate signs
    ternary, scales = ternarize_tensor(w, group_size=128, threshold_factor=0.7)
    # All entries should be nonzero
    assert (ternary != 0).all(), "Large values should all be nonzero"
    # Check signs match
    for i in range(128):
        expected = -1 if i % 2 == 0 else 1
        assert ternary[0, i].item() == expected, (
            f"Position {i}: expected {expected}, got {ternary[0, i].item()}"
        )

    print("OK")


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_pack_unpack_known()
    test_ternarize_output_values()
    test_pad_to_multiple()
    test_round_trip_shapes()
    test_all_zero_group()
    test_large_values()

    print()
    print("All tests passed ✓")
