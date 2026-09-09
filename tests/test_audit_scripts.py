"""CPU unit tests for the Work-Package-D thesis-artifact audit scripts.

These pin the *math* of the two audit guards on synthetic data — no real dataset
IO — so the guards themselves stay trustworthy:

  * scripts/audit_mmipc.py       — the MM-IPC reconstruction residual check
                                   (rms / expected_retained_es / residual_rms).
  * scripts/audit_split_leakage.py — the source-set intersection logic
                                   (find_original_path_columns / source_set),
                                   including the role-swap-aware speaker pooling.
"""

import math

import pandas as pd
import pytest
import torch

from scripts.audit_mmipc import (
    DEFAULT_THRESHOLD,
    expected_retained_es,
    residual_rms,
    rms,
)
from scripts.audit_split_leakage import (
    SPEAKER_COLUMNS,
    find_original_path_columns,
    source_set,
)


# --------------------------------------------------------------------------- #
# D1 — MM-IPC reconstruction residual math
# --------------------------------------------------------------------------- #

def _const_layers(has_reverb):
    """Synthetic component layers with distinct constant values (length 100)."""
    n = 100
    layers = {
        "sp1_dry": torch.full((n,), 0.10),
        "sp2_dry": torch.full((n,), 0.20),
        "scene": torch.full((n,), 0.30),
        "event_dry": torch.full((n,), 0.40),
    }
    if has_reverb:
        layers["sp1_reverb"] = torch.full((n,), 0.01)
        layers["sp2_reverb"] = torch.full((n,), 0.02)
        layers["event_reverb"] = torch.full((n,), 0.04)
    return layers


def test_rms_matches_manual():
    assert rms(torch.tensor([3.0, 4.0])) == pytest.approx(math.sqrt(12.5))
    assert rms(torch.zeros(50)) == pytest.approx(0.0)


@pytest.mark.parametrize("variant,expected_value", [
    ("SER", 0.10 + 0.01 + 0.30 + 0.40 + 0.04),  # keep scene + event(+rev) + sp1 reverb
    ("SR",  0.10 + 0.01 + 0.30),                 # keep scene + sp1 reverb, drop event
    ("ER",  0.10 + 0.01 + 0.40 + 0.04),          # keep event(+rev) + sp1 reverb, drop scene
    ("R",   0.10 + 0.01),                        # keep sp1 reverb only
    ("C",   0.10),                               # keep nothing -> dry speaker1
])
def test_expected_retained_es_indoor(variant, expected_value):
    layers = _const_layers(has_reverb=True)
    target = expected_retained_es(layers, variant, has_reverb=True)
    assert torch.allclose(target, torch.full((100,), expected_value), atol=1e-6), \
        f"indoor {variant} should retain a constant {expected_value}"


@pytest.mark.parametrize("variant,expected_value", [
    ("SE", 0.10 + 0.30 + 0.40),  # keep scene + event (no reverb tails outdoor)
    ("S",  0.10 + 0.30),         # keep scene, drop event
    ("E",  0.10 + 0.40),         # keep event, drop scene
    ("C",  0.10),                # keep nothing -> dry speaker1
])
def test_expected_retained_es_outdoor(variant, expected_value):
    layers = _const_layers(has_reverb=False)
    target = expected_retained_es(layers, variant, has_reverb=False)
    assert torch.allclose(target, torch.full((100,), expected_value), atol=1e-6), \
        f"outdoor {variant} should retain a constant {expected_value}"


def test_es_c_reconstructs_dry_speaker1():
    """The headline claim: ES+C retains exactly dry speaker1 (indoor and outdoor)."""
    for has_reverb in (True, False):
        layers = _const_layers(has_reverb)
        target = expected_retained_es(layers, "C", has_reverb)
        assert torch.allclose(target, layers["sp1_dry"], atol=0.0)


def test_residual_rms_zero_on_exact_match():
    a = torch.randn(200)
    assert residual_rms(a, a.clone()) == pytest.approx(0.0, abs=1e-7)


def test_residual_rms_aligns_to_shorter_length():
    a = torch.ones(120)
    b = torch.ones(100)
    # Extra samples in `a` beyond index 100 must be ignored, not misalign the diff.
    assert residual_rms(a, b) == pytest.approx(0.0, abs=1e-7)


def test_broken_reconstruction_exceeds_threshold():
    """A dropped layer (signal-scale error) must blow past the PCM-floor threshold."""
    layers = _const_layers(has_reverb=True)
    ideal = expected_retained_es(layers, "SER", has_reverb=True)
    # Simulate a broken MM-IPC output that forgot to subtract the 0.30 scene layer.
    broken = ideal + layers["scene"]
    resid = residual_rms(broken, ideal)
    assert resid == pytest.approx(0.30, abs=1e-6)
    assert resid > DEFAULT_THRESHOLD  # the guard would FAIL, as intended


# --------------------------------------------------------------------------- #
# D2 — split-leakage intersection logic
# --------------------------------------------------------------------------- #

def _toy_corpus(speech1, speech2, scene, event):
    """Build a minimal corpus DataFrame with the provenance columns."""
    n = max(len(speech1), len(speech2), len(scene), len(event))

    def pad(xs):
        return list(xs) + [None] * (n - len(xs))

    return pd.DataFrame({
        "subset": ["x"] * n,
        "speech1OryginalPath": pad(speech1),
        "speech2OryginalPath": pad(speech2),
        "sceneOryginalPath": pad(scene),
        "eventOryginalPath": pad(event),
        "sceneClass": ["park"] * n,  # a decoy non-provenance column
    })


def test_find_original_path_columns():
    df = _toy_corpus(["a"], ["b"], ["s"], ["e"])
    cols = find_original_path_columns(df)
    assert set(cols) == {
        "speech1OryginalPath", "speech2OryginalPath",
        "sceneOryginalPath", "eventOryginalPath",
    }
    assert "sceneClass" not in cols and "subset" not in cols


def test_source_set_dropna_and_union():
    df = _toy_corpus(["a", "b", None], ["c"], ["s"], ["e"])
    assert source_set(df, "speech1OryginalPath") == {"a", "b"}
    # Pooled speaker set is the union of both speaker columns.
    assert source_set(df, SPEAKER_COLUMNS) == {"a", "b", "c"}


def test_pairwise_intersections_detect_train_leak_and_val_test_share():
    train = _toy_corpus(["a", "b"], ["c", "d"], ["s1", "s2"], ["e1"])
    val = _toy_corpus(["x"], ["y"], ["s2", "s3"], ["e2"])
    test = _toy_corpus(["b"], ["z"], ["s2", "s4"], ["e3"])

    # train <-> test share speaker source "b" (a real leak).
    assert (source_set(train, "speech1OryginalPath")
            & source_set(test, "speech1OryginalPath")) == {"b"}
    # train <-> val share nothing.
    assert not (source_set(train, "speech1OryginalPath")
                & source_set(val, "speech1OryginalPath"))
    # val <-> test share scene "s2" (informational, not a training leak).
    assert (source_set(val, "sceneOryginalPath")
            & source_set(test, "sceneOryginalPath")) == {"s2"}


def test_pooled_speaker_check_catches_role_swap():
    """A speaker that is speech1 in train but speech2 in test is missed column-wise
    but caught by the pooled speech1 ∪ speech2 set — the reason the pool exists."""
    train = _toy_corpus(["spkA"], ["spkB"], ["s1"], ["e1"])
    test = _toy_corpus(["spkC"], ["spkA"], ["s2"], ["e2"])

    # Column-by-column: no overlap (the trap).
    assert not (source_set(train, "speech1OryginalPath")
                & source_set(test, "speech1OryginalPath"))
    assert not (source_set(train, "speech2OryginalPath")
                & source_set(test, "speech2OryginalPath"))
    # Pooled: the role-swapped speaker is caught.
    assert (source_set(train, SPEAKER_COLUMNS)
            & source_set(test, SPEAKER_COLUMNS)) == {"spkA"}
