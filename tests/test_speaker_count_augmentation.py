"""Tests for SB speaker-count augmentation in PolSESSDataset and training/setup.py.

A tiny synthetic PolSESS corpus is written as float32 WAVs whose mix file is
the exact sum of its layer files, so every expected mixture can be rebuilt
from the layers and compared to the dataset output (CPU-only, no real corpus).
"""

import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from datasets.polsess_dataset import PolSESSDataset

SR = 8000
T = 8000  # 1 s clips
CORPUS = "PolSESS_SC"
N_ROWS = 8  # alternating indoor (even idx) / outdoor (odd idx)


def _write(path, signal):
    sf.write(str(path), signal, SR, subtype="FLOAT")


@pytest.fixture
def corpus(tmp_path):
    """Synthetic corpus; returns (data_root, layers[subset][idx] -> dict of np arrays)."""
    data_root = tmp_path / CORPUS
    rng = np.random.default_rng(0)
    layers = {}
    for subset in ("train", "val"):
        base = data_root / subset
        for folder in ("mix", "clean", "scene", "event", "sp1_reverb", "sp2_reverb", "ev_reverb"):
            (base / folder).mkdir(parents=True, exist_ok=True)
        rows, subset_layers = [], []
        for i in range(N_ROWS):
            indoor = i % 2 == 0
            # Positive dry speakers so g = target / dry is well defined in tests.
            lay = {
                "speaker1": (0.05 + 0.05 * rng.random(T)).astype(np.float32),
                "speaker2": (0.05 + 0.05 * rng.random(T)).astype(np.float32),
                "scene": (0.05 * rng.standard_normal(T)).astype(np.float32),
                "event": (0.05 * rng.standard_normal(T)).astype(np.float32),
            }
            if indoor:
                for key in ("sp1_reverb", "sp2_reverb", "ev_reverb"):
                    lay[key] = (0.02 * rng.standard_normal(T)).astype(np.float32)
            lay["mix"] = sum(lay.values()).astype(np.float32)

            row = {
                "mixFile": f"mix_{i}.wav",
                "speaker1File": f"s1_{i}.wav",
                "speaker2File": f"s2_{i}.wav",
                "sceneFile": f"scene_{i}.wav",
                "eventFile": f"event_{i}.wav",
                "reverbForSpeaker1": f"r1_{i}.wav" if indoor else None,
                "reverbForSpeaker2": f"r2_{i}.wav" if indoor else None,
                "reverbForEvent": f"rev_{i}.wav" if indoor else None,
            }
            _write(base / "mix" / row["mixFile"], lay["mix"])
            _write(base / "clean" / row["speaker1File"], lay["speaker1"])
            _write(base / "clean" / row["speaker2File"], lay["speaker2"])
            _write(base / "scene" / row["sceneFile"], lay["scene"])
            _write(base / "event" / row["eventFile"], lay["event"])
            if indoor:
                _write(base / "sp1_reverb" / row["reverbForSpeaker1"], lay["sp1_reverb"])
                _write(base / "sp2_reverb" / row["reverbForSpeaker2"], lay["sp2_reverb"])
                _write(base / "ev_reverb" / row["reverbForEvent"], lay["ev_reverb"])
            rows.append(row)
            subset_layers.append({k: torch.from_numpy(v) for k, v in lay.items()})
        pd.DataFrame(rows).to_csv(base / f"corpus_{CORPUS}_{subset}_final.csv", index=False)
        layers[subset] = subset_layers
    return data_root, layers


def _variant_mix(lay, variant, indoor):
    """Expected 2-speaker SB mix for a variant, rebuilt from the layer files."""
    mix = lay["speaker1"] + lay["speaker2"]
    if indoor and variant != "C":
        mix = mix + lay["sp1_reverb"] + lay["sp2_reverb"]
    if "S" in variant:
        mix = mix + lay["scene"]
    if "E" in variant:
        mix = mix + lay["event"]
        if indoor:
            mix = mix + lay["ev_reverb"]
    return mix


def _contribution(lay, k, variant, indoor):
    """Speaker k's layers still present in the variant mix."""
    contrib = lay[f"speaker{k}"]
    if indoor and variant != "C":
        contrib = contrib + lay[f"sp{k}_reverb"]
    return contrib


def _layers_of(ds, i, subset_layers):
    """Layers of dataset item i. A single allowed variant filters the metadata
    to one reverb condition, so the row is matched by its mix file name."""
    mix_file = ds.metadata.iloc[i]["mixFile"]  # "mix_<row>.wav"
    return subset_layers[int(mix_file[len("mix_"):-len(".wav")])]


def _dropped_slot(clean):
    """Return k in {1, 2} of the all-zero target slot (asserting exactly one)."""
    zero = [bool(torch.all(clean[j] == 0)) for j in range(2)]
    assert sum(zero) == 1, f"expected exactly one all-zero slot, got {zero}"
    return zero.index(True) + 1


# ---------------------------------------------------------------------------
# Defaults: behaviour and RNG stream unchanged
# ---------------------------------------------------------------------------

def test_defaults_match_plain_mmipc_and_consume_no_extra_randomness(corpus):
    data_root, _ = corpus
    ds = PolSESSDataset(data_root, subset="train", task="SB", sample_rate=SR)

    for idx in (0, 1):
        random.seed(123)
        item = ds[idx]
        after_item = random.random()

        # Control: the only draw a default training item may make is the variant.
        random.seed(123)
        row = ds.metadata.iloc[idx]
        has_reverb = pd.notna(row["reverbForSpeaker1"])
        variant = ds._choose_variant(has_reverb, idx)
        after_control = random.random()
        assert after_item == after_control
        assert item["background_complexity"] == variant

        audio = ds._lazy_load(ds._build_paths(row, has_reverb), variant, has_reverb)
        assert torch.equal(item["mix"], ds._apply_mmipc(audio, has_reverb))
        assert torch.equal(item["clean"], ds._compute_clean(audio))


def test_val_defaults_consume_no_global_randomness(corpus):
    data_root, _ = corpus
    ds = PolSESSDataset(data_root, subset="val", task="SB")
    random.seed(7)
    ds[0]
    ds[1]
    got = random.random()
    random.seed(7)
    assert got == random.random()


# ---------------------------------------------------------------------------
# single: one speaker cancelled from mix, its slot all-zero
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant,idx", [
    ("SER", 0), ("SR", 0), ("R", 0), ("C", 0),   # indoor
    ("SE", 1), ("S", 1), ("C", 1),               # outdoor
])
def test_single_speaker_mix_and_target(corpus, variant, idx):
    data_root, layers = corpus
    ds = PolSESSDataset(
        data_root, subset="train", task="SB", allowed_variants=[variant],
        single_speaker_prob=1.0,
    )
    indoor = idx == 0
    seen = set()
    for seed in range(20):
        random.seed(seed)
        item = ds[0]
        lay = _layers_of(ds, 0, layers["train"])

        k = _dropped_slot(item["clean"])
        seen.add(k)
        other = 3 - k
        assert item["clean"].shape == (2, T)
        assert item["clean"].dtype == torch.float32
        assert torch.equal(item["clean"][other - 1], lay[f"speaker{other}"])

        expected = _variant_mix(lay, variant, indoor) - _contribution(lay, k, variant, indoor)
        torch.testing.assert_close(item["mix"], expected, atol=1e-5, rtol=0)
    assert seen == {1, 2}, "dropped speaker should vary across draws"


# ---------------------------------------------------------------------------
# partial: envelope properties and gated algebra
# ---------------------------------------------------------------------------

def test_partial_envelope_properties(corpus):
    data_root, _ = corpus
    ds = PolSESSDataset(data_root, subset="train", task="SB", partial_overlap_prob=1.0)
    fade = round(ds.PARTIAL_FADE_SECONDS * SR)
    onsets = 0
    for seed in range(50):
        g = ds._partial_envelope(T, SR, random.Random(seed))
        assert g.shape == (T,)
        assert torch.all(g >= 0) and torch.all(g <= 1)
        in_fade = (g > 0) & (g < 1)
        assert int(in_fade.sum()) == fade
        active = int((g > 0).sum())
        assert 0.25 * T <= active <= 0.75 * T

        # One contiguous active segment at one end of the clip.
        if g[0] > 0:
            assert torch.all(g[:active] > 0) and torch.all(g[active:] == 0)
            assert torch.all(g[1:] <= g[:-1])  # monotone decreasing (offset)
        else:
            onsets += 1
            assert torch.all(g[: T - active] == 0) and torch.all(g[T - active:] > 0)
            assert torch.all(g[1:] >= g[:-1])  # monotone increasing (onset)
    assert 0 < onsets < 50, "both onsets and offsets should occur"


@pytest.mark.parametrize("variant,idx", [("SER", 0), ("ER", 0), ("C", 0), ("SE", 1), ("E", 1)])
def test_partial_mix_and_target(corpus, variant, idx):
    data_root, layers = corpus
    ds = PolSESSDataset(
        data_root, subset="train", task="SB", allowed_variants=[variant],
        partial_overlap_prob=1.0, sample_rate=SR,
    )
    indoor = idx == 0
    for seed in range(10):
        random.seed(seed)
        item = ds[0]
        lay = _layers_of(ds, 0, layers["train"])

        # The untouched slot equals its dry file; the other is gated.
        untouched = [torch.equal(item["clean"][j], lay[f"speaker{j + 1}"]) for j in range(2)]
        assert sum(untouched) == 1
        k = 2 if untouched[0] else 1

        g = item["clean"][k - 1] / lay[f"speaker{k}"]
        assert torch.all(g >= -1e-6) and torch.all(g <= 1 + 1e-6)
        active = int((g > 1e-6).sum())
        assert 0.25 * T <= active <= 0.75 * T

        expected = (_variant_mix(lay, variant, indoor)
                    - (1 - g) * _contribution(lay, k, variant, indoor))
        torch.testing.assert_close(item["mix"], expected, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# val determinism, task guard
# ---------------------------------------------------------------------------

def test_val_speaker_count_is_deterministic_and_varies(corpus):
    data_root, _ = corpus
    kwargs = dict(subset="val", task="SB", single_speaker_prob=0.5, partial_overlap_prob=0.5)
    a = PolSESSDataset(data_root, **kwargs)
    b = PolSESSDataset(data_root, **kwargs)
    for idx in range(len(a)):
        random.seed(idx)  # the global stream must not matter on val
        item_a = a[idx]
        random.seed(idx + 1000)
        item_b = b[idx]
        assert torch.equal(item_a["mix"], item_b["mix"])
        assert torch.equal(item_a["clean"], item_b["clean"])
        assert item_a["background_complexity"] == item_b["background_complexity"]

    single = PolSESSDataset(data_root, subset="val", task="SB", single_speaker_prob=1.0)
    dropped = {_dropped_slot(single[idx]["clean"]) for idx in range(len(single))}
    assert dropped == {1, 2}


@pytest.mark.parametrize("task", ["ES", "EB"])
def test_non_sb_task_rejected(corpus, task):
    data_root, _ = corpus
    with pytest.raises(ValueError, match="task='SB'"):
        PolSESSDataset(data_root, subset="train", task=task, single_speaker_prob=0.1)
    with pytest.raises(ValueError, match="task='SB'"):
        PolSESSDataset(data_root, subset="train", task=task, partial_overlap_prob=0.1)
    # Off (the default) stays valid for every task.
    PolSESSDataset(data_root, subset="train", task=task)


def test_invalid_probabilities_rejected(corpus):
    data_root, _ = corpus
    with pytest.raises(ValueError, match="sum to <= 1"):
        PolSESSDataset(data_root, task="SB", single_speaker_prob=0.6, partial_overlap_prob=0.6)
    with pytest.raises(ValueError, match="sum to <= 1"):
        PolSESSDataset(data_root, task="SB", single_speaker_prob=-0.1)


# ---------------------------------------------------------------------------
# training/setup.py
# ---------------------------------------------------------------------------

def _setup_config(data_root, single=0.0, partial=0.0):
    return SimpleNamespace(
        data=SimpleNamespace(
            dataset_type="polsess",
            polsess=SimpleNamespace(data_root=str(data_root)),
            task="SB",
            batch_size=3,
            num_workers=0,
            prefetch_factor=2,
            val_max_samples=None,
            sample_rate=SR,
            single_speaker_prob=single,
            partial_overlap_prob=partial,
        ),
        training=SimpleNamespace(validation_variants=None),
    )


def test_single_speaker_val_loader_none_when_off(corpus):
    from training.setup import build_single_speaker_val_loader
    data_root, _ = corpus
    assert build_single_speaker_val_loader(_setup_config(data_root)) is None
    # Stub configs without the fields count as "off" too.
    cfg = _setup_config(data_root)
    del cfg.data.single_speaker_prob, cfg.data.partial_overlap_prob
    assert build_single_speaker_val_loader(cfg) is None


@pytest.mark.parametrize("single,partial", [(0.3, 0.0), (0.0, 0.2)])
def test_single_speaker_val_loader_items(corpus, single, partial):
    from training.setup import build_single_speaker_val_loader
    data_root, _ = corpus
    loader = build_single_speaker_val_loader(_setup_config(data_root, single, partial))
    assert loader is not None
    assert loader.dataset.subset == "val"
    assert len(loader.dataset) == N_ROWS
    variants = set()
    for batch in loader:
        for clean in batch["clean"]:
            _dropped_slot(clean)
        variants.update(batch["background_complexity"])
    assert len(variants) > 1  # normal per-item variant choice, not a fixed one
