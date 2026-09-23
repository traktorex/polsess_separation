"""Tests for the `data.sample_rate` field (added 2026-09-07 for 16 kHz PolSESS).

The loaders are rate-agnostic; the field is provenance + a guard. Covered:
config default / YAML round-trip / legacy-dict fallback / sweep override,
the PolSESSDataset header check, its wiring through build_dataloaders, and
the evaluate.py helpers that derive PESQ mode + STOI rate from a checkpoint.
"""

import math

import pandas as pd
import pytest
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from config import (
    Config,
    DataConfig,
    PolSESSParams,
    load_config_from_dict,
    load_config_from_yaml,
    load_config_for_run,
    save_config_to_yaml,
)
from datasets.polsess_dataset import PolSESSDataset, polsess_collate_fn
from evaluate import (
    LEGACY_SAMPLE_RATE,
    checkpoint_sample_rate,
    evaluate_model,
    pesq_mode_for,
)
from training.setup import build_dataloaders


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_corpus(root, sample_rate, duration_s=1.0):
    """Minimal PolSESS render (one outdoor row per split) stored at `sample_rate`."""
    n = int(sample_rate * duration_s)
    for subset in ("train", "val", "test"):
        for folder in ("mix", "clean", "scene", "event"):
            (root / subset / folder).mkdir(parents=True, exist_ok=True)
        files = {
            "mix": ("mix.wav", 0.9),
            "clean/sp1": ("sp1.wav", 0.15),
            "clean/sp2": ("sp2.wav", 0.25),
            "scene": ("scene.wav", 0.2),
            "event": ("event.wav", 0.3),
        }
        for key, (name, value) in files.items():
            folder = key.split("/")[0]
            torchaudio.save(str(root / subset / folder / name), torch.ones(1, n) * value, sample_rate)
        pd.DataFrame([{
            "mixFile": "mix.wav",
            "speaker1File": "sp1.wav",
            "speaker2File": "sp2.wav",
            "sceneFile": "scene.wav",
            "eventFile": "event.wav",
            "reverbForSpeaker1": None,
            "reverbForSpeaker2": None,
            "reverbForEvent": None,
        }]).to_csv(root / subset / f"corpus_{root.name}_{subset}_final.csv", index=False)
    return root


@pytest.fixture
def corpus_16k(tmp_path):
    return _make_corpus(tmp_path / "PolSESS_16k", 16000)


@pytest.fixture
def corpus_8k(tmp_path):
    return _make_corpus(tmp_path / "PolSESS_8k", 8000)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_default_is_8k():
    assert DataConfig().sample_rate == 8000
    assert Config().data.sample_rate == 8000


def test_yaml_round_trip(tmp_path):
    config = Config()
    config.data.sample_rate = 16000
    path = tmp_path / "cfg.yaml"
    save_config_to_yaml(config, str(path))
    assert load_config_from_yaml(str(path)).data.sample_rate == 16000


def test_legacy_checkpoint_config_without_field_loads_as_8k():
    """Every checkpoint saved before the field existed is 8 kHz."""
    legacy = {"data": {"task": "SB", "batch_size": 2}, "model": {"model_type": "dprnn"}, "training": {}}
    assert load_config_from_dict(legacy).data.sample_rate == 8000


def test_sweep_override(tmp_path):
    base = tmp_path / "base.yaml"
    save_config_to_yaml(Config(), str(base))

    class FakeWandbConfig:
        """wandb.config look-alike: `in` and attribute access, no dict access."""
        def __init__(self, d):
            self._d = d

        def __contains__(self, key):
            return key in self._d

        def get(self, key, default=None):
            return self._d.get(key, default)

        def __getattr__(self, key):
            try:
                return self.__dict__["_d"][key]
            except KeyError:
                raise AttributeError(key)

    config = load_config_for_run(FakeWandbConfig({"config": str(base), "sample_rate": 16000}))
    assert config.data.sample_rate == 16000


# ---------------------------------------------------------------------------
# Dataset guard
# ---------------------------------------------------------------------------

def test_dataset_accepts_matching_rate(corpus_16k):
    ds = PolSESSDataset(corpus_16k, subset="train", task="SB", sample_rate=16000)
    assert len(ds) == 1
    assert ds[0]["mix"].shape[-1] == 16000  # files returned as stored, no resampling


def test_dataset_rejects_mismatched_rate(corpus_16k):
    with pytest.raises(ValueError, match="16000 Hz.*sample_rate=8000"):
        PolSESSDataset(corpus_16k, subset="train", task="SB", sample_rate=8000)


def test_dataset_skips_check_when_unset(corpus_16k):
    """sample_rate=None keeps the pre-field behaviour: load whatever is there."""
    ds = PolSESSDataset(corpus_16k, subset="train", task="SB")
    assert ds.sample_rate is None
    assert len(ds) == 1


def test_guard_fires_through_build_dataloaders(corpus_16k):
    """A 8 kHz config pointed at a 16 kHz corpus fails at setup, before training."""
    config = Config()
    config.data.polsess = PolSESSParams(data_root=str(corpus_16k))
    config.data.num_workers = 0
    config.data.sample_rate = 8000
    with pytest.raises(ValueError, match="16000 Hz"):
        build_dataloaders(config, summary_info={})


def test_build_dataloaders_with_matching_rate(corpus_8k):
    config = Config()
    config.data.polsess = PolSESSParams(data_root=str(corpus_8k))
    config.data.num_workers = 0
    config.data.sample_rate = 8000
    train_loader, val_loader, _ = build_dataloaders(config, summary_info={})
    assert len(train_loader.dataset) == 1
    assert len(val_loader.dataset) == 1


# ---------------------------------------------------------------------------
# evaluate.py helpers
# ---------------------------------------------------------------------------

def test_pesq_mode_for():
    assert pesq_mode_for(8000) == "nb"
    assert pesq_mode_for(16000) == "wb"
    with pytest.raises(ValueError):
        pesq_mode_for(44100)


def test_checkpoint_sample_rate():
    assert LEGACY_SAMPLE_RATE == 8000
    assert checkpoint_sample_rate({}) == 8000
    assert checkpoint_sample_rate({"config": None}) == 8000
    assert checkpoint_sample_rate({"config": {"data": {"task": "SB"}}}) == 8000
    assert checkpoint_sample_rate({"config": {"data": {"sample_rate": 16000}}}) == 16000


class _Identity(nn.Module):
    def forward(self, x):
        return x


def _noisy_loader(sample_rate, n_items=2):
    g = torch.Generator().manual_seed(0)
    items = []
    for _ in range(n_items):
        t = torch.linspace(0, 1, sample_rate)
        clean = 0.5 * torch.sin(2 * math.pi * 220 * t) * (1 + 0.3 * torch.sin(2 * math.pi * 3 * t))
        mix = clean + 0.05 * torch.randn(sample_rate, generator=g)
        items.append({"mix": mix, "clean": clean, "background_complexity": "C"})
    return DataLoader(items, batch_size=1, shuffle=False, collate_fn=polsess_collate_fn)


@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_evaluate_model_scores_stoi_at_rate(sample_rate):
    """STOI must be constructed at the audio's own rate; identity model, ES task."""
    result = evaluate_model(
        _Identity(), _noisy_loader(sample_rate), device="cpu",
        compute_pesq=False, compute_stoi=True, task="ES", sample_rate=sample_rate,
    )
    assert result["stoi"] is not None and 0.0 <= result["stoi"] <= 1.0
    assert result["per_sample"][0]["stoi"] is not None


def test_evaluate_model_pesq_wideband_at_16k():
    """PESQ at 16 kHz runs in wideband mode (nb would be the wrong ITU curve)."""
    result = evaluate_model(
        _Identity(), _noisy_loader(16000), device="cpu",
        compute_pesq=True, compute_stoi=False, task="ES", sample_rate=16000,
    )
    assert result["pesq"] is not None and 1.0 <= result["pesq"] <= 4.7
