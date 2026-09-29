"""Tests for the silence-aware PIT SI-SDR loss (utils/losses.py), its metric
counterpart (utils/metrics.py) and the Trainer wiring around speaker-count
augmentation: loss selection, 1-speaker validation, the checkpoint-overwrite
guard and weights-only initialisation (init_from). CPU-only."""

import math

import pytest
import torch
import torch.nn.functional as F
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from torch.utils.data import DataLoader, Dataset
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from datasets import polsess_collate_fn
from utils.losses import DEFAULT_TAU_DB, SILENT_RMS_DBFS, silence_aware_pit_sisdr
from utils.metrics import compute_silence_aware_sisdr_and_sisdri, compute_sisdr_and_sisdri

from tests.test_trainer import DummyModel, SyntheticDataset, make_config

T = 800


def _two_speaker_batch(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    tgt = torch.randn(B, 2, T, generator=g)
    est = tgt.flip(1) + 0.3 * torch.randn(B, 2, T, generator=g)  # swapped order
    return est, tgt


def _one_speaker_sample(active_slot, speech_out, leak, seed=0):
    """One item: target speech in `active_slot`, zero in the other; the model
    puts the speech (plus a little noise, ~20 dB SI-SDR) in output `speech_out`
    and `leak` * (speech + equal-power noise, ~0 dB SI-SDR) in the other, so the
    speech output is unambiguously the better match."""
    g = torch.Generator().manual_seed(seed)
    speech = torch.randn(T, generator=g)
    tgt = torch.zeros(2, T)
    tgt[active_slot] = speech
    est = torch.zeros(2, T)
    est[speech_out] = speech + 0.1 * torch.randn(T, generator=g)
    est[1 - speech_out] = leak * (speech + torch.randn(T, generator=g))
    return est, tgt


# --- loss ---------------------------------------------------------------------

def test_two_speaker_batch_matches_asteroid_pit():
    est, tgt = _two_speaker_batch()
    ref_loss, ref_reordered = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")(
        est, tgt, return_est=True
    )
    result = silence_aware_pit_sisdr(est, tgt)
    # Same pairwise matrix, same permutation search, same batch mean: bit-identical.
    assert torch.equal(result.loss, ref_loss)
    assert torch.allclose(result.reordered, ref_reordered)
    assert result.active.all()
    assert torch.isnan(result.silent_rel_db).all()
    # Mean SI-SDR over (all active) slots is minus the PIT loss.
    assert torch.allclose(result.si_sdr.mean(), -ref_loss)


def test_one_speaker_loss_finite_and_silent_output_gets_gradient():
    est, tgt = _one_speaker_sample(active_slot=0, speech_out=1, leak=0.5)
    est = est.unsqueeze(0).requires_grad_(True)
    result = silence_aware_pit_sisdr(est, tgt.unsqueeze(0))
    assert torch.isfinite(result.loss)
    result.loss.backward()
    # Output 0 is the one assigned to silence: under plain PIT its gradient is
    # exactly zero; here it must not be.
    assert est.grad[0, 0].abs().sum() > 0


def test_plain_pit_gives_zero_gradient_to_silent_output():
    """Documents the problem the silence-aware loss fixes."""
    est, tgt = _one_speaker_sample(active_slot=0, speech_out=1, leak=0.5)
    est = est.unsqueeze(0).requires_grad_(True)
    loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")(est, tgt.unsqueeze(0))
    loss.backward()
    assert est.grad[0, 0].abs().sum() == 0


@pytest.mark.parametrize("active_slot,speech_out", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_one_speaker_assignment_picks_best_output(active_slot, speech_out):
    est, tgt = _one_speaker_sample(active_slot, speech_out, leak=0.2)
    result = silence_aware_pit_sisdr(est.unsqueeze(0), tgt.unsqueeze(0))
    # The speech output is aligned with the active slot, the leak with the silent one.
    assert torch.equal(result.reordered[0, active_slot], est[speech_out])
    assert torch.equal(result.reordered[0, 1 - active_slot], est[1 - speech_out])
    assert result.active[0].tolist() == [active_slot == 0, active_slot == 1]
    assert math.isnan(result.si_sdr[0, 1 - active_slot].item())
    assert result.si_sdr[0, active_slot] > 10

    # Hand formula: (-SI-SDR(e_i*, t_j) + 10 log10(E_o / E_i* + tau)) / 2
    def zm_energy(x):
        return ((x - x.mean()) ** 2).sum()

    e_best, e_other = est[speech_out], est[1 - speech_out]
    neg_sisdr = pairwise_neg_sisdr(est.unsqueeze(0), tgt.unsqueeze(0))[0, speech_out, active_slot]
    tau = 10 ** (-DEFAULT_TAU_DB / 10)
    expected = (neg_sisdr + 10 * torch.log10(zm_energy(e_other) / zm_energy(e_best) + tau)) / 2
    assert torch.allclose(result.loss, expected, atol=1e-5)
    assert torch.allclose(
        result.silent_rel_db[0],
        10 * torch.log10(zm_energy(e_other) / zm_energy(e_best)),
        atol=1e-4,
    )


def test_silent_penalty_saturates_at_tau():
    est, tgt = _one_speaker_sample(0, 0, leak=0.0)  # perfectly silent other output
    result = silence_aware_pit_sisdr(est.unsqueeze(0), tgt.unsqueeze(0), tau_db=30.0)
    neg_sisdr = -result.si_sdr[0, 0]
    # loss_silent = 10 log10(0 + tau) = -30 dB exactly.
    assert torch.allclose(result.loss, (neg_sisdr - 30.0) / 2, atol=1e-4)

    # Far below tau the penalty barely moves: -60 dB vs -80 dB leak differ by < 0.01 dB.
    losses = []
    for leak in (1e-3, 1e-4):
        e, t = _one_speaker_sample(0, 0, leak=leak)
        r = silence_aware_pit_sisdr(e.unsqueeze(0), t.unsqueeze(0), tau_db=30.0)
        losses.append(r.loss.item())
    assert abs(losses[0] - losses[1]) < 0.01


def test_mixed_batch_is_mean_of_per_sample_losses():
    est2, tgt2 = _two_speaker_batch(B=2)
    est1, tgt1 = _one_speaker_sample(1, 0, leak=0.3)
    est = torch.cat([est2, est1.unsqueeze(0)])
    tgt = torch.cat([tgt2, tgt1.unsqueeze(0)])

    result = silence_aware_pit_sisdr(est, tgt)
    loss_two = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")(est2, tgt2)
    loss_one = silence_aware_pit_sisdr(est1.unsqueeze(0), tgt1.unsqueeze(0)).loss
    assert torch.allclose(result.loss, (2 * loss_two + loss_one) / 3, atol=1e-5)
    assert result.active.sum().item() == 5
    assert torch.isnan(result.silent_rel_db[:2]).all()
    assert torch.isfinite(result.silent_rel_db[2])


def test_all_silent_batch_has_no_loss():
    """Corpus defect rows (both speakers silent) leave nothing to train on."""
    result = silence_aware_pit_sisdr(torch.randn(1, 2, T), torch.zeros(1, 2, T))
    assert result.loss is None
    assert not result.valid.any()


def test_near_silent_target_counts_as_silent():
    """1-LSB dither (~-96 dBFS) and anything up to -60 dBFS is not speech."""
    lsb = 1.0 / 32768
    dither = lsb * torch.randint(-1, 2, (T,)).float()
    assert 10 * torch.log10((dither**2).mean()) < SILENT_RMS_DBFS
    est, tgt = _one_speaker_sample(0, 1, leak=0.1)
    tgt = tgt.clone()
    tgt[1] = dither  # the "removed" slot holds dither instead of exact zeros
    result = silence_aware_pit_sisdr(est.unsqueeze(0), tgt.unsqueeze(0))
    assert result.active[0].tolist() == [True, False]
    assert torch.isfinite(result.silent_rel_db[0])


def test_silent_item_is_left_out_of_batch_mean():
    est1, tgt1 = _one_speaker_sample(0, 1, leak=0.1)
    alone = silence_aware_pit_sisdr(est1.unsqueeze(0), tgt1.unsqueeze(0)).loss
    est = torch.stack([est1, torch.randn(2, T)])
    tgt = torch.stack([tgt1, torch.zeros(2, T)])
    result = silence_aware_pit_sisdr(est, tgt)
    assert result.valid.tolist() == [True, False]
    assert torch.allclose(result.loss, alone)


def test_metrics_on_all_silent_batch():
    metric = ScaleInvariantSignalDistortionRatio(zero_mean=True)
    out = compute_silence_aware_sisdr_and_sisdri(
        torch.randn(1, 2, T), torch.zeros(1, 2, T), torch.randn(1, T), metric
    )
    assert out["n_active_slots"] == 0 and out["si_sdr"] is None


def test_computes_in_fp32_for_bf16_inputs():
    est, tgt = _one_speaker_sample(0, 1, leak=0.1)
    est_bf16 = est.unsqueeze(0).bfloat16().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = silence_aware_pit_sisdr(est_bf16, tgt.unsqueeze(0).bfloat16())
    assert result.loss.dtype == torch.float32
    assert result.reordered.dtype == torch.float32
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert est_bf16.grad is not None and torch.isfinite(est_bf16.grad.float()).all()


# --- metrics ------------------------------------------------------------------

def test_silence_aware_metrics_hand_built_example():
    si_sdr_metric = ScaleInvariantSignalDistortionRatio(zero_mean=True)
    est, tgt = _one_speaker_sample(active_slot=1, speech_out=0, leak=0.1)
    g = torch.Generator().manual_seed(5)
    mix = tgt[1] + 0.5 * torch.randn(T, generator=g)

    out = compute_silence_aware_sisdr_and_sisdri(
        est.unsqueeze(0), tgt.unsqueeze(0), mix.unsqueeze(0), si_sdr_metric
    )
    expected_sisdr = si_sdr_metric(est[0], tgt[1]).item()
    expected_base = si_sdr_metric(mix, tgt[1]).item()
    assert out["n_active_slots"] == 1
    assert out["si_sdr"] == pytest.approx(expected_sisdr, abs=1e-3)
    assert out["si_sdri"] == pytest.approx(expected_sisdr - expected_base, abs=1e-3)
    zm = lambda x: ((x - x.mean()) ** 2).sum().item()  # noqa: E731
    expected_rel = 10 * math.log10(zm(est[1]) / zm(est[0]))
    assert expected_rel == pytest.approx(-17.0, abs=0.5)  # leak 0.1 of ~2x power
    assert out["silent_rel_db"] == pytest.approx(expected_rel, abs=1e-3)
    assert len(out["silent_rel_db_values"]) == 1


def test_silence_aware_metrics_match_default_on_two_speaker_batch():
    si_sdr_metric = ScaleInvariantSignalDistortionRatio(zero_mean=True)
    pit = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    est, tgt = _two_speaker_batch(B=4, seed=3)
    mix = tgt.sum(1) + 0.1 * torch.randn(4, T)
    ref_sisdr, ref_sisdri, _ = compute_sisdr_and_sisdri(est, tgt, mix, "SB", si_sdr_metric, pit)
    out = compute_silence_aware_sisdr_and_sisdri(est, tgt, mix, si_sdr_metric)
    assert out["si_sdr"] == pytest.approx(ref_sisdr, abs=1e-4)
    assert out["si_sdri"] == pytest.approx(ref_sisdri, abs=1e-4)
    assert out["silent_rel_db"] is None


def test_silence_aware_metrics_require_zero_mean():
    est, tgt = _one_speaker_sample(0, 0, leak=0.1)
    with pytest.raises(ValueError, match="zero_mean"):
        compute_silence_aware_sisdr_and_sisdri(
            est.unsqueeze(0), tgt.unsqueeze(0), tgt[0].unsqueeze(0),
            ScaleInvariantSignalDistortionRatio(zero_mean=False),
        )


# --- Trainer wiring -----------------------------------------------------------

class OneSpeakerDataset(Dataset):
    """1-speaker SB items: speech in slot 0, zero in slot 1, mix = speech + noise."""

    def __init__(self, n=4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        speech = torch.randn(256, generator=g)
        clean = torch.stack([speech, torch.zeros(256)])
        mix = speech + 0.3 * torch.randn(256, generator=g)
        return {"mix": mix, "clean": clean, "background_complexity": "S"}


def _sb_trainer(tmp_path, single_speaker_prob=0.0, one_spk_loader=None, model=None):
    from training.trainer import Trainer

    cfg = make_config(tmp_path, task="SB")
    cfg.data.single_speaker_prob = single_speaker_prob
    cfg.data.partial_overlap_prob = 0.0
    loader = DataLoader(SyntheticDataset(4, task="SB"), batch_size=2, collate_fn=polsess_collate_fn)
    val_loader = DataLoader(SyntheticDataset(2, task="SB"), batch_size=2, collate_fn=polsess_collate_fn)
    return Trainer(
        model or DummyModel(C=2), loader, val_loader, cfg,
        device="cpu", logger=None, wandb_logger=None,
        single_speaker_val_loader=one_spk_loader,
    )


def test_default_config_keeps_plain_pit_loss(tmp_path):
    trainer = _sb_trainer(tmp_path)
    assert trainer.silence_aware is False
    assert trainer.loss_fn == trainer._pit_loss_wrapper
    # SimpleNamespace configs without the new fields behave the same.
    from training.trainer import Trainer
    cfg = make_config(tmp_path, task="SB")
    loader = DataLoader(SyntheticDataset(2, task="SB"), batch_size=2, collate_fn=polsess_collate_fn)
    t = Trainer(DummyModel(C=2), loader, loader, cfg, device="cpu")
    assert t.loss_fn == t._pit_loss_wrapper


def test_augmentation_switches_to_silence_aware_loss(tmp_path):
    trainer = _sb_trainer(tmp_path, single_speaker_prob=0.3)
    assert trainer.silence_aware is True
    assert trainer.loss_fn == trainer._silence_aware_loss_wrapper
    est, tgt = _one_speaker_sample(0, 1, leak=0.5)
    loss, sisdr = trainer.loss_fn(est.unsqueeze(0), tgt.unsqueeze(0))
    assert torch.isfinite(loss) and isinstance(sisdr, float)


def test_validate_single_speaker_metrics(tmp_path):
    one_spk_loader = DataLoader(OneSpeakerDataset(4), batch_size=2, collate_fn=polsess_collate_fn)
    trainer = _sb_trainer(tmp_path, single_speaker_prob=0.3, one_spk_loader=one_spk_loader)
    # DummyModel copies the mix into both outputs: every item is "split" (0 dB).
    r = trainer.validate_single_speaker()
    assert r["silent_rel_db"] == pytest.approx(0.0, abs=1e-4)
    assert r["split_frac_10db"] == 1.0
    assert math.isfinite(r["si_sdr"]) and math.isfinite(r["si_sdri"])


def test_single_speaker_val_logged_and_included_in_preflight(tmp_path):
    from tests.test_trainer import _RecordingModel

    one_spk_loader = DataLoader(OneSpeakerDataset(2), batch_size=2, collate_fn=polsess_collate_fn)
    model = _RecordingModel(C=2)
    trainer = _sb_trainer(tmp_path, single_speaker_prob=0.3, one_spk_loader=one_spk_loader, model=model)
    trainer._preflight_validation()
    assert model.calls == [(False, False)] * 2  # 2-spk val + 1-spk val

    logged = []

    class _FakeWandb:
        enabled = False
        run = None

        def log_metrics(self, metrics, step=None):
            logged.append(metrics)

        def log_model(self, path):
            pass

    trainer.wandb_logger = _FakeWandb()
    trainer.train(num_epochs=1, save_dir=str(tmp_path / "ckpt"))
    epoch_metrics = next(m for m in logged if "val_si_sdr" in m)
    for key in ("si_sdr", "si_sdri", "silent_rel_db", "split_frac_10db"):
        assert f"val_1spk_{key}" in epoch_metrics


# --- overwrite guard ----------------------------------------------------------

def _mse(estimates, clean):
    loss = F.mse_loss(estimates, clean)
    return loss, loss.item()


def test_overwrite_guard_raises_for_existing_checkpoint(tmp_path):
    from training.trainer import ExistingCheckpointError

    trainer = _sb_trainer(tmp_path)
    trainer.config.training.wandb_run_name = "my_run"
    save_dir = tmp_path / "ckpt"
    run_dir = save_dir / "convtasnet" / "SB" / "my_run"
    run_dir.mkdir(parents=True)
    (run_dir / "convtasnet_SB_best.pt").write_bytes(b"")

    trainer._preflight_validation = lambda: pytest.fail("guard must run before preflight")
    with pytest.raises(ExistingCheckpointError, match="my_run"):
        trainer.train(num_epochs=1, save_dir=str(save_dir))


def test_overwrite_guard_passes_for_empty_dir_and_resumed_run(tmp_path):
    save_dir = tmp_path / "ckpt"
    # Fresh run into a fresh directory: trains and writes its checkpoint.
    trainer = _sb_trainer(tmp_path)
    trainer.config.training.wandb_run_name = "my_run"
    trainer.loss_fn = _mse
    trainer.train(num_epochs=1, save_dir=str(save_dir))
    ckpt = save_dir / "convtasnet" / "SB" / "my_run" / "convtasnet_SB_best.pt"
    assert ckpt.exists()

    # Resuming that run into the same directory is allowed.
    resumed = _sb_trainer(tmp_path)
    resumed.config.training.wandb_run_name = "my_run"
    resumed.load_checkpoint(str(ckpt))
    assert resumed.resumed is True
    resumed.loss_fn = _mse
    resumed.train(num_epochs=1, save_dir=str(save_dir))


# --- init_from ----------------------------------------------------------------

def _source_checkpoint(tmp_path, compiled_prefix=False):
    src = _sb_trainer(tmp_path)
    with torch.no_grad():
        src.model.p.fill_(0.75)
    src.global_step = 123
    ckpt = src._serialize_checkpoint_data(epoch=9, val_sisdr=11.5)
    if compiled_prefix:
        ckpt["model_state_dict"] = {
            f"_orig_mod.{k}": v for k, v in ckpt["model_state_dict"].items()
        }
    path = tmp_path / ("src_compiled.pt" if compiled_prefix else "src.pt")
    torch.save(ckpt, path)
    return path


@pytest.mark.parametrize("compiled_prefix", [False, True])
def test_init_from_loads_weights_only(tmp_path, compiled_prefix):
    path = _source_checkpoint(tmp_path, compiled_prefix=compiled_prefix)
    trainer = _sb_trainer(tmp_path)
    trainer.init_from_checkpoint(str(path))

    assert trainer.model.p.item() == pytest.approx(0.75)
    # Everything but the weights stays fresh.
    assert trainer.global_step == 0
    assert trainer.current_epoch == 0
    assert trainer.best_val_sisdr == -float("inf")
    assert trainer.epochs_without_improvement == 0
    assert trainer.resumed is False
    assert trainer.optimizer.state_dict()["state"] == {}
    # Provenance: the parent is recorded in this run's checkpoints.
    ckpt = trainer._serialize_checkpoint_data(epoch=0, val_sisdr=1.0)
    assert ckpt["init_from"] == {"path": str(path), "epoch": 9, "best_val_sisdr": 11.5}


def test_init_from_into_compiled_model(tmp_path):
    """A torch.compile-wrapped model exposes _orig_mod; weights go into it."""
    path = _source_checkpoint(tmp_path)

    class _Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

        def forward(self, x):
            return self._orig_mod(x)

    inner = DummyModel(C=2)
    trainer = _sb_trainer(tmp_path, model=_Wrapped(inner))
    trainer.init_from_checkpoint(str(path))
    assert inner.p.item() == pytest.approx(0.75)


def test_init_from_is_strict(tmp_path):
    path = _source_checkpoint(tmp_path)
    ckpt = torch.load(path, weights_only=False)
    ckpt["model_state_dict"]["extra"] = torch.zeros(1)
    torch.save(ckpt, path)
    trainer = _sb_trainer(tmp_path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        trainer.init_from_checkpoint(str(path))


class _SilentRowDataset(SyntheticDataset):
    """SB items; every item with odd idx has both speakers silent (corpus defect)."""

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if idx % 2 == 1:
            item["clean"] = torch.zeros_like(item["clean"])
        return item


def test_train_epoch_skips_batches_without_speech(tmp_path):
    trainer = _sb_trainer(tmp_path, single_speaker_prob=0.3)
    trainer.train_loader = DataLoader(
        _SilentRowDataset(4, task="SB"), batch_size=1, collate_fn=polsess_collate_fn
    )
    sisdr, sisdri = trainer.train_epoch()
    assert trainer.silent_batches_skipped == 2
    assert trainer.consecutive_nan_batches == 0
    assert math.isfinite(sisdr) and math.isfinite(sisdri)
