import numpy as np
import pandas as pd
import pytest
import torch

import kronosquant.forecast  # noqa: F401  (puts vendor/Kronos on sys.path)
from kronosquant.finetune import fine_tune, fit_tune_ranges, prepare_arrays, token_loss, window_batch, window_starts
from model import Kronos, KronosTokenizer


def candles(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {"timestamps": pd.date_range("2024-01-01", periods=n, freq="1h"), "open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": rng.uniform(1, 2, n), "amount": rng.uniform(1, 2, n)}
    )


def tiny_models():
    torch.manual_seed(0)
    tok = KronosTokenizer(d_in=6, d_model=16, n_heads=2, ff_dim=32, n_enc_layers=1, n_dec_layers=1, ffn_dropout_p=0.0, attn_dropout_p=0.0,
                          resid_dropout_p=0.0, s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4)
    model = Kronos(s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=2, ff_dim=64, ffn_dropout_p=0.0, attn_dropout_p=0.0,
                   resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True)
    return model.eval(), tok.eval()


def test_window_batch_is_zscored_per_window_and_uses_only_its_own_rows():
    values, stamps = prepare_arrays(candles())
    poisoned = values.copy()
    poisoned[:100] = 1e9  # rows before the window must not matter
    poisoned[200:] = -1e9  # nor rows after it
    x, st = window_batch(values, stamps, [100], 100)
    xp, _ = window_batch(poisoned, stamps, [100], 100)
    assert torch.allclose(x, xp, atol=1e-4) and x.shape == (1, 100, 6) and st.shape == (1, 100, 5)
    assert x.abs().max() <= 5.0 and abs(float(x[0, :, 3].mean())) < 1e-5


def test_fit_and_tune_ranges_never_share_a_window():
    fit, tune = fit_tune_ranges(range(0, 1000), length=50, tune_frac=0.2)
    assert fit.stop + 50 == tune.start and tune.stop == 1000
    last_fit_window_end = window_starts(fit, 50, 1)[-1] + 50
    first_tune_window_start = window_starts(tune, 50, 25)[0]
    assert last_fit_window_end <= fit.stop < first_tune_window_start


def test_training_lowers_loss_on_a_tiny_model_and_keeps_best_checkpoint(tmp_path):
    values, stamps = prepare_arrays(candles(600))
    model, tok = tiny_models()
    fit, tune = range(0, 450), range(480, 600)
    result = fine_tune(model, tok, values, stamps, fit, tune, length=32, batch_size=8, lr=3e-3, steps=60, eval_every=20, seed=1,
                       device="cpu", out_dir=tmp_path, log=lambda m: None)
    assert result["best_tune_loss"] <= result["pretrained_tune_loss"]
    assert result["best_step"] > 0  # on this task training does help
    assert (tmp_path / "best" / "model.safetensors").exists() and (tmp_path / "history.json").exists()


def test_loss_is_finite_and_roughly_log_vocab_for_an_untrained_model():
    values, stamps = prepare_arrays(candles())
    model, tok = tiny_models()
    x, st = window_batch(values, stamps, [0, 50], 64)
    loss, ce1, ce2 = token_loss(model, tok, x, st)
    assert np.isfinite(float(loss))
    assert ce1 == pytest.approx(np.log(16), rel=0.5) and ce2 == pytest.approx(np.log(16), rel=0.5)


def test_checkpoint_changes_the_cache_key_and_pretrained_key_is_unchanged(tmp_path):
    from kronosquant.cache import config_key
    from kronosquant.forecast import forecast_config, load_sample_candles

    c = load_sample_candles()
    base = forecast_config(c, "cpu", 400, 12, 1.0, 0.9, 5)
    assert "finetuned_weights_sha256" not in base  # old caches keep their keys
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "model.safetensors").write_bytes(b"one")
    (tmp_path / "b" / "model.safetensors").write_bytes(b"two")
    ka = config_key(forecast_config(c, "cpu", 400, 12, 1.0, 0.9, 5, checkpoint_dir=tmp_path / "a"))
    kb = config_key(forecast_config(c, "cpu", 400, 12, 1.0, 0.9, 5, checkpoint_dir=tmp_path / "b"))
    assert len({config_key(base), ka, kb}) == 3
