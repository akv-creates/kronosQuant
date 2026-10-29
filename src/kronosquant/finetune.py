"""Stage 9a: fine-tune the Kronos predictor on the TRAIN split only (the tokenizer stays frozen).

Objective (same as upstream): next-token cross-entropy. A z-scored window of candles is turned into (s1, s2) token
pairs by the frozen tokenizer; the model reads tokens 0..L-2 and must predict tokens 1..L-1.
This is a likelihood objective. Lower loss does not mean a better trading signal.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kronosquant.forecast import FEATURES

TIME_FEATURES = ["minute", "hour", "weekday", "day", "month"]  # same order the upstream predictor uses


def prepare_arrays(candles: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(rows, 6) candle features and (rows, 5) calendar features as float32."""
    ts = candles["timestamps"]
    stamps = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], axis=1)
    return candles[FEATURES].to_numpy(np.float32), stamps.astype(np.float32)


def window_batch(values: np.ndarray, stamps: np.ndarray, starts: list[int], length: int, clip: float = 5.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Each window is z-scored with its OWN mean/std and clipped, exactly like KronosPredictor.predict does for its history."""
    xs, ss = [], []
    for s in starts:
        x = values[s : s + length]
        x = np.clip((x - x.mean(axis=0)) / (x.std(axis=0) + 1e-5), -clip, clip)
        xs.append(x)
        ss.append(stamps[s : s + length])
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ss))


def window_starts(rows: range, length: int, stride: int) -> list[int]:
    """Start rows of windows lying completely inside `rows`."""
    return list(range(rows.start, rows.stop - length + 1, stride))


def fit_tune_ranges(train: range, length: int, tune_frac: float = 0.2) -> tuple[range, range]:
    """Split the train range in time: the first part trains, the last part picks the checkpoint.
    A gap of `length` rows separates them so no window straddles both."""
    cut = train.start + int(len(train) * (1 - tune_frac))
    return range(train.start, cut), range(cut + length, train.stop)


def token_loss(model, tokenizer, x: torch.Tensor, stamp: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Mean next-token cross-entropy over s1 and s2 for a batch of windows."""
    with torch.no_grad():
        s1, s2 = tokenizer.encode(x, half=True)
    logits = model(s1[:, :-1], s2[:, :-1], stamp[:, :-1, :])
    loss, ce1, ce2 = model.head.compute_loss(logits[0], logits[1], s1[:, 1:], s2[:, 1:])
    return loss, ce1.item(), ce2.item()


def evaluate_loss(model, tokenizer, values, stamps, starts, length, device, batch_size=16, seed=0) -> float:
    """Token loss over fixed windows. The model samples s1 inside forward(), so the seed is fixed to make this repeatable."""
    model.eval()
    torch.manual_seed(seed)
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            chunk = starts[i : i + batch_size]
            x, st = window_batch(values, stamps, chunk, length)
            loss, _, _ = token_loss(model, tokenizer, x.to(device), st.to(device))
            total += float(loss) * len(chunk)
            count += len(chunk)
    return total / count


def fine_tune(model, tokenizer, values, stamps, fit: range, tune: range, *, length: int, batch_size: int, lr: float, steps: int,
              eval_every: int, seed: int, device: str, out_dir: Path, log=print) -> dict:
    """Train on random windows of `fit`; every `eval_every` steps measure the loss on `tune` windows and keep the best checkpoint."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    fit_starts = window_starts(fit, length, 1)
    tune_starts = window_starts(tune, length, length // 4)  # overlapping windows: more of them, but they are correlated
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = [{"step": 0, "tune_loss": evaluate_loss(model, tokenizer, values, stamps, tune_starts, length, device)}]
    log(f"step 0  tune loss {history[0]['tune_loss']:.4f}  (pretrained model; {len(fit_starts)} fit windows, {len(tune_starts)} tune windows)")
    best = history[0]["tune_loss"]
    model.save_pretrained(out_dir / "best")  # step 0 = the pretrained model, so "best" is never worse than not tuning
    best_step, t0, run_loss = 0, time.time(), []
    for step in range(1, steps + 1):
        model.train()
        starts = rng.choice(fit_starts, size=batch_size, replace=False).tolist()
        x, st = window_batch(values, stamps, starts, length)
        loss, _, _ = token_loss(model, tokenizer, x.to(device), st.to(device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        opt.step()
        sched.step()
        run_loss.append(loss.item())
        if step % eval_every == 0 or step == steps:
            tune_loss = evaluate_loss(model, tokenizer, values, stamps, tune_starts, length, device)
            history.append({"step": step, "train_loss": float(np.mean(run_loss)), "tune_loss": tune_loss})
            run_loss = []
            if tune_loss < best:
                best, best_step = tune_loss, step
                model.save_pretrained(out_dir / "best")
            log(f"step {step}  train {history[-1]['train_loss']:.4f}  tune {tune_loss:.4f}  best {best:.4f}@{best_step}  {(time.time() - t0) / step:.1f}s/step")
    result = {"best_tune_loss": best, "best_step": best_step, "pretrained_tune_loss": history[0]["tune_loss"], "history": history}
    (out_dir / "history.json").write_text(json.dumps(result, indent=2))
    return result
