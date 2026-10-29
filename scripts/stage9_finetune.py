"""Stage 9a: fine-tune Kronos-small's predictor on the BTC 1h TRAIN split. Evaluation is a separate script.

DECLARED BEFORE TRAINING (do not edit after seeing results):
  model:      Kronos-small (pinned revision) predictor trained; Kronos-Tokenizer-base (pinned) frozen.
  data:       BTCUSDT 1h, chronological split 60/20/20. Only the TRAIN rows are read: the first 80% of them fit the model,
              the last 20% (after a gap of one window length) choose the checkpoint. Validation and test rows are not used.
  windows:    length 412 rows (400 history + 12), random starts, each z-scored with its own mean/std, clipped to +-5.
  optimiser:  AdamW (betas 0.9/0.95, weight decay 0.1, grad-clip 3.0), cosine learning-rate decay, one configuration only.
  checkpoint: lowest tune-set token loss; step 0 (the untouched pretrained model) counts, so "best" is never worse than not tuning.
  tune loss is a likelihood diagnostic. It is NOT evidence of forecasting skill.
  success is decided later, on the validation split, by the same rule as the Stage 3 grid (IC, 99% bootstrap CI lower bound > 0);
  only if that passes is the test split touched, once.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from kronosquant.data import load_candles  # noqa: E402
from kronosquant.finetune import fine_tune, fit_tune_ranges, prepare_arrays  # noqa: E402
from kronosquant.forecast import MODEL_ID, MODEL_REVISION, ROOT, TOKENIZER_ID, TOKENIZER_REVISION  # noqa: E402
from kronosquant.splits import chronological_split  # noqa: E402
from model import Kronos, KronosTokenizer  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="data/BTCUSDT_1h.csv")
parser.add_argument("--length", type=int, default=412)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--eval-every", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cpu")
parser.add_argument("--out", default="outputs/finetune/btc1h_v1")
args = parser.parse_args()

candles = load_candles(ROOT / args.csv)
train = chronological_split(len(candles))["train"]
fit, tune = fit_tune_ranges(train, args.length)
values, stamps = prepare_arrays(candles.iloc[: train.stop])  # slice: later rows cannot be read even by mistake
print(f"train rows {train.start}..{train.stop - 1}; fit {fit.start}..{fit.stop - 1}; tune {tune.start}..{tune.stop - 1}; args {vars(args)}", flush=True)

tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION).to(args.device).eval()
for p in tokenizer.parameters():
    p.requires_grad_(False)
model = Kronos.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(args.device)
print(f"predictor parameters: {sum(p.numel() for p in model.parameters()):,}; torch threads {torch.get_num_threads()}", flush=True)

result = fine_tune(model, tokenizer, values, stamps, fit, tune, length=args.length, batch_size=args.batch_size, lr=args.lr, steps=args.steps,
                   eval_every=args.eval_every, seed=args.seed, device=args.device, out_dir=ROOT / args.out, log=lambda m: print(m, flush=True))
print(f"\npretrained tune loss {result['pretrained_tune_loss']:.4f} -> best {result['best_tune_loss']:.4f} at step {result['best_step']}")
