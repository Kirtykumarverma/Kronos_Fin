"""Forecast OHLCV candles with a pretrained Kronos model.

Example:
    python run_prediction.py --csv tests/data/regression_input.csv --lookback 400 --pred-len 120

The CSV needs columns: timestamps, open, high, low, close (volume/amount optional).
Pretrained weights are downloaded from Hugging Face (NeoQuasar/*) on first run.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save plots to file instead of opening a window
import matplotlib.pyplot as plt
import pandas as pd

from model import Kronos, KronosPredictor, KronosTokenizer

MODELS = {
    "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="tests/data/regression_input.csv", help="input K-line CSV")
    p.add_argument("--model", choices=MODELS, default="small")
    p.add_argument("--lookback", type=int, default=400, help="number of historical candles used as context")
    p.add_argument("--pred-len", type=int, default=120, help="number of candles to forecast")
    p.add_argument("--start", type=int, default=0, help="row index where the lookback window starts")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--samples", type=int, default=1, help="number of sampled paths to average")
    p.add_argument("--device", default=None, help="cpu / cuda:0 / mps (auto-detected if omitted)")
    p.add_argument("--out-dir", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    model_id, tokenizer_id, max_context = MODELS[args.model]
    if args.lookback > max_context:
        raise SystemExit(f"--lookback {args.lookback} exceeds max context {max_context} for Kronos-{args.model}")

    df = pd.read_csv(args.csv)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    end = args.start + args.lookback
    if end > len(df):
        raise SystemExit(f"CSV has {len(df)} rows; need at least {end} for this lookback")

    cols = [c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df.columns]
    x_df = df.iloc[args.start:end][cols].reset_index(drop=True)
    x_ts = df.iloc[args.start:end]["timestamps"].reset_index(drop=True)

    # Use real future timestamps if present (backtest), otherwise extrapolate the bar interval.
    future = df.iloc[end:end + args.pred_len]
    if len(future) == args.pred_len:
        y_ts = future["timestamps"].reset_index(drop=True)
    else:
        step = x_ts.diff().median()
        y_ts = pd.Series(pd.date_range(x_ts.iloc[-1] + step, periods=args.pred_len, freq=step))
        future = None

    print(f"Loading {model_id} + {tokenizer_id} ...")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(model_id)
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=max_context)

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=args.pred_len,
        T=args.temperature, top_p=args.top_p, sample_count=args.samples, verbose=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.csv).stem
    csv_path = out_dir / f"{stem}_forecast.csv"
    pred_df.to_csv(csv_path, index_label="timestamps")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, col in zip(axes, ["close", "volume"]):
        if col not in x_df:
            continue
        ax.plot(x_ts, x_df[col], color="blue", label="History")
        if future is not None:
            ax.plot(y_ts, future[col].values, color="green", label="Ground truth")
        ax.plot(y_ts, pred_df[col].values, color="red", label="Forecast")
        ax.set_ylabel(col.capitalize())
        ax.legend(loc="upper left")
        ax.grid(True)
    plt.tight_layout()
    png_path = out_dir / f"{stem}_forecast.png"
    fig.savefig(png_path, dpi=120)

    print(pred_df.head())
    print(f"Saved forecast to {csv_path} and plot to {png_path}")


if __name__ == "__main__":
    main()
