"""Offline smoke test: builds tiny randomly-initialised Kronos models and runs
the full prediction pipeline. Needs no network access, so it verifies that the
environment is installed correctly. Forecast values are meaningless."""
from pathlib import Path

import pandas as pd
import torch

from model import Kronos, KronosPredictor, KronosTokenizer

INPUT_DATA_PATH = Path(__file__).parent / "data" / "regression_input.csv"


def build_tiny_models():
    tokenizer = KronosTokenizer(
        d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=1, n_dec_layers=1,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4,
    )
    model = Kronos(
        s1_bits=4, s2_bits=4, n_layers=1, d_model=32, n_heads=2, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=True,
    )
    return tokenizer.eval(), model.eval()


def test_offline_prediction_pipeline():
    torch.manual_seed(0)
    tokenizer, model = build_tiny_models()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])
    lookback, pred_len = 64, 8
    x_df = df.loc[: lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_ts = df.loc[: lookback - 1, "timestamps"]
    y_ts = df.loc[lookback : lookback + pred_len - 1, "timestamps"]

    pred_df = predictor.predict(x_df, x_ts, y_ts, pred_len=pred_len, verbose=False)

    assert pred_df.shape == (pred_len, 6)
    assert not pred_df.isnull().values.any()
