"""
Features tecnicas y labels buy/hold/sell por percentiles.
"""
import pandas as pd
import numpy as np
import ta

def add_features(df):
    df = df.copy()
    df["return_1"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["vol_20"] = df["return_1"].rolling(20).std()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma_ratio"] = df["sma_10"] / df["sma_50"]
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_pct"] = (df["close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"])

    df["atr_14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["return_1"].shift(lag)
         
    df = df.replace([np.inf, -np.inf], np.nan)
 

    return df

def add_labels(df, horizon=1):
    df = df.copy()
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    lower = df["future_return"].quantile(0.33)
    upper = df["future_return"].quantile(0.66)

    def label_row(r):
        if pd.isna(r):
            return np.nan
        if r <= lower:
            return 0
        elif r >= upper:
            return 2
        else:
            return 1

    df["label"] = df["future_return"].apply(label_row)
    return df

FEATURE_COLS = [
    "return_1", "log_return", "vol_20",
    "sma_ratio", "rsi_14",
    "macd", "macd_signal", "macd_diff",
    "bb_pct", "atr_14",
    "volume_ratio",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
]