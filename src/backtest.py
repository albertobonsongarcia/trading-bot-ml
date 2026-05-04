"""
Backtest simple con SL/TP basados en ATR.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import add_features, add_labels, FEATURE_COLS

TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
INITIAL_CAPITAL = 10_000
FEE = 0.001
ATR_SL_MULT = 2.0
RR_RATIO = 2.0

def backtest_ticker(ticker):
    df = pd.read_parquet(f"data/{ticker}.parquet")
    df = add_features(df)
    df = add_labels(df, horizon=1)
    df = df.dropna().reset_index(drop=True)

    cut = int(len(df) * 0.8)
    test = df.iloc[cut:].reset_index(drop=True).copy()

    rf = joblib.load(f"models/rf_{ticker}.pkl")
    test["pred"] = rf.predict(test[FEATURE_COLS])

    capital = INITIAL_CAPITAL
    equity_curve = []
    in_position = False
    entry_price = sl = tp = 0.0
    trades = []

    for i, row in test.iterrows():
        price = row["close"]

        if in_position:
            hit_sl = row["low"] <= sl
            hit_tp = row["high"] >= tp
            exit_price = None
            if hit_sl and hit_tp:
                exit_price = sl
            elif hit_sl:
                exit_price = sl
            elif hit_tp:
                exit_price = tp

            if exit_price is not None:
                pnl_pct = (exit_price - entry_price) / entry_price - 2 * FEE
                capital *= (1 + pnl_pct)
                trades.append({"entry": entry_price, "exit": exit_price, "pnl_pct": pnl_pct, "ts": row["timestamp"]})
                in_position = False

        if not in_position and row["pred"] == 2:
            entry_price = price
            atr = row["atr_14"]
            sl = entry_price - ATR_SL_MULT * atr
            tp = entry_price + ATR_SL_MULT * RR_RATIO * atr
            in_position = True

        equity_curve.append({"timestamp": row["timestamp"], "equity": capital})

    eq = pd.DataFrame(equity_curve)
    eq["returns"] = eq["equity"].pct_change()

    total_return = capital / INITIAL_CAPITAL - 1
    sharpe = np.sqrt(24 * 365) * eq["returns"].mean() / eq["returns"].std() if eq["returns"].std() > 0 else 0
    rolling_max = eq["equity"].cummax()
    drawdown = (eq["equity"] / rolling_max - 1)
    max_dd = drawdown.min()
    win_rate = np.mean([t["pnl_pct"] > 0 for t in trades]) if trades else 0

    print(f"\n{ticker}")
    print(f"  trades: {len(trades)}")
    print(f"  ROI: {total_return*100:.2f}%")
    print(f"  Sharpe: {sharpe:.3f}")
    print(f"  Max drawdown: {max_dd*100:.2f}%")
    print(f"  Win rate: {win_rate*100:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq["timestamp"], eq["equity"], label="Equity")
    ax.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"Equity curve - {ticker} (ROI {total_return*100:.1f}%)")
    ax.set_ylabel("USD")
    ax.legend()
    fig.tight_layout()
    os.makedirs("reports", exist_ok=True)
    fig.savefig(f"reports/equity_{ticker}.png", dpi=120)
    plt.close(fig)

    return {
        "ticker": ticker,
        "trades": len(trades),
        "roi_pct": total_return * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "win_rate_pct": win_rate * 100,
    }

def main():
    results = [backtest_ticker(t) for t in TICKERS]
    df = pd.DataFrame(results)
    df.to_csv("reports/backtest_summary.csv", index=False)
    print("\n=== Backtest summary ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()