"""
Bot live: lee data reciente, predice, y coloca orden BUY en Bybit testnet.
"""
import os
import sys
import joblib
import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import add_features, FEATURE_COLS

load_dotenv()

TICKERS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "60"
ATR_SL_MULT = 2.0
RR_RATIO = 2.0
RISK_PER_TRADE = 0.01

session = HTTP(
    testnet=True,
    api_key=os.getenv("BYBIT_API_KEY"),
    api_secret=os.getenv("BYBIT_API_SECRET"),
)

def get_recent_klines(symbol, limit=200):
    resp = session.get_kline(category="spot", symbol=symbol, interval=INTERVAL, limit=limit)
    rows = resp["result"]["list"]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("Int64"), unit="ms")
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)

def get_balance_usdt():
    try:
        resp = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        coins = resp["result"]["list"][0]["coin"]
        for c in coins:
            if c["coin"] == "USDT":
                return float(c["walletBalance"])
    except Exception as e:
        print(f"Error consultando balance: {e}")
    return 0.0

def run_once():
    balance = get_balance_usdt()
    print(f"Balance USDT: {balance:.2f}")

    for ticker in TICKERS:
        print(f"\nEvaluando {ticker}...")
        df = get_recent_klines(ticker, limit=200)
        df = add_features(df).dropna()
        if df.empty:
            print("  no hay datos suficientes")
            continue

        last = df.iloc[-1]
        X = last[FEATURE_COLS].values.reshape(1, -1)

        rf = joblib.load(f"models/rf_{ticker}.pkl")
        pred = int(rf.predict(X)[0])
        labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
        print(f"  Senal: {labels[pred]} (precio {last['close']:.2f}, ATR {last['atr_14']:.2f})")

        if pred == 2:
            entry = last["close"]
            atr = last["atr_14"]
            sl = entry - ATR_SL_MULT * atr
            risk_per_unit = entry - sl
            if risk_per_unit <= 0:
                print("  SL invalido, skip")
                continue
            qty_usdt = balance * RISK_PER_TRADE
            qty = round(qty_usdt / entry, 6)
            print(f"  Colocando BUY qty={qty} entry={entry:.2f} SL={sl:.2f}")
            try:
                order = session.place_order(
                    category="spot",
                    symbol=ticker,
                    side="Buy",
                    orderType="Market",
                    qty=str(qty),
                )
                print(f"  Orden ejecutada: {order['result']}")
            except Exception as e:
                print(f"  Error: {e}")

if __name__ == "__main__":
    run_once()