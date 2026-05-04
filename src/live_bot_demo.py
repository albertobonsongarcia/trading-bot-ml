"""
Bot live en modo paper trading: descarga data real, predice senal real,
y simula la ejecucion de la orden con logging detallado.
"""
import os
import sys
import json
import joblib
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import add_features, FEATURE_COLS

load_dotenv()

TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
INTERVAL = "60"
ATR_SL_MULT = 2.0
RR_RATIO = 2.0
RISK_PER_TRADE = 0.01
DEMO_BALANCE = 10000.0  # balance simulado
PAPER_TRADING = True

session = HTTP(testnet=True)  # data publica, no necesita keys

def get_recent_klines(symbol, limit=200):
    resp = session.get_kline(category="spot", symbol=symbol, interval=INTERVAL, limit=limit)
    rows = resp["result"]["list"]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)

def simulate_order(ticker, side, qty, entry, sl, tp):
    """Simula una orden y la guarda en log JSON."""
    order = {
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "side": side,
        "qty": qty,
        "entry_price": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "risk_reward": RR_RATIO,
        "mode": "PAPER_TRADING",
    }
    os.makedirs("reports", exist_ok=True)
    log_path = "reports/paper_trades.json"

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            trades = json.load(f)
    else:
        trades = []

    trades.append(order)
    with open(log_path, "w") as f:
        json.dump(trades, f, indent=2, default=str)

    return order

def run_once():
    balance = DEMO_BALANCE
    print(f"=== BOT EN MODO PAPER TRADING ===")
    print(f"Balance simulado: ${balance:.2f} USDT\n")

    for ticker in TICKERS:
        print(f"--- Evaluando {ticker} ---")
        try:
            df = get_recent_klines(ticker, limit=200)
            df_feat = add_features(df).dropna()
            if df_feat.empty:
                print("  Datos insuficientes, skip\n")
                continue

            last = df_feat.iloc[-1]
            X = pd.DataFrame([last[FEATURE_COLS].values], columns=FEATURE_COLS)

            rf = joblib.load(f"models/rf_{ticker}.pkl")
            pred = int(rf.predict(X)[0])
            proba = rf.predict_proba(X)[0]

            labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
            print(f"  Precio actual: ${last['close']:.2f}")
            print(f"  ATR(14): ${last['atr_14']:.2f}")
            print(f"  Senal modelo: {labels[pred]}")
            classes = rf.classes_
            proba_dict = {int(c): float(p) for c, p in zip(classes, proba)}
            sell_p = proba_dict.get(0, 0.0)
            hold_p = proba_dict.get(1, 0.0)
            buy_p = proba_dict.get(2, 0.0)
            print(f"  Confianza: SELL={sell_p:.2%} HOLD={hold_p:.2%} BUY={buy_p:.2%}")

            if pred == 2:
                entry = float(last["close"])
                atr = float(last["atr_14"])
                sl = entry - ATR_SL_MULT * atr
                tp = entry + ATR_SL_MULT * RR_RATIO * atr
                qty_usdt = balance * RISK_PER_TRADE
                qty = round(qty_usdt / entry, 6)

                order = simulate_order(ticker, "Buy", qty, entry, sl, tp)
                print(f"  ✓ ORDEN SIMULADA EJECUTADA")
                print(f"    qty: {qty} {ticker.replace('USDT','')}")
                print(f"    entry: ${entry:.2f}")
                print(f"    SL: ${sl:.2f} (-{(1-sl/entry)*100:.2f}%)")
                print(f"    TP: ${tp:.2f} (+{(tp/entry-1)*100:.2f}%)")
                print(f"    Riesgo: ${qty_usdt:.2f} USDT")
            else:
                print(f"  - No se ejecuta orden (senal {labels[pred]})")
            print()
        except Exception as e:
            print(f"  Error: {e}\n")

    print(f"\n=== Log de operaciones guardado en reports/paper_trades.json ===")

if __name__ == "__main__":
    run_once()