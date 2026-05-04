"""
Genera predicciones live para todos los tickers y las sube a Firestore.
Esto es lo que la app web va a leer.
"""
import os
import sys
import joblib
import pandas as pd
from datetime import datetime
from pybit.unified_trading import HTTP
import firebase_admin
from firebase_admin import credentials, firestore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import add_features, FEATURE_COLS

# Config
TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
INTERVAL = "60"
ATR_SL_MULT = 2.0
RR_RATIO = 2.0

# Inicializar Firebase
KEY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "firebase_key.json")
cred = credentials.Certificate(KEY_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Cliente Bybit (data publica, no necesita keys)
session = HTTP(testnet=True)

def get_recent_klines(symbol, limit=200):
    resp = session.get_kline(category="spot", symbol=symbol, interval=INTERVAL, limit=limit)
    rows = resp["result"]["list"]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)

def predict_and_upload(ticker):
    print(f"\n--- {ticker} ---")
    try:
        df = get_recent_klines(ticker, limit=200)
        df_feat = add_features(df).dropna()
        if df_feat.empty:
            print("  Datos insuficientes")
            return

        last = df_feat.iloc[-1]
        X = pd.DataFrame([last[FEATURE_COLS].values], columns=FEATURE_COLS)

        rf = joblib.load(f"models/rf_{ticker}.pkl")
        pred = int(rf.predict(X)[0])
        proba = rf.predict_proba(X)[0]

        # Map de confianzas (manejo robusto si una clase falta)
        classes = rf.classes_
        proba_dict = {int(c): float(p) for c, p in zip(classes, proba)}
        sell_p = proba_dict.get(0, 0.0)
        hold_p = proba_dict.get(1, 0.0)
        buy_p = proba_dict.get(2, 0.0)

        labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
        signal = labels[pred]

        entry = float(last["close"])
        atr = float(last["atr_14"])
        sl = entry - ATR_SL_MULT * atr
        tp = entry + ATR_SL_MULT * RR_RATIO * atr

        # Documento que se sube a Firestore
        doc = {
            "ticker": ticker,
            "signal": signal,
            "current_price": entry,
            "atr": atr,
            "stop_loss": sl,
            "take_profit": tp,
            "sl_pct": (1 - sl / entry) * 100,
            "tp_pct": (tp / entry - 1) * 100,
            "risk_reward": RR_RATIO,
            "confidence_sell": sell_p,
            "confidence_hold": hold_p,
            "confidence_buy": buy_p,
            "max_confidence": max(sell_p, hold_p, buy_p),
            "updated_at": datetime.now().isoformat(),
            "model": "RandomForest",
            "timeframe": "1H",
        }

        # Sube a Firestore en la coleccion 'predictions'
        db.collection("predictions").document(ticker).set(doc)

        print(f"  Senal: {signal}")
        print(f"  Precio: ${entry:.2f}")
        print(f"  Confianza: SELL={sell_p:.2%} HOLD={hold_p:.2%} BUY={buy_p:.2%}")
        print(f"  SL: ${sl:.2f}  TP: ${tp:.2f}")
        print(f"  Subido a Firestore")

    except Exception as e:
        print(f"  Error: {e}")

def main():
    print("=== Generando predicciones y subiendo a Firebase ===\n")
    for ticker in TICKERS:
        predict_and_upload(ticker)
    print("\n=== Listo. Datos disponibles para la app web. ===")

if __name__ == "__main__":
    main()