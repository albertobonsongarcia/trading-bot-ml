"""
Descarga OHLCV historico de Bybit Spot testnet en velas de 1H.
Guarda parquet por ticker en /data.
"""
import os
import time
import pandas as pd
from pybit.unified_trading import HTTP

TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL = "60"   # 60 minutos = 1H
DAYS_BACK = 365   # 1 ano
LIMIT = 1000

session = HTTP(testnet=True)

def fetch_klines(symbol, interval, days_back):
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days_back * 24 * 60 * 60 * 1000
    all_rows = []
    cursor_end = end_ms

    while cursor_end > start_ms:
        resp = session.get_kline(
            category="spot",
            symbol=symbol,
            interval=interval,
            end=cursor_end,
            limit=LIMIT,
        )
        rows = resp["result"]["list"]
        if not rows:
            break
        all_rows.extend(rows)
        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ms:
            break
        cursor_end = oldest_ts - 1
        time.sleep(0.2)

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("Int64"), unit="ms")
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df

def main():
    os.makedirs("data", exist_ok=True)
    for ticker in TICKERS:
        print(f"Descargando {ticker}...")
        df = fetch_klines(ticker, INTERVAL, DAYS_BACK)
        out = f"data/{ticker}.parquet"
        df.to_parquet(out, index=False)
        print(f"  guardado: {len(df)} filas en {out}")

if __name__ == "__main__":
    main()