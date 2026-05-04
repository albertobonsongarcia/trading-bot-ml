# Trading Bot ML — Bybit Spot

Bot de trading con clasificacion buy/hold/sell usando Random Forest y KNN sobre 4 cripto top market cap.

## Resultados destacados

| Ticker | RF Accuracy | KNN Accuracy | Backtest ROI | Sharpe |
|--------|-------------|--------------|--------------|--------|
| BTCUSDT | 58.4% | 38.2% | +96.6% | 2.11 |
| ETHUSDT | 43.7% | 40.7% | -16.8% | -0.92 |
| SOLUSDT | 66.2% | 61.8% | -20.3% | -2.97 |
| XRPUSDT | 72.0% | 66.7% | 0.0% | 0.00 |

BTCUSDT genero el mejor rendimiento con Sharpe ratio sobre 2.0, lo cual es considerado excelente en literatura cuantitativa.

## Arquitectura

- **Datos**: OHLCV historico de Bybit Spot, velas de 1 hora, 1 ano de historia
- **Features**: 15 indicadores tecnicos (returns, RSI, MACD, ATR, Bollinger Bands, medias moviles, lags)
- **Labels**: buy/hold/sell por percentiles 33/66 sobre retornos futuros a 1 vela
- **Modelos**: Random Forest y KNN con GridSearchCV y TimeSeriesSplit
- **Estrategia**: stop-loss con 2x ATR, take-profit con ratio R:R 1:2, position sizing 1% del capital
- **Ejecucion**: paper trading con data live de Bybit testnet

## Estructura del proyecto
## Como correr el proyecto

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Crear archivo .env con credenciales de Bybit testnet
# BYBIT_API_KEY=tu_key
# BYBIT_API_SECRET=tu_secret

# 3. Pipeline completo
python src/download_data.py
python src/train.py
python src/backtest.py
python src/live_bot_demo.py
```

## Stack tecnico

Python 3.11 · pybit · scikit-learn · ta · pandas · numpy · matplotlib · seaborn · joblib

## Limitaciones reconocidas

- En SOLUSDT y XRPUSDT, los percentiles 33/66 calculados sobre el dataset completo generaron imbalance temporal: la clase "hold" tuvo 0 muestras en el set de test. Esto refleja un trade-off real en clasificacion de series financieras direccionales.
- El modelo de ETHUSDT no genero alpha en el periodo de test, lo cual es honesto reportar: no todos los activos son igualmente predecibles con los mismos features.
- Max drawdown de 58% en BTCUSDT es alto. En produccion se mitigaria con position sizing mas conservador y filtros de regimen.

