"""
Entrena Random Forest y KNN. Genera matriz de confusion por ticker.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Permitir import de features.py estando en /src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import add_features, add_labels, FEATURE_COLS

TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
LABEL_NAMES = ["sell", "hold", "buy"]

def prepare(ticker):
    df = pd.read_parquet(f"data/{ticker}.parquet")
    df = add_features(df)
    df = add_labels(df, horizon=1)
    df = df.dropna().reset_index(drop=True)
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut], df.iloc[cut:]
    X_train, y_train = train[FEATURE_COLS], train["label"].astype(int)
    X_test, y_test = test[FEATURE_COLS], test["label"].astype(int)
    return df, train, test, X_train, y_train, X_test, y_test

def train_rf(X_train, y_train):
    param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
    tscv = TimeSeriesSplit(n_splits=3)
    rf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    gs = GridSearchCV(rf, param_grid, cv=tscv, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"  RF best params: {gs.best_params_}")
    return gs.best_estimator_

def train_knn(X_train, y_train):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    param_grid = {"n_neighbors": [5, 15, 25, 50]}
    tscv = TimeSeriesSplit(n_splits=3)
    knn = KNeighborsClassifier()
    gs = GridSearchCV(knn, param_grid, cv=tscv, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train_s, y_train)
    print(f"  KNN best params: {gs.best_params_}")
    return gs.best_estimator_, scaler

def evaluate(name, model, X_test, y_test, ticker, scaler=None):
    X = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X)
    acc = accuracy_score(y_test, y_pred)
    print(f"  {name} accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, digits=4, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f"{name} - {ticker}")
    ax.set_ylabel("Real")
    ax.set_xlabel("Predicho")
    os.makedirs("reports", exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"reports/cm_{name}_{ticker}.png", dpi=120)
    plt.close(fig)
    return acc

def main():
    os.makedirs("models", exist_ok=True)
    summary = []
    for ticker in TICKERS:
        print(f"\n=== {ticker} ===")
        df, train, test, X_train, y_train, X_test, y_test = prepare(ticker)
        print(f"  train={len(train)} test={len(test)} features={len(FEATURE_COLS)}")

        rf = train_rf(X_train, y_train)
        rf_acc = evaluate("RF", rf, X_test, y_test, ticker)
        joblib.dump(rf, f"models/rf_{ticker}.pkl")

        knn, scaler = train_knn(X_train, y_train)
        knn_acc = evaluate("KNN", knn, X_test, y_test, ticker, scaler=scaler)
        joblib.dump((knn, scaler), f"models/knn_{ticker}.pkl")

        summary.append({"ticker": ticker, "rf_acc": rf_acc, "knn_acc": knn_acc})

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv("reports/model_summary.csv", index=False)
    print("\n=== Summary ===")
    print(df_sum)

if __name__ == "__main__":
    main()