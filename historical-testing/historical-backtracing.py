# ml_signal_strategy.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import ta
import requests
from typing import Optional, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def download_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, group_by="column")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna()


def fetch_fear_greed_index() -> pd.DataFrame:
    url = "https://api.alternative.me/fng/?limit=365"
    response = requests.get(url)
    data = response.json()
    records = []
    for item in data['data']:
        date = pd.to_datetime(pd.to_numeric(item['timestamp']), unit='s')
        value = int(item['value'])
        records.append({'date': date, 'fear_greed': value})
    df = pd.DataFrame(records).set_index('date').sort_index()
    return df


def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    bb = ta.volatility.BollingerBands(close=data["Close"], window=20, window_dev=2)
    data["bb_upper"] = bb.bollinger_hband()
    data["bb_lower"] = bb.bollinger_lband()

    stoch = ta.momentum.StochasticOscillator(
        high=data["High"], low=data["Low"], close=data["Close"], window=14, smooth_window=3
    )
    data["stoch_k"] = stoch.stoch()
    data["stoch_d"] = stoch.stoch_signal()

    data["vol_avg30"] = data["Volume"].rolling(window=30).mean()

    rsi = ta.momentum.RSIIndicator(close=data["Close"], window=14)
    data["rsi"] = rsi.rsi()

    data["close_mean20"] = data["Close"].rolling(window=20).mean()
    data["close_std20"] = data["Close"].rolling(window=20).std()
    data["z_score"] = (data["Close"] - data["close_mean20"]) / data["close_std20"]

    data["ema50"] = data["Close"].ewm(span=50).mean()
    atr = ta.volatility.AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14)
    data["atr"] = atr.average_true_range()

    data["rsi_rolling_max"] = data["rsi"].rolling(window=60).max()
    data["rsi_rolling_min"] = data["rsi"].rolling(window=60).min()
    data["rsi_dynamic_norm"] = (
        (data["rsi"] - data["rsi_rolling_min"]) /
        (data["rsi_rolling_max"] - data["rsi_rolling_min"] + 1e-9)
    ).clip(0, 1)

    return data.dropna(subset=["z_score", "stoch_k", "rsi", "atr"])


def create_ml_labels(data: pd.DataFrame, horizon_days: int = 5, threshold: float = 0.05) -> pd.DataFrame:
    data["future_return"] = data["Close"].shift(-horizon_days) / data["Close"] - 1
    data["target"] = (data["future_return"] > threshold).astype(int)
    return data.dropna(subset=["target"])


def train_model(data: pd.DataFrame):
    feature_cols = [
        "z_score", "stoch_k", "stoch_d", "rsi", "rsi_dynamic_norm",
        "bb_upper", "bb_lower", "ema50", "fear_greed"
    ]
    data = data.dropna(subset=feature_cols + ["target"])

    X = data[feature_cols]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    data["ml_prediction"] = model.predict(X)
    return model, data


def simulate_trades_from_ml(data: pd.DataFrame, stop_loss_pct=0.09, take_profit_multiple=3.0, max_hold_days=5):
    trades = []
    position = None
    entry_index = None
    data["entry"] = np.nan
    data["exit"] = np.nan

    for i in range(len(data)):
        current_price = data["Close"].iloc[i]
        current_date = data.index[i]

        if position is None and data["ml_prediction"].iloc[i] == 1:
            if data["rsi"].iloc[i] < 40 and data["z_score"].iloc[i] < -0.5:
                position = {
                    "entry_date": current_date,
                    "entry_price": current_price,
                    "atr": data["atr"].iloc[i],
                    "max_price": current_price
                }
                entry_index = i
                data.loc[current_date, "entry"] = current_price

        elif position is not None:
            entry_price = position["entry_price"]
            atr = position["atr"]

            if current_price > position["max_price"]:
                position["max_price"] = current_price

            take_profit_price = entry_price + take_profit_multiple * atr
            stop_loss_price = entry_price * (1 - stop_loss_pct)

            days_held = (current_date - position["entry_date"]).days
            exit_condition = (
                current_price >= take_profit_price or
                current_price <= stop_loss_price or
                days_held >= max_hold_days
            )

            if exit_condition:
                exit_price = current_price
                pct_return = (exit_price - entry_price) / entry_price
                trades.append({
                    "entry_date": position["entry_date"],
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": pct_return,
                    "days_held": days_held
                })
                data.loc[current_date, "exit"] = exit_price
                position = None
                entry_index = None

    return pd.DataFrame(trades), data


def print_performance(trades_df):
    total_trades = len(trades_df)
    win_rate = (trades_df["return"] > 0).mean() if total_trades > 0 else 0
    avg_gain = trades_df[trades_df["return"] > 0]["return"].mean() if total_trades > 0 else 0
    avg_loss = trades_df[trades_df["return"] < 0]["return"].mean() if total_trades > 0 else 0
    expectancy = win_rate * avg_gain + (1 - win_rate) * avg_loss if total_trades > 0 else 0

    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Avg Gain: {avg_gain:.2%}")
    print(f"Avg Loss: {avg_loss:.2%}")
    print(f"Expectancy per Trade: {expectancy:.2%}")


def plot_signals(data):
    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data["Close"], label="Close Price", color="blue")
    plt.scatter(data.index, data["entry"], label="Entry", color="green", marker="^", s=100)
    plt.scatter(data.index, data["exit"], label="Exit", color="red", marker="v", s=100)
    plt.title("ML Strategy Buy/Sell Points")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    ticker = "BITX"
    data = download_data(ticker)
    data = compute_indicators(data)

    fg = fetch_fear_greed_index()
    data = data.merge(fg, left_index=True, right_index=True, how="left")
    data["fear_greed"] = data["fear_greed"].ffill()

    data = create_ml_labels(data)
    model, data = train_model(data)

    trades_df, data = simulate_trades_from_ml(data)

    print(trades_df[["entry_date", "exit_date", "return", "days_held"]])
    print_performance(trades_df)
    plot_signals(data)


if __name__ == "__main__":
    main()
