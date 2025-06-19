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
    data["ml_prediction_proba"] = model.predict_proba(data[feature_cols])[:, 1]
    data["ml_prediction"] = (data["ml_prediction_proba"] >= 0.5).astype(int)

    return model, data


def simulate_trades_from_ml(data: pd.DataFrame, stop_loss_pct=0.10, take_profit_multiple=3.0, max_hold_days=5, prob_threshold=0.7, rsi_norm_threshold=0.5):
    trades = []
    open_trades = []

    data["entry"] = np.nan
    data["exit"] = np.nan

    for i in range(len(data)):
        current_price = data["Close"].iloc[i]
        current_date = data.index[i]
        prob = data["ml_prediction_proba"].iloc[i]
        rsi_norm = data["rsi_dynamic_norm"].iloc[i]  # normalized RSI between 0 and 1

        # Entry: only if prob above threshold and normalized RSI below threshold
        if prob >= prob_threshold and rsi_norm <= rsi_norm_threshold:
            open_trades.append({
                "entry_date": current_date,
                "entry_price": current_price,
                "atr": data["atr"].iloc[i],
                "max_price": current_price,
                "entry_index": i
            })
            data.loc[current_date, "entry"] = current_price

        trades_to_close = []
        for idx, trade in enumerate(open_trades):
            entry_price = trade["entry_price"]
            atr = trade["atr"]
            max_price = trade["max_price"]

            if current_price > max_price:
                open_trades[idx]["max_price"] = current_price

            take_profit_price = entry_price + take_profit_multiple * atr
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            days_held = (current_date - trade["entry_date"]).days

            if (current_price >= take_profit_price or
                current_price <= stop_loss_price or
                days_held >= max_hold_days):
                
                pct_return = (current_price - entry_price) / entry_price
                trades.append({
                    "entry_date": trade["entry_date"],
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "return": pct_return,
                    "days_held": days_held
                })
                data.loc[current_date, "exit"] = current_price
                trades_to_close.append(idx)

        for idx in reversed(trades_to_close):
            open_trades.pop(idx)

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
    total_gains = trades_df.loc[trades_df["return"] > 0, "return"].sum() * 100
    total_losses = -trades_df.loc[trades_df["return"] < 0, "return"].sum() * 100  # make losses positive
    net_return = total_gains - total_losses

    print(f"Total Gains: {total_gains:.2f}%")
    print(f"Total Losses: {total_losses:.2f}%")
    print(f"Net Return (Gains - Losses): {net_return:.2f}%")


def plot_signals(data: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

    # 1) Price + Bollinger Bands + Entry/Exit points
    ax = axes[0]
    ax.plot(data.index, data["Close"], label="Close", color="blue")
    ax.plot(data.index, data["bb_upper"], label="BB Upper", linestyle="--", color="gray")
    ax.plot(data.index, data["bb_lower"], label="BB Lower", linestyle="--", color="gray")
    ax.scatter(data.index[data["entry"].notna()], data["entry"].dropna(), marker="^", color="green", label="Buy Entry", s=100)
    ax.scatter(data.index[data["exit"].notna()], data["exit"].dropna(), marker="v", color="red", label="Sell Exit", s=100)
    ax.set_ylabel("Price")
    ax.set_title("Close Price and Bollinger Bands with Buy/Sell Signals")
    ax.legend()
    ax.grid(True)

    # 2) Stochastic Oscillator
    ax = axes[1]
    ax.plot(data.index, data["stoch_k"], label="%K", color="orange")
    ax.plot(data.index, data["stoch_d"], label="%D", color="purple")
    ax.axhline(20, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(80, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Stoch")
    ax.set_title("Stochastic Oscillator")
    ax.legend()
    ax.grid(True)

    # 3) Volume + 30-day Average Volume
    ax = axes[2]
    ax.bar(data.index, data["Volume"], label="Volume", color="lightblue")
    ax.plot(data.index, data["vol_avg30"], label="30-day Avg Volume", color="blue", linewidth=1.5)
    ax.set_ylabel("Volume")
    ax.set_title("Volume and 30-day Average")
    ax.legend()
    ax.grid(True)

    # 4) RSI with threshold lines
    ax = axes[3]
    ax.plot(data.index, data["rsi"], label="RSI", color="darkcyan")
    ax.axhline(70, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(30, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("RSI")
    ax.set_title("Relative Strength Index (RSI)")
    ax.legend()
    ax.grid(True)

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
