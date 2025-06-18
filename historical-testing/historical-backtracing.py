# trading_strategy.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import ta
from typing import List, Dict, Optional

def download_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, group_by="column")
    
    # Flatten MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    return data.dropna()

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

    # Calculate Z-score of Close price over rolling 20-day window
    data["close_mean20"] = data["Close"].rolling(window=20).mean()
    data["close_std20"] = data["Close"].rolling(window=20).std()
    data["z_score"] = (data["Close"] - data["close_mean20"]) / data["close_std20"]
    

    # Clean up NaN rows created by rolling calculations
    data = data.dropna(subset=["z_score", "stoch_k", "rsi"])

    return data

def generate_signals(data: pd.DataFrame, score_threshold_quantile: float = 0.80) -> pd.DataFrame:
    # Composite score components (negative z-score means price is below recent mean)
    z_score_neg = -data["z_score"]  # flip sign so that strong negative = high positive score
    stoch_oversold = (20 - data["stoch_k"]).clip(lower=0)  # only positive if stoch_k < 20
    rsi_oversold = (45 - data["rsi"]).clip(lower=0)  # only positive if rsi < 45

    # Normalize components by max to scale between 0 and 1 (avoid divide by zero)
    z_norm = z_score_neg / (z_score_neg.max() if z_score_neg.max() != 0 else 1)
    stoch_norm = stoch_oversold / (stoch_oversold.max() if stoch_oversold.max() != 0 else 1)
    rsi_norm = rsi_oversold / (rsi_oversold.max() if rsi_oversold.max() != 0 else 1)

    # Composite score (equal weights)
    data["composite_score"] = z_norm + stoch_norm + rsi_norm

    # Threshold for buy signals - top (1 - score_threshold_quantile) quantile, e.g. 0.80 means top 20%
    threshold = data["composite_score"].quantile(score_threshold_quantile)
    data["buy_signal"] = data["composite_score"] >= threshold

    # Keep old raw sell signal logic for sell signals
    rsi = data["rsi"]
    close = data["Close"]
    ma20 = close.rolling(window=20).mean()

    data["raw_sell_signal"] = (
        (rsi > 60) |
        (close > ma20)
    )

    data["sell_signal"] = False

    return data

def simulate_trades(data: pd.DataFrame) -> pd.DataFrame:
    trades = []
    position: Optional[Dict] = None
    entry_index = None

    data["sell_signal"] = False  # Reset to ensure clean plotting

    for i in range(len(data)):
        current_price = data["Close"].iloc[i]
        current_date = data.index[i]

        # BUY: Only if no position
        if position is None and data["buy_signal"].iloc[i]:
            position = {
                "entry_date": current_date,
                "entry_price": current_price
            }
            entry_index = i

        # SELL: Only if holding a position
        elif position is not None:
            days_held = i - entry_index
            entry_price = position["entry_price"]
            pct_change = (current_price - entry_price) / entry_price

            take_profit = pct_change >= 0.20
            stop_loss = pct_change <= -0.10
            time_exit = days_held >= 5
            raw_sell = data["raw_sell_signal"].iloc[i]

            if take_profit or stop_loss or time_exit or raw_sell:
                position.update({
                    "exit_date": current_date,
                    "exit_price": current_price,
                    "return": pct_change,
                    "days_held": days_held,
                    "exit_reason": (
                        "Take Profit" if take_profit else
                        "Stop Loss" if stop_loss else
                        "Time Exit" if time_exit else
                        "Signal Exit"
                    )
                })
                trades.append(position)
                data.at[current_date, "sell_signal"] = True
                position = None
                entry_index = None

    return pd.DataFrame(trades)


def plot_signals(data: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

    ax = axes[0]
    ax.plot(data.index, data["Close"], label="Close", color="blue")
    ax.plot(data.index, data["bb_upper"], label="BB Upper", linestyle="--", color="gray")
    ax.plot(data.index, data["bb_lower"], label="BB Lower", linestyle="--", color="gray")
    ax.scatter(data.index[data["buy_signal"]], data["Close"][data["buy_signal"]], marker="^", color="green", label="Buy", s=100)
    ax.scatter(data.index[data["sell_signal"]], data["Close"][data["sell_signal"]], marker="v", color="red", label="Sell", s=100)
    ax.set_ylabel("Price")
    ax.set_title("BITX Close Price and Bollinger Bands with Buy/Sell Signals")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.plot(data.index, data["stoch_k"], label="%K", color="orange")
    ax.plot(data.index, data["stoch_d"], label="%D", color="purple")
    ax.axhline(20, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(80, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Stoch")
    ax.set_title("Stochastic Oscillator")
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    ax.bar(data.index, data["Volume"], label="Volume", color="lightblue")
    ax.plot(data.index, data["vol_avg30"], label="30-day Avg Volume", color="blue", linewidth=1.5)
    ax.set_ylabel("Volume")
    ax.set_title("Volume and 30-day Average")
    ax.legend()
    ax.grid(True)

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
    data = generate_signals(data)
    trades_df = simulate_trades(data)

    print("\nTrade Log:")
    print(trades_df[["entry_date", "exit_date", "return", "exit_reason", "days_held"]])
    print("\nAverage Return: {:.2f}%".format(trades_df["return"].mean() * 100))

    plot_signals(data)

if __name__ == "__main__":
    main()
