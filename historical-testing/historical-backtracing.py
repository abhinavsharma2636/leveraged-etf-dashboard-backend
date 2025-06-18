# trading_strategy.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import ta
from typing import List, Dict, Optional

def download_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval)
    return data.dropna()

def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=data["Close"].squeeze(), window=20, window_dev=2)
    data["bb_upper"] = bb.bollinger_hband().squeeze()
    data["bb_lower"] = bb.bollinger_lband().squeeze()

    # Stochastic Oscillator
    high = data["High"].squeeze()
    low = data["Low"].squeeze()
    close = data["Close"].squeeze()

    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    data["stoch_k"] = stoch.stoch().squeeze()
    data["stoch_d"] = stoch.stoch_signal().squeeze()

    # Volume average
    data["vol_avg30"] = data["Volume"].rolling(window=30).mean().squeeze()

    # RSI
    rsi = ta.momentum.RSIIndicator(close=close, window=14)
    data["rsi"] = rsi.rsi().squeeze()

    return data



def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    close = data["Close"].squeeze()
    bb_lower = data["bb_lower"].squeeze()
    bb_upper = data["bb_upper"].squeeze()
    stoch_k = data["stoch_k"].squeeze()
    stoch_d = data["stoch_d"].squeeze()
    rsi = data["rsi"].squeeze()
    vol = data["Volume"].squeeze()
    vol_avg30 = data["vol_avg30"].squeeze()

    # Buy Signal:
    # - Close is near the lower BB (within 2%)
    # - Stochastic crossover below 20 (bullish reversal)
    # - RSI below 35 (oversold)
    # - Volume is surging
    data["buy_signal"] = (
        ((close - bb_lower) / bb_lower < 0.02) &  # within 2% of lower BB
        (stoch_k > stoch_d) &
        (stoch_k.shift(1) < stoch_d.shift(1)) &
        (stoch_k < 20) &
        (rsi < 35) &
        (vol > 1.5 * vol_avg30)
    )

    # We'll generate a "raw" sell signal first, then clean it up later
    data["raw_sell_signal"] = (
        ((bb_upper - close) / bb_upper < 0.02) &  # near upper BB (within 2%)
        (stoch_k < stoch_d) &
        (stoch_k.shift(1) > stoch_d.shift(1)) &
        (stoch_k > 80) &
        (rsi > 65)
    )

    # Placeholder for clean sell_signal: we’ll filter only if we’re in a trade
    data["sell_signal"] = False  # Will update during simulation

    return data


def simulate_trades(data: pd.DataFrame) -> pd.DataFrame:
    trades = []
    position: Optional[Dict] = None

    for i in range(len(data)):
        if data["buy_signal"].iloc[i] and position is None:
            position = {
                "entry_date": data.index[i],
                "entry_price": data["Close"].iloc[i]
            }
        elif data["raw_sell_signal"].iloc[i] and position is not None:
            position.update({
                "exit_date": data.index[i],
                "exit_price": data["Close"].iloc[i],
                "return": (data["Close"].iloc[i] - position["entry_price"]) / position["entry_price"]
            })
            trades.append(position)
            # Mark sell_signal on this day
            data.at[data.index[i], "sell_signal"] = True
            position = None

    return pd.DataFrame(trades)


def plot_signals(data: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

    # Price + Bollinger Bands
    ax = axes[0]
    ax.plot(data.index, data["Close"].squeeze(), label="Close", color="blue")
    ax.plot(data.index, data["bb_upper"].squeeze(), label="BB Upper", linestyle="--", color="gray")
    ax.plot(data.index, data["bb_lower"].squeeze(), label="BB Lower", linestyle="--", color="gray")
    ax.scatter(data.index[data["buy_signal"]], data["Close"][data["buy_signal"]], marker="^", color="green", label="Buy", s=100)
    ax.scatter(data.index[data["sell_signal"]], data["Close"][data["sell_signal"]], marker="v", color="red", label="Sell", s=100)
    ax.set_ylabel("Price")
    ax.set_title("BITX Close Price and Bollinger Bands with Buy/Sell Signals")
    ax.legend()
    ax.grid(True)

    # Stochastic Oscillator
    ax = axes[1]
    ax.plot(data.index, data["stoch_k"].squeeze(), label="%K", color="orange")
    ax.plot(data.index, data["stoch_d"].squeeze(), label="%D", color="purple")
    ax.axhline(20, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(80, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Stoch")
    ax.set_title("Stochastic Oscillator")
    ax.legend()
    ax.grid(True)

    # Volume
    ax = axes[2]
    ax.bar(data.index, data["Volume"].squeeze(), label="Volume", color="lightblue")
    ax.plot(data.index, data["vol_avg30"].squeeze(), label="30-day Avg Volume", color="blue", linewidth=1.5)
    ax.set_ylabel("Volume")
    ax.set_title("Volume and 30-day Average")
    ax.legend()
    ax.grid(True)

    # RSI
    ax = axes[3]
    ax.plot(data.index, data["rsi"].squeeze(), label="RSI", color="darkcyan")
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

    print("\nAverage Return: {:.2f}%".format(trades_df["return"].mean() * 100))

    plot_signals(data)

if __name__ == "__main__":
    main()
