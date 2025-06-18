import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import ta
import requests
from typing import Optional, Dict


def download_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, group_by="column")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna()


def fetch_fear_greed_index() -> pd.DataFrame:
    """Fetches the Crypto Fear & Greed Index data from alternative.me API"""
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


    data = data.dropna(subset=["z_score", "stoch_k", "rsi", "atr"])
    return data


def generate_short_signals(
    data: pd.DataFrame,
    score_threshold_quantile: float = 0.55,
    volume_spike_ratio: float = 1.1,
    ema_condition: bool = True,
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    if weights is None:
        weights = {
            "z": 0.25,
            "stoch": 0.25,
            "rsi": 0.25,
            "bb": 0.15,
            "macd": 0.10,
            "fear_greed": 0.05  # small weight for fear & greed index
        }

    # MACD line calculation for use in scoring
    macd_line = data["Close"].ewm(span=12).mean() - data["Close"].ewm(span=26).mean()

    # Tier 1: Must-Have Filters for SHORT entry
    data["below_ema50"] = data["Close"] < data["ema50"] if ema_condition else True
    data["volume_spike"] = data["Volume"] > volume_spike_ratio * data["vol_avg30"]

    # Eligible rows must pass filters
    eligible = data["below_ema50"] & data["volume_spike"]

    # Tier 2: Scoring based on overbought indicators — mean reversion short logic

    # Normalize positive z-score (price above mean = potential to revert down)
    z_score_pos = data["z_score"].clip(lower=0)
    z_norm = (z_score_pos - z_score_pos.min()) / (z_score_pos.max() - z_score_pos.min() + 1e-9)

    # Stochastic %K overbought normalized
    stoch_overbought = (data["stoch_k"] - 80).clip(lower=0)
    stoch_norm = stoch_overbought / (stoch_overbought.max() + 1e-9)

        # NEW dynamic RSI normalization
    rsi_overbought = (data["rsi_dynamic_norm"] - 0.6).clip(lower=0)
    rsi_norm = rsi_overbought / (rsi_overbought.max() + 1e-9)

    # Bollinger Bands proximity normalized (closer to upper band = higher value)
    bb_width = data["bb_upper"] - data["bb_lower"]
    bb_pos = (data["Close"] - data["bb_lower"]) / (bb_width + 1e-9)
    bb_norm = bb_pos.clip(0, 1)

    # MACD normalized (higher MACD line could indicate momentum exhaustion)
    macd_norm = (macd_line - macd_line.min()) / (macd_line.max() - macd_line.min() + 1e-9)

    # Fear & Greed Index normalized (higher greed = better short opportunity)
    fear_greed_norm = data["fear_greed"].fillna(50) / 100

    # Composite score calculation
    data["composite_score"] = (
        weights["z"] * z_norm +
        weights["stoch"] * stoch_norm +
        weights["rsi"] * rsi_norm +
        weights["bb"] * bb_norm +
        weights["macd"] * macd_norm +
        weights["fear_greed"] * fear_greed_norm
    )

    # Tier 3: Entry Rule — composite score above quantile threshold + filters
    threshold = data["composite_score"].quantile(score_threshold_quantile)
    data["short_signal"] = eligible & (data["composite_score"] >= threshold)

    # Exit Rule: RSI < 35 or price < lower BB (oversold conditions = cover)
    data["raw_cover_signal"] = (
    (data["rsi"] < data["rsi_rolling_min"] + 5) | 
    (data["Close"] < data["bb_lower"])
    )

    data["cover_signal"] = False



    return data


def simulate_short_trades(
    data: pd.DataFrame,
    stop_loss_pct: float = 0.09,
    take_profit_multiple: float = 3.0,
    use_trailing_stop: bool = False
) -> pd.DataFrame:
    trades = []
    position = None
    entry_index = None
    trailing_stop_price = None

    # Clear existing signals for plotting
    data["short_entry_real"] = False
    data["cover_signal"] = False

    for i in range(len(data)):
        current_price = data["Close"].iloc[i]
        current_date = data.index[i]

        if position is None and data["short_signal"].iloc[i]:
            # Enter new SHORT position
            position = {
                "entry_date": current_date,
                "entry_price": current_price,
                "atr": data["atr"].iloc[i],
                "min_price": current_price  # For trailing stop tracking
            }
            entry_index = i
            trailing_stop_price = None
            data.at[current_date, "short_entry_real"] = True

        elif position is not None:
            entry_price = position["entry_price"]
            atr = position["atr"]

            # Update min price for trailing stop (lowest price during short)
            if current_price < position["min_price"]:
                position["min_price"] = current_price

            # Calculate trailing stop price if enabled
            if use_trailing_stop:
                trailing_stop_price = position["min_price"] * (1 + stop_loss_pct)
                trailing_stop_triggered = current_price >= trailing_stop_price
            else:
                trailing_stop_triggered = False

            # Fixed stop loss price level
            stop_loss_price = entry_price * (1 + stop_loss_pct)
            stop_loss_triggered = current_price >= stop_loss_price

            # Take profit price level
            take_profit_price = entry_price - take_profit_multiple * atr
            take_profit_triggered = current_price <= take_profit_price

            # Early exit signals
            cover_signal = data["raw_cover_signal"].iloc[i]
            early_exit_rsi = data["rsi"].iloc[i] > data["rsi_rolling_max"].iloc[i] - 5

            # Determine if exit conditions met
            if take_profit_triggered or stop_loss_triggered or trailing_stop_triggered or cover_signal or early_exit_rsi:
                exit_reason = (
                    "Take Profit" if take_profit_triggered else
                    "Stop Loss" if stop_loss_triggered else
                    "Trailing Stop" if trailing_stop_triggered else
                    "Signal Exit" if cover_signal else
                    "Early Exit RSI"
                )

                exit_price = current_price
                if stop_loss_triggered:
                    # Use max price for stop loss exit (conservative)
                    exit_price = max(exit_price, stop_loss_price)

                ret = (entry_price - exit_price) / entry_price  # Short return

                trades.append({
                    "entry_date": position["entry_date"],
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": ret,
                    "exit_reason": exit_reason,
                    "days_held": (current_date - position["entry_date"]).days
                })
                data.at[current_date, "cover_signal"] = True
                position = None
                entry_index = None
                trailing_stop_price = None

    # Clean up if position still open at the end
    if position is not None:
        data.at[position["entry_date"], "short_entry_real"] = False
        position = None

    return pd.DataFrame(trades)


def print_performance(trades_df: pd.DataFrame):
    total_trades = len(trades_df)
    win_rate = (trades_df["return"] > 0).mean() if total_trades > 0 else 0
    avg_gain = trades_df[trades_df["return"] > 0]["return"].mean() if total_trades > 0 else 0
    avg_loss = trades_df[trades_df["return"] < 0]["return"].mean() if total_trades > 0 else 0
    expectancy = win_rate * avg_gain + (1 - win_rate) * avg_loss if total_trades > 0 else 0

    print(f"\nTotal Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Avg Gain: {avg_gain:.2%}")
    print(f"Avg Loss: {avg_loss:.2%}")
    print(f"Expectancy per Trade: {expectancy:.2%}")


def plot_signals(data: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

    ax = axes[0]
    ax.plot(data.index, data["Close"], label="Close", color="blue")
    ax.plot(data.index, data["bb_upper"], label="BB Upper", linestyle="--", color="gray")
    ax.plot(data.index, data["bb_lower"], label="BB Lower", linestyle="--", color="gray")
    ax.scatter(data.index[data["short_entry_real"]], data["Close"][data["short_entry_real"]], marker="^", color="green", label="Short Entry", s=100)
    ax.scatter(data.index[data["cover_signal"]], data["Close"][data["cover_signal"]], marker="v", color="red", label="Cover", s=100)
    ax.set_ylabel("Price")
    ax.set_title("SBIT Close Price and Bollinger Bands with Short Entry/Cover Signals")
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
    ticker = "SBIT"
    data = download_data(ticker)
    data = compute_indicators(data)

    fear_greed_df = fetch_fear_greed_index()
    data = data.merge(fear_greed_df, left_index=True, right_index=True, how='left')
    data["fear_greed"] = data["fear_greed"].ffill()

    best_params = {
        "quantile": 0.4,
        "vol_spike": 1.05,
        "ema": True,
        "stop_loss_pct": 0.09,
        "take_profit_multiple": 3.0,
        "use_trailing_stop": False
    }

    data = generate_short_signals(
        data,
        score_threshold_quantile=best_params["quantile"],
        volume_spike_ratio=best_params["vol_spike"],
        ema_condition=best_params["ema"]
    )
    trades_df = simulate_short_trades(
        data,
        stop_loss_pct=best_params["stop_loss_pct"],
        take_profit_multiple=best_params["take_profit_multiple"],
        use_trailing_stop=best_params["use_trailing_stop"]
    )

    print("\nTrade Log:")
    print(trades_df[["entry_date", "exit_date", "return", "exit_reason", "days_held"]])
    print("\nAverage Return: {:.2f}%".format(trades_df["return"].mean() * 100))

    print_performance(trades_df)

    plot_signals(data)


if __name__ == "__main__":
    main()
