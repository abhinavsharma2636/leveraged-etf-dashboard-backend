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

    data = data.dropna(subset=["z_score", "stoch_k", "rsi", "atr"])
    return data


def generate_signals(
    data: pd.DataFrame,
    score_threshold_quantile: float = 0.55,
    volume_spike_ratio: float = 1.1,
    ema_condition: bool = True,
    weights: Dict[str, float] = None
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

    # Tier 1: Must-Have Filters
    data["above_ema50"] = data["Close"] > data["ema50"]
    data["volume_spike"] = data["Volume"] > volume_spike_ratio * data["vol_avg30"]
    eligible = data["above_ema50"] & data["volume_spike"] if ema_condition else data["volume_spike"]

    # Tier 2: Compute scoring indicators

    # Normalized negative z-score (mean reversion)
    z_score_neg = -data["z_score"]
    z_norm = (z_score_neg - z_score_neg.min()) / (z_score_neg.max() - z_score_neg.min() + 1e-9)

    # Stochastic %K oversold normalized
    stoch_oversold = (20 - data["stoch_k"]).clip(lower=0)
    stoch_norm = stoch_oversold / (stoch_oversold.max() + 1e-9)

    # RSI oversold normalized
    rsi_oversold = (45 - data["rsi"]).clip(lower=0)
    rsi_norm = rsi_oversold / (rsi_oversold.max() + 1e-9)

    # Bollinger Bands % distance from lower band (lower is better)
    bb_width = data["bb_upper"] - data["bb_lower"]
    bb_pos = (data["Close"] - data["bb_lower"]) / (bb_width + 1e-9)
    bb_norm = (1 - bb_pos).clip(0, 1)

    # MACD normalized (difference between 12 and 26 EMA)
    macd_line = data["Close"].ewm(span=12).mean() - data["Close"].ewm(span=26).mean()
    macd_norm = (macd_line - macd_line.min()) / (macd_line.max() - macd_line.min() + 1e-9)

    # Fear & Greed index normalized (0-1), default neutral 0.5 if missing
    fear_greed_norm = data["fear_greed"].fillna(50) / 100

    # Composite score including fear & greed as a soft factor
    data["composite_score"] = (
        weights["z"] * z_norm +
        weights["stoch"] * stoch_norm +
        weights["rsi"] * rsi_norm +
        weights["bb"] * bb_norm +
        weights["macd"] * macd_norm +
        weights["fear_greed"] * fear_greed_norm
    )

    # Tier 3: Entry Rule
    threshold = data["composite_score"].quantile(score_threshold_quantile)
    data["buy_signal"] = eligible & (data["composite_score"] >= threshold)

    # Exit Rule: RSI > 65 AND price above 20-day moving average
    ma20 = data["Close"].rolling(window=20).mean()
    data["raw_sell_signal"] = (data["rsi"] > 65) & (data["Close"] > ma20)

    data["sell_signal"] = False

    return data



def simulate_trades(
    data: pd.DataFrame,
    stop_loss_pct: float = 0.09,
    take_profit_multiple: float = 3.0,
    use_trailing_stop: bool = False
) -> pd.DataFrame:
    trades = []
    position = None
    entry_index = None
    trailing_stop_price = None

    # Clear existing signals for precise plotting
    data["buy_signal_real"] = False
    data["sell_signal"] = False

    for i in range(len(data)):
        current_price = data["Close"].iloc[i]
        current_date = data.index[i]

        if position is None and data["buy_signal"].iloc[i]:
            # Enter new position
            position = {
                "entry_date": current_date,
                "entry_price": current_price,
                "atr": data["atr"].iloc[i],
                "max_price": current_price  # For trailing stop
            }
            entry_index = i
            trailing_stop_price = None

            # Mark the real buy signal here
            data.at[current_date, "buy_signal_real"] = True

        elif position is not None:
            entry_price = position["entry_price"]
            atr = position["atr"]

            if current_price > position["max_price"]:
                position["max_price"] = current_price

            take_profit = current_price >= entry_price + take_profit_multiple * atr

            if use_trailing_stop:
                trailing_stop_price = position["max_price"] - atr
                trailing_stop_triggered = current_price <= trailing_stop_price
            else:
                trailing_stop_triggered = False

            stop_loss = data["Low"].iloc[i] <= entry_price * (1 - stop_loss_pct)
            raw_sell = data["raw_sell_signal"].iloc[i]

            if take_profit or stop_loss or raw_sell or trailing_stop_triggered:
                exit_reason = "Take Profit" if take_profit else \
                              "Stop Loss" if stop_loss else \
                              "Trailing Stop" if trailing_stop_triggered else \
                              "Signal Exit"

                exit_price = current_price
                if exit_reason == "Stop Loss":
                    exit_price = data["Low"].iloc[i]

                pct_return = (exit_price - entry_price) / entry_price

                position.update({
                    "exit_date": current_date,
                    "exit_price": exit_price,
                    "return": pct_return,
                    "days_held": i - entry_index,
                    "exit_reason": exit_reason
                })
                trades.append(position)
                data.at[current_date, "sell_signal"] = True
                position = None
                entry_index = None
                trailing_stop_price = None

    if position is not None:
        data.at[position["entry_date"], "buy_signal_real"] = False
        position = None  

    return pd.DataFrame(trades)


def print_performance(trades_df):
    total_trades = len(trades_df)
    win_rate = (trades_df["return"] > 0).mean()
    avg_gain = trades_df[trades_df["return"] > 0]["return"].mean()
    avg_loss = trades_df[trades_df["return"] < 0]["return"].mean()
    expectancy = win_rate * avg_gain + (1 - win_rate) * avg_loss

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
    ax.scatter(data.index[data["buy_signal_real"]], data["Close"][data["buy_signal_real"]], marker="^", color="green", label="Buy", s=100)
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


def save_indicators_to_txt(data: pd.DataFrame, filename: str = "indicators_dump.txt"):
    with open(filename, "w") as f:
        # Write header
        f.write(str(data.columns.to_list()) + "\n\n")
        # Write each row as a string line
        for idx, row in data.iterrows():
            f.write(f"{idx} : {row.to_dict()}\n")


def main():
    ticker = "BITX"
    data = download_data(ticker)
    data = compute_indicators(data)

    # Fetch and merge fear & greed index
    fear_greed_df = fetch_fear_greed_index()
    data = data.merge(fear_greed_df, left_index=True, right_index=True, how='left')
    data["fear_greed"] = data["fear_greed"].ffill()

    # Use fixed best params directly here:
    best_params = {
        "quantile": 0.4,
        "vol_spike": 1.05,
        "ema": True,
        "stop_loss_pct": 0.09,
        "take_profit_multiple": 3.0,
        "use_trailing_stop": False
    }

    data = generate_signals(
        data,
        score_threshold_quantile=best_params["quantile"],
        volume_spike_ratio=best_params["vol_spike"],
        ema_condition=best_params["ema"]
    )
    trades_df = simulate_trades(
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
