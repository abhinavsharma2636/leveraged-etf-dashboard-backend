import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import ta
import requests
from typing import Optional, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def download_data(ticker: str, years_back: int = 5, interval: str = "1d") -> pd.DataFrame:
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365 * years_back)
    
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval)
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
    data["bb_pct"] = (data["Close"] - data["bb_lower"]) / (data["bb_upper"] - data["bb_lower"] + 1e-9)

    stoch = ta.momentum.StochasticOscillator(
        high=data["High"], low=data["Low"], close=data["Close"], window=14, smooth_window=3
    )
    data["stoch_k"] = stoch.stoch()
    data["stoch_d"] = stoch.stoch_signal()

    rsi = ta.momentum.RSIIndicator(close=data["Close"], window=14)
    data["rsi"] = rsi.rsi()

    macd = ta.trend.MACD(close=data["Close"])
    data["macd_diff"] = macd.macd_diff()

    data["ema20"] = data["Close"].ewm(span=20).mean()
    data["ema50"] = data["Close"].ewm(span=50).mean()
    data["price_above_ema50"] = (data["Close"] > data["ema50"]).astype(int)

    atr = ta.volatility.AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14)
    data["atr"] = atr.average_true_range()

    data["z_score"] = (
        (data["Close"] - data["Close"].rolling(20).mean()) /
        (data["Close"].rolling(20).std() + 1e-9)
    )

    data["vol_avg30"] = data["Volume"].rolling(window=30).mean()
    data["volume_surge"] = data["Volume"] / (data["vol_avg30"] + 1e-9)

    # Dynamic RSI normalization
    data["rsi_rolling_max"] = data["rsi"].rolling(window=60).max()
    data["rsi_rolling_min"] = data["rsi"].rolling(window=60).min()
    data["rsi_dynamic_norm"] = (
        (data["rsi"] - data["rsi_rolling_min"]) /
        (data["rsi_rolling_max"] - data["rsi_rolling_min"] + 1e-9)
    ).clip(0, 1)

    return data.dropna(subset=[
        "bb_pct", "stoch_k", "stoch_d", "rsi", "macd_diff",
        "ema20", "ema50", "price_above_ema50", "atr", "z_score",
        "volume_surge", "rsi_dynamic_norm"
    ])


def create_ml_labels(data: pd.DataFrame, horizon_days: int = 7, top_pct: float = 0.3) -> pd.DataFrame:
    data["future_return"] = data["Close"].shift(-horizon_days) / data["Close"] - 1

    # Rank future returns into top X% as 1, bottom X% as 0, ignore middle
    returns = data["future_return"].dropna()
    top_cutoff = returns.quantile(1 - top_pct)
    bottom_cutoff = returns.quantile(top_pct)

    def label_row(r):
        if r >= top_cutoff:
            return 1
        elif r <= bottom_cutoff:
            return 0
        else:
            return np.nan  # skip middling returns

    data["target"] = data["future_return"].apply(label_row)
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

    # Predict only on X_test (not full data)
    data.loc[X_test.index, "ml_prediction_proba"] = model.predict_proba(X_test)[:, 1]
    data.loc[X_test.index, "ml_prediction"] = (data.loc[X_test.index, "ml_prediction_proba"] >= 0.5).astype(int)

    return model, data



def simulate_trades_from_ml(data: pd.DataFrame, stop_loss_pct=0.10, take_profit_multiple=3.0, max_hold_days=5, prob_threshold=0.80, rsi_norm_threshold=0.85):
    trades = []
    open_trades = []

    data["entry"] = np.nan
    data["exit"] = np.nan

    for i in range(len(data)):
        current_price = data["Close"].iloc[i]
        current_date = data.index[i]
        prob = data["ml_prediction_proba"].iloc[i]
        rsi_norm = data["rsi_dynamic_norm"].iloc[i]

        if prob >= prob_threshold:
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
    total_losses = -trades_df.loc[trades_df["return"] < 0, "return"].sum() * 100
    net_return = total_gains - total_losses

    print(f"Total Gains: {total_gains:.2f}%")
    print(f"Total Losses: {total_losses:.2f}%")
    print(f"Net Return (Gains - Losses): {net_return:.2f}%")


def plot_signals(data: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

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
    ticker = "NVDA"
    split_date = pd.to_datetime("2024-12-31")
    rolling_years = 3  # Train on last 3 years

    # Step 1: Download & compute indicators
    data = download_data(ticker, years_back=5)
    data = compute_indicators(data)

    # Step 2: Add Fear & Greed Index
    fg = fetch_fear_greed_index()
    data = data.merge(fg, left_index=True, right_index=True, how="left")
    data["fear_greed"] = data["fear_greed"].ffill().bfill()

    # Step 3: Create improved ML labels (top/bottom 30% of future returns)
    data = create_ml_labels(data, horizon_days=7, top_pct=0.3)

    # Step 4: Train/test split
    train_start_date = split_date - pd.DateOffset(years=rolling_years)
    train_data = data[(data.index >= train_start_date) & (data.index <= split_date)].copy()
    test_data = data[(data.index > pd.to_datetime("2024-12-31")) & (data.index <= pd.to_datetime("2025-12-31"))].copy()

    print(f"Train data from {train_data.index.min().date()} to {train_data.index.max().date()}, rows: {len(train_data)}")
    print(f"Test data from {test_data.index.min().date()} to {test_data.index.max().date()}, rows: {len(test_data)}")

    # Step 5: Feature selection (improved set)
    feature_cols = [
        "bb_pct", "stoch_k", "stoch_d", "rsi", "macd_diff",
        "price_above_ema50", "atr", "z_score", "volume_surge",
        "rsi_dynamic_norm", "fear_greed"
    ]

    train_data = train_data.dropna(subset=feature_cols + ["target"])
    test_data = test_data.dropna(subset=feature_cols)

    if len(train_data) == 0 or len(test_data) == 0:
        print("Insufficient data after cleaning.")
        return

    # Step 6: Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(train_data[feature_cols], train_data["target"])

    # Step 7: Predict on test data
    test_data["ml_prediction_proba"] = model.predict_proba(test_data[feature_cols])[:, 1]
    test_data["ml_prediction"] = (test_data["ml_prediction_proba"] >= 0.5).astype(int)

    max_confidence = test_data["ml_prediction_proba"].max()
    max_confidence_date = test_data["ml_prediction_proba"].idxmax()
    max_confidence_price = test_data.loc[max_confidence_date, "Close"]

    print(f"Highest predicted confidence on test set: {max_confidence:.4f}")
    print(f"Date of highest confidence: {max_confidence_date.date()}")
    print(f"Close price on that date: ${max_confidence_price:.2f}")

    # Step 8: Simulate trades
    trades_df, test_data = simulate_trades_from_ml(test_data)

    if not trades_df.empty:
        print(trades_df[["entry_date", "exit_date", "return", "days_held"]])
        print_performance(trades_df)
    else:
        print("No trades were made in the test period.")

    # Optional: Plot
    plot_signals(test_data)



if __name__ == "__main__":
    main()
