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
import xgboost as xgb



def download_data(ticker: str, years_back: int = 5, interval: str = "1d") -> pd.DataFrame:
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365 * years_back)
    
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna()

def get_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d")
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix[["Close"]].rename(columns={"Close": "vix_close"}).dropna()
    vix["vix_ema5"] = vix["vix_close"].ewm(span=5).mean()
    vix["vix_ema10"] = vix["vix_close"].ewm(span=10).mean()
    vix["vix_uptrend"] = (vix["vix_ema5"] > vix["vix_ema10"]).astype(int)
    vix["vix_drift_up"] = (vix["vix_close"] > vix["vix_close"].rolling(3).min()).astype(int)

    return vix


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


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # --- EMAs for trend ---
    df["ema20"] = df["Close"].ewm(span=20).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()
    df["ema200"] = df["Close"].ewm(span=200).mean()
    df["price_above_ema200"] = (df["Close"] > df["ema200"]).astype(int)
    df["ema_stack_score"] = ((df["ema20"] > df["ema50"]).astype(int) +
                             (df["ema50"] > df["ema200"]).astype(int))

    # --- RSI and RSI Recovery ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_min_20"] = df["rsi"].rolling(20).min()
    df["rsi_max_20"] = df["rsi"].rolling(20).max()
    df["rsi_dynamic_norm"] = (df["rsi"] - df["rsi_min_20"]) / (df["rsi_max_20"] - df["rsi_min_20"])
    df["rsi_dynamic_norm"] = df["rsi_dynamic_norm"].clip(0, 1)

    # --- Bollinger Bands ---
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_distance"] = (df["bb_upper"] - df["bb_lower"]) / df["Close"]
    df["near_upper_bb"] = (df["Close"] > df["bb_upper"] * 0.98).astype(int)

    # --- RSI + BB Combo ---
    df["rsi_bb_combo"] = df["rsi"] * (df["Close"] - bb_mid) / bb_std

    # --- MACD ---
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["macd_diff"] = macd - signal
    df["macd_bearish"] = (df["macd_diff"] < 0).astype(int)

    # --- Volume Surge ---
    df["volume_surge"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # --- ATR (basic version) ---
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # --- Breakout Confirmation ---
    df["prev_high_20"]       = df["High"].rolling(20).max()
    df["breakout_confirmed"] = (df["Close"] > df["prev_high_20"].shift(1)).astype(int)


     # --- Stochastic D ---
    low_14 = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()
    k = 100 * (df["Close"] - low_14) / (high_14 - low_14)
    df["stoch_d"] = k.rolling(3).mean()

    return df

def create_ml_labels(data, horizon_days=7, min_return=0.03):
    data["future_return"] = data["Close"].shift(-horizon_days) / data["Close"] - 1
    data["target"] = (data["future_return"] > min_return).astype(int)
    return data.dropna(subset=["target"])


def simulate_trades_from_ml(
    data: pd.DataFrame,
    stop_loss_pct=0.10,
    trailing_stop_pct=0.05,
    take_profit_multiple=3.0,
    base_max_hold_days=5,
    prob_threshold=0.85,
    rsi_exit_threshold=75,
    cooldown_days: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = []
    open_trades = []
    data["entry"] = np.nan
    data["exit"] = np.nan
    last_exit_date: Optional[pd.Timestamp] = None   # track when we last closed


    for i in range(len(data)):
        current_price = data["Close"].iloc[i]
        current_date = data.index[i]
        prob = data["ml_prediction_proba"].iloc[i]
        rsi = data["rsi"].iloc[i]

        if prob >= prob_threshold and not open_trades:
            # only open if we've waited long enough since last exit
            if last_exit_date is None or (current_date - last_exit_date).days >= cooldown_days:
            # Prevent duplicate buy if a trade is already open
                # Log all relevant indicators at buy
                indicators = [
                "rsi", "rsi_dynamic_norm", "rsi_bb_combo",
                "macd_diff", "macd_bearish",
                "volume_surge", "atr",
                "near_upper_bb", "bb_distance",
                "price_above_ema200", "ema_stack_score",
                "breakout_confirmed"
            ]

                log = {k: data[k].iloc[i] for k in indicators}
                print(f"\n📈 Buy Signal on {current_date.date()} | Confidence: {prob:.4f}")
                for k, v in log.items():
                    print(f"{k:>20}: {v:.4f}")

                open_trades.append({
                    "entry_date": current_date,
                    "entry_price": current_price,
                    "atr": data["atr"].iloc[i],
                    "max_price": current_price,
                    "entry_index": i,
                    "ml_confidence": prob
                })
                data.loc[current_date, "entry"] = current_price

        # Check exits
        trades_to_close = []
        for idx, trade in enumerate(open_trades):
            entry_price = trade["entry_price"]
            atr = trade["atr"]
            max_price = trade["max_price"]

            # Update trailing max
            if current_price > max_price:
                open_trades[idx]["max_price"] = current_price

            trailing_stop = max_price * (1 - trailing_stop_pct)
            take_profit_price = entry_price + take_profit_multiple * atr
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            days_held = (current_date - trade["entry_date"]).days
            max_hold_days = base_max_hold_days + int(atr * 100)  # Volatility-scaled hold time

            if (
                current_price <= stop_loss_price or
                current_price <= trailing_stop or
                current_price >= take_profit_price or
                rsi >= rsi_exit_threshold or
                days_held >= max_hold_days
            ):
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
                last_exit_date = current_date   # ← update last exit here!


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

    ax2 = ax.twinx()
    ax2.plot(data.index, data["vix_close"], label="VIX", color="gray", alpha=0.3)
    ax2.set_ylabel("VIX", color="gray")
    ax2.tick_params(axis='y', labelcolor='gray')


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
    tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "AMD", "NFLX", "INTC"]
    split_date = pd.to_datetime("2024-12-31")
    rolling_years = 3

    for ticker in tickers:
        print(f"\n===== Running model for {ticker} =====")
        data = download_data(ticker, years_back=5)
        data = compute_indicators(data)

        vix_data = get_vix_data(start_date=data.index.min().strftime("%Y-%m-%d"),
                                end_date=data.index.max().strftime("%Y-%m-%d"))
        data = data.merge(vix_data, left_index=True, right_index=True, how="left")
        data["vix_close"] = data["vix_close"].ffill()
        data["vix_uptrend"] = data["vix_uptrend"].fillna(0)
        data["vix_spike"] = (data["vix_close"].pct_change().rolling(3).max() > 0.15).astype(int)

        fg = fetch_fear_greed_index()
        data = data.merge(fg, left_index=True, right_index=True, how="left")
        data["fear_greed"] = data["fear_greed"].ffill().bfill()
        data["fear_extreme"] = (data["fear_greed"] < 25).astype(int)

        data = create_ml_labels(data, horizon_days=7, min_return=0.03)

        train_start_date = split_date - pd.DateOffset(years=rolling_years)
        train_data = data[(data.index >= train_start_date) & (data.index <= split_date)].copy()
        test_data = data[(data.index > pd.to_datetime("2024-12-31")) & (data.index <= pd.to_datetime("2025-12-31"))].copy()

        print(f"Train data: {len(train_data)}, Test data: {len(test_data)}")
        feature_cols = [
        "rsi", "rsi_dynamic_norm", "rsi_bb_combo",
        "macd_diff", "macd_bearish",
        "volume_surge", "atr",
        "near_upper_bb", "bb_distance",
        "price_above_ema200", "ema_stack_score",
        "breakout_confirmed"
        ]


        train_data = train_data.dropna(subset=feature_cols + ["target"])
        train_data = train_data[
            ~(
                (train_data["rsi"] > 68) &
                (train_data["rsi_dynamic_norm"] > 0.85) &
                (train_data["stoch_d"] > 80) &
                (train_data["Close"] > train_data["bb_upper"] * 0.98))
        ]

        test_data = test_data.dropna(subset=feature_cols)

        if len(train_data) == 0 or len(test_data) == 0:
            print("Insufficient data after cleaning.")
            continue

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False
        )
        model.fit(train_data[feature_cols], train_data["target"])

        data = data.dropna(subset=feature_cols)
        data["ml_prediction_proba"] = model.predict_proba(data[feature_cols])[:, 1]
        train_preds = model.predict_proba(train_data[feature_cols])[:, 1]
        threshold = np.percentile(train_preds, 90)
        entry_conditions = (
            (data["ml_prediction_proba"] >= threshold)
            & (data["price_above_ema200"]  == 1)
            & (data["ema_stack_score"]     >= 2))

        print(f"Entry opportunities: {entry_conditions.sum()} of {len(data)} days")

        test_data["ml_prediction_proba"] = model.predict_proba(test_data[feature_cols])[:, 1]
        test_data["ml_prediction"] = (test_data["ml_prediction_proba"] >= 0.5).astype(int)

        max_confidence = test_data["ml_prediction_proba"].max()
        max_confidence_date = test_data["ml_prediction_proba"].idxmax()
        max_confidence_price = test_data.loc[max_confidence_date, "Close"]

        print(f"Max confidence: {max_confidence:.4f} on {max_confidence_date.date()} at ${max_confidence_price:.2f}")

        trades_df, test_data = simulate_trades_from_ml(test_data)

        if not trades_df.empty:
            print(trades_df[["entry_date", "exit_date", "return", "days_held"]])
            print_performance(trades_df)
        else:
            print("No trades made.")


if __name__ == "__main__":
    main()
