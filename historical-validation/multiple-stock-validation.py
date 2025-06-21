#!/usr/bin/env python3
import argparse
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import requests
from xgboost import XGBClassifier
from typing import List
from tabulate import tabulate


# ─── USER-CONFIGURED TICKERS ─────────────────────────────────────────────────────
training_tickers = [
    # Tech (longest track records)
    "AAPL",  # public since 1980
    "MSFT",  # 1986
    "INTC",  # 1971
    "CSCO",  # 1990
    "IBM",   # 1915
    "ORCL",  # 1986

    # Semiconductors
    "TXN",   # 1960
    "ADI",   # 1980s

    # Financials
    "JPM",   # 1969
    "BAC",   # 1970
    "WFC",   # 1978
    "GS",    # 1999
    "AXP",   # 1977

    # Consumer Defensive / Staples
    "PG",    # 1891
    "KO",    # 1919
    "PEP",   # 1980s
    "WMT",   # 1972
    "COST",  # 1980s

    # Industrials
    "GE",    # 1892
    "CAT",   # 1960s
    "MMM",   # 1946
    "HON",   # 1970s

    # Healthcare
    "JNJ",   # 1944
    "PFE",   # 1980
    "MRK",   # 1978
    "ABBV",  # spin-off of ABT, use ABT for longer data
    "UNH",   # 1980s

    # Broad Market ETFs
    "SPY",   # since 1993
    "QQQ",   # 1999
    "DIA",   # 1998
    "IWM",   # 2000

    # Energy & Utilities
    "XOM",   # 1970s
    "CVX",   # 1980s
    "DUK",   # 1970s
    "NEE",   # 1970s

    # Discretionary / Retail
    "HD",    # 1981
    "LOW",   # 1979
    "TGT",   # 1970s
    "MCD",   # 1960s
    "NKE",   # 1980
    "SBUX",  # 1992
]


# ─── ARGPARSE ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="High-Confidence Entry Detection with Extended Exit Rules")
    p.add_argument("--test_year", type=int, required=True)
    p.add_argument("--test_tickers", nargs="+", default=training_tickers)
    p.add_argument("--profit_target", type=float, default=0.10)
    p.add_argument("--stop_loss", type=float, default=0.025)
    p.add_argument("--max_duration", type=int, default=25)
    p.add_argument("--min_hold_days", type=int, default=5)
    p.add_argument("--enable_momentum", action="store_true")
    p.add_argument("--enable_indicator", action="store_true")
    p.add_argument("--enable_trailing", action="store_true")
    p.add_argument("--trailing_stop", type=float, default=0.05)
    return p.parse_args()

# ─── DATA DOWNLOAD & MACRO HELPERS ──────────────────────────────────────────────
def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.rename(columns=str.capitalize, inplace=True)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def get_vix(start: str, end: str) -> pd.Series:
    v = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(v.columns, pd.MultiIndex):
        v.columns = v.columns.get_level_values(0)
    s = v["Close"].copy()
    s.name = "vix_close"
    return s

def get_fear_greed() -> pd.DataFrame:
    url = "https://api.alternative.me/fng/?limit=0"
    data = requests.get(url).json()["data"]
    rows = [{
        "date": pd.to_datetime(int(d["timestamp"]), unit="s"),
        "fear_greed": int(d["value"])
    } for d in data]
    fg = pd.DataFrame(rows)
    if isinstance(fg.columns, pd.MultiIndex):
        fg.columns = fg.columns.get_level_values(0)
    return fg.set_index("date").sort_index()

def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    body = (df["Close"] - df["Open"]).abs()
    upper_wick = df["High"] - df[["Close", "Open"]].max(axis=1)
    lower_wick = df[["Close", "Open"]].min(axis=1) - df["Low"]
    df["is_hammer"] = ((body < (df["High"] - df["Low"]) * 0.3) & (lower_wick > body * 2) & (upper_wick < body)).astype(int)
    prev_open = df["Open"].shift(1)
    prev_close = df["Close"].shift(1)
    df["is_bullish_engulfing"] = ((prev_close < prev_open) & (df["Close"] > df["Open"]) & (df["Close"] > prev_open) & (df["Open"] < prev_close)).astype(int)
    return df

def compute_features(df: pd.DataFrame, fear: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    data = df.copy()
    bb = ta.volatility.BollingerBands(data.Close, window=20, window_dev=2)
    data["bb_upper"] = bb.bollinger_hband()
    data["bb_lower"] = bb.bollinger_lband()
    data["bb_pct"] = (data.Close - data.bb_lower) / (data.bb_upper - data.bb_lower + 1e-9)
    for span in (20, 50, 200):
        data[f"ema{span}"] = data.Close.ewm(span=span, adjust=False).mean()
    data["price_vs_ema20"] = data.Close / data.ema20 - 1
    data["price_vs_ema50"] = data.Close / data.ema50 - 1
    data["price_vs_ema200"] = data.Close / data.ema200 - 1
    data["atr"] = ta.volatility.average_true_range(data.High, data.Low, data.Close, window=14)
    data["rsi"] = ta.momentum.rsi(data.Close, window=14)
    macd = ta.trend.MACD(data.Close)
    data["macd_diff"] = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(data.High, data.Low, data.Close, window=14, smooth_window=3)
    data["stoch_k"], data["stoch_d"] = stoch.stoch(), stoch.stoch_signal()
    data["vol_avg5"] = data.Volume.rolling(5).mean()
    data["volume_surge"] = (data.Volume / data.vol_avg5 - 1).clip(lower=0)
    data["trend_strength"] =  data["price_vs_ema50"] + data["price_vs_ema200"] 
    data = add_candle_patterns(data)
    data = data.join(vix, how="left").ffill()
    data = data.join(fear, how="left").ffill()
    return data.dropna()

# ─── LABELING ───────────────────────────────────────────────────────────────────
def label_data(
    df: pd.DataFrame,
    profit_target: float,
    max_drawdown: float,
    lookahead_days: int
) -> pd.DataFrame:
    d = df.copy()
    d["future_max"] = (
        d.Close.shift(-lookahead_days)
         .rolling(lookahead_days).max() / d.Close
        - 1
    )
    d["future_min"] = (
        d.Close.shift(-lookahead_days)
         .rolling(lookahead_days).min() / d.Close
        - 1
    )

    def mk(r):
        if pd.isna(r.future_max) or pd.isna(r.future_min):
            return np.nan
        return int(
            (r.future_max >= profit_target) and
            (r.future_min >= -max_drawdown)
        )

    d["target"] = d.apply(mk, axis=1)
    return d.dropna(subset=["target"])

# ─── MODEL TRAINING ─────────────────────────────────────────────────────────────
def train_model(train_df: pd.DataFrame, feature_cols: List[str]) -> XGBClassifier:
    X = train_df[feature_cols]
    y = train_df["target"]
    model = XGBClassifier(
    objective="binary:logistic",   # explicit logistic objective
    base_score=0.5,                # valid initial probability
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
    )

    model.fit(X, y)
    return model

# ─── TRADE SIMULATION WITH ALL EXITS ────────────────────────────────────────────
def simulate_trades(
    model,
    df: pd.DataFrame,
    feature_cols: List[str],
    profit_target: float,
    stop_loss: float,
    max_duration: int,
    min_hold_days: int,
    enable_mom: bool,
    enable_ind: bool,
    enable_trail: bool,
    trail_pct: float
) -> pd.DataFrame:
    df = df.copy()
    df["proba"] = model.predict_proba(df[feature_cols])[:, 1]

    
    dynamic_thresh = df["proba"].quantile(0.98)
    print(f"Dynamic threshold (top 2%): {dynamic_thresh:.4f}")
    signals = df[df["proba"] >= dynamic_thresh].index

    trades = []
    for entry in signals:

        # Realistic confirmation: skip entry if next bar doesn’t move up
        next_idx = df.index.get_loc(entry) + 1
        if next_idx >= len(df):
            continue
        if df.iloc[next_idx]["High"] <= df.loc[entry, "Close"]:
            continue  # skip if next candle doesn't confirm upward move
        entry_price = df.at[entry, "Close"]
        highest = entry_price
        prev_vol = df.at[entry, "Volume"]
        prev_macd = df.at[entry, "macd_diff"]
        exit_date = exit_price = exit_type = None

        window = df.loc[entry:]
        for i, (dt, row) in enumerate(window.iterrows()):
            high, low, close = row["High"], row["Low"], row["Close"]
            rsi, vol, macd = row["rsi"], row["Volume"], row["macd_diff"]
            days_held = i

            if high > highest:
                highest = high

            # 1) Stop loss allowed immediately
            if low <= entry_price * (1 - stop_loss):
                exit_price, exit_type, exit_date = (
                    entry_price * (1 - stop_loss), "stop_loss", dt)
                break

            if days_held < min_hold_days:
                continue

            # 2) Profit Target
            if high >= entry_price * (1 + profit_target):
                exit_price, exit_type, exit_date = (
                    entry_price * (1 + profit_target), "profit_target", dt)
                break

            # 3) RSI Overbought Exit (if profitable)
            if rsi > 70 and close > entry_price:
                exit_price, exit_type, exit_date = (
                    close, "rsi_exit", dt)
                break

            # 4) MACD Flip Exit (if profitable)
            if prev_macd > 0 and macd < 0 and close > entry_price:
                exit_price, exit_type, exit_date = (
                    close, "macd_reversal", dt)
                break

            # 3) Trailing Stop
            if enable_trail and close <= highest * (1 - trail_pct):
                exit_price, exit_type, exit_date = (
                    close, "trailing_stop", dt)
                break

            # 4) Indicator Exit
            if enable_ind and (prev_macd > 0 and macd < 0):
                exit_price, exit_type, exit_date = (
                    close, "indicator_exit", dt)
                break

            # 5) Momentum Exit
            if enable_mom and high <= highest and vol < prev_vol:
                exit_price, exit_type, exit_date = (
                    close, "momentum_exit", dt)
                break

            # 6) Max Duration
            if days_held >= max_duration:
                exit_price, exit_type, exit_date = (
                    close, "max_duration", dt)
                break

            prev_vol, prev_macd = vol, macd

        if exit_date is None:
            continue

        segment = df.loc[entry:exit_date]
        trades.append({
            "entry_date": entry,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "max_high": segment["High"].max(),
            "min_low": segment["Low"].min(),
            "return": exit_price / entry_price - 1,
            "exit_type": exit_type
        })

    return pd.DataFrame(trades)


# ─── PERFORMANCE REPORT ─────────────────────────────────────────────────────────

def expand_trade_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    if trades.empty:
        print(">>> No trades to evaluate.")
        return trades

    wins = trades[trades["return"] > 0]
    losses = trades[trades["return"] <= 0]

    win_rate = len(wins) / len(trades)
    avg_win = wins["return"].mean() if not wins.empty else 0
    avg_loss = losses["return"].mean() if not losses.empty else 0
    sharpe = trades["return"].mean() / (trades["return"].std() + 1e-9)
    hold_durations = (pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])).dt.days
    max_hold = hold_durations.max() if not hold_durations.empty else 0

    print(f"Total trades: {len(trades)}")
    print(f"Win rate:     {win_rate:.2%}")
    print(f"Net return:   {trades['return'].sum():.2f}")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Avg win:      {avg_win:.2%}")
    print(f"Avg loss:     {avg_loss:.2%}")
    print(f"Max hold:     {max_hold} days")
    print("Exit types:")
    print(trades["exit_type"].value_counts().to_string())

    return trades

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    yr    = args.test_year
    start = (datetime.date(yr, 1, 1) - datetime.timedelta(days=365*12)).isoformat()
    end   = f"{yr}-12-31"

    print("Downloading macro data…")
    fear = get_fear_greed()
    vix  = get_vix(start, end)

    print("Building training set…")
    trains = []
    for t in training_tickers:
        df    = download_data(t, start, end)
        feats = compute_features(df, fear, vix)
        lbl   = label_data(
            feats,
            profit_target=args.profit_target,
            max_drawdown=args.stop_loss,
            lookahead_days=args.max_duration
        )
        trains.append(lbl)

    train_df = pd.concat(trains).dropna()
    feat_cols = [
        "atr", "vix_close", "fear_greed", "macd_diff",
        "rsi", "volume_surge", "price_vs_ema50", "stoch_d"
    ]

    print(f"Training on {len(train_df)} rows…")
    model = train_model(train_df, feat_cols)

    all_trades = []
    for t in args.test_tickers:
        print(f"\n=== TESTING {t} on {yr} ===")
        df    = download_data(t, start, end)
        feats = compute_features(df, fear, vix)
        test  = feats[feats.index.year == yr].dropna(subset=feat_cols)

        trades = simulate_trades(
            model, test, feat_cols,
            args.profit_target,
            args.stop_loss, args.max_duration,
            args.enable_momentum, args.enable_indicator,
            args.enable_trailing, args.trailing_stop, args.min_hold_days
        )
        expand_trade_metrics(trades)

        if not trades.empty:
            trades["ticker"] = t
            all_trades.append(trades)

    if all_trades:
        combined = pd.concat(all_trades).reset_index(drop=True)
        combined.to_csv("trade_log.csv", index=False)

        summary = combined.groupby("ticker").agg(
            total_trades=("return", "count"),
            win_rate=("return", lambda x: (x >= 0).mean()),
            net_return=("return", "sum"),
            sharpe_ratio=("return", lambda x: np.nan if x.std() == 0 else x.mean() / x.std())
        ).sort_values(by="net_return", ascending=False)

        print("\n=== Trade Summary by Ticker ===")
        # Format specific columns as percentages manually
        summary_fmt = summary.copy()
        summary_fmt["win_rate"] = summary_fmt["win_rate"].apply(lambda x: f"{x:.2%}")
        summary_fmt["net_return"] = summary_fmt["net_return"].apply(lambda x: f"{x:.2%}")
        summary_fmt["sharpe_ratio"] = summary_fmt["sharpe_ratio"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

        print(tabulate(summary_fmt.reset_index(), headers="keys", tablefmt="pretty"))
        total_trades = len(combined)
        win_rate = (combined["return"] >= 0).mean()
        net_return = combined["return"].sum()

        print("\n=== FINAL RESULTS ===")
        print(f"Total Trades: {total_trades}")
        print(f"Final Win Rate: {win_rate:.2%}")
        print(f"Net Return: {net_return:.2%}")
    else:
        print("\n✓ Done → no trades to log")

    

if __name__ == "__main__":
    main()
