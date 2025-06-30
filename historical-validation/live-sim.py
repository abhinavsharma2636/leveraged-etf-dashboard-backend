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
import matplotlib.pyplot as plt
import seaborn as sns


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
    return p.parse_args()

# ─── DATA DOWNLOAD & MACRO HELPERS ──────────────────────────────────────────────
def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    # print(f"[{ticker}] raw data date range: {df.index.min()} → {df.index.max()}")
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
    vix_p20 = np.percentile(data["vix_close"], 20)
    vix_p80 = np.percentile(data["vix_close"], 80)
    data["volatility_regime"] = pd.cut(
        data["vix_close"],
        bins=[-np.inf, vix_p20, vix_p80, np.inf],
        labels=["low", "neutral", "high"]
        
    )
    data["rsi2"] = ta.momentum.rsi(data["Close"], window=2)
    data["vix_rsi2"] = ta.momentum.rsi(data["vix_close"], window=2)


    # One-hot encode and join
    regime_dummies = pd.get_dummies(data["volatility_regime"], prefix="volatility_regime")
    data = pd.concat([data, regime_dummies], axis=1)
    
    required_feats = [
    "Close", "vix_close", "atr", "rsi",
    "macd_diff", "stoch_d", "price_vs_ema50", "volume_surge"
]

    data = data.dropna(subset=required_feats)
    return data

def debug_label(df: pd.DataFrame, lookahead_days=10) -> pd.DataFrame:
    df = df.copy()
    df["target"] = 1  # Mark everything as a buy
    return df

def label_simple_rsi_reversal(
    df: pd.DataFrame,
    lookahead_days: int = 10,
    rsi_thresh: float = 30,
    min_mfe: float = 0.02,   # e.g. 2% bounce required
) -> pd.DataFrame:
    df = df.copy()
    df["target"] = 0
    df["mfe"] = np.nan

    for i in range(len(df) - lookahead_days):
        row = df.iloc[i]
        entry_price = row["Close"]

        # Only label if RSI is deeply oversold
        if row["rsi"] > rsi_thresh:
            continue

        future = df.iloc[i+1 : i+1+lookahead_days]
        max_high = future["High"].max()
        mfe = (max_high - entry_price) / entry_price

        df.at[df.index[i], "mfe"] = mfe

        if mfe >= min_mfe:
            df.at[df.index[i], "target"] = 1

    return df





def label_stage2_breakout(df: pd.DataFrame, lookahead_days: int, min_gain=0.05) -> pd.DataFrame:
    d = df.copy()
    labels = []

    for i in range(len(d)):
        if i + lookahead_days >= len(d):
            labels.append(np.nan)
            continue

        row = d.iloc[i]
        price = row["Close"]
        ema200 = row["ema200"]

        # Stage 2 breakout = price above EMA200 and positive trend
        if price <= ema200 or row["price_vs_ema200"] < 0:
            labels.append(0)
            continue

        future = d.iloc[i+1:i+1+lookahead_days]
        max_high = future["High"].max()
        gain = (max_high - price) / price

        labels.append(1 if gain >= min_gain else 0)

    d["target"] = labels
    return d.dropna(subset=["target"])

def label_stage2_confirmed(df: pd.DataFrame, lookahead_days: int = 10, min_gain: float = 0.05) -> pd.DataFrame:
    d = df.copy()
    labels = []

    for i in range(len(d)):
        if i + lookahead_days >= len(d):
            labels.append(np.nan)
            continue

        row = d.iloc[i]
        price = row["Close"]
        ema200 = row["ema200"]
        
        # Core Stage 2 conditions
        if price <= ema200 or row["price_vs_ema200"] < 0:
            labels.append(0)
            continue

        # Confirmation checks: any of the following must be true
        confirm = (
            row.get("volume_surge", 0) > 0.3 or
            row.get("macd_diff", 0) > 0 or
            row.get("is_hammer", 0) == 1 or
            row.get("is_bullish_engulfing", 0) == 1
        )

        if not confirm:
            labels.append(0)
            continue

        # Lookahead for price breakout
        future = d.iloc[i+1:i+1+lookahead_days]
        max_high = future["High"].max()
        gain = (max_high - price) / price

        labels.append(1 if gain >= min_gain else 0)

    d["target"] = labels
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

# ─── ENRTY SIMULATION────────────────────────────────────────────
def simulate_entry_only(model, df, feat_cols, lookahead_days, percentile_thresh):
    df = df.copy()

    if df.empty:
        return pd.DataFrame()
    df["proba"] = model.predict_proba(df[feat_cols])[:, 1]

    # Dynamic threshold based on percentile
    if percentile_thresh is not None:
        threshold = np.percentile(df["proba"], percentile_thresh)
        df = df[df["proba"] >= threshold]
    
    results = []
    for i in range(len(df)):
        if i + lookahead_days >= len(df):
            continue

        entry_date = df.index[i]
        exit_date = df.index[i + lookahead_days]
        entry_price = df.iloc[i]["Close"]
        future = df.iloc[i+1:i+1+lookahead_days]

        max_high = future["High"].max()
        min_low = future["Low"].min()
        final_close = future.iloc[-1]["Close"]
        
        mfe = (max_high - entry_price) / entry_price
        mae = (min_low - entry_price) / entry_price
        final_return = (final_close - entry_price) / entry_price

        results.append({
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_price": entry_price,
        "final_return": final_return,
        "mfe": mfe,
        "mae": mae,
        "proba": df.iloc[i]["proba"]
        })

    return pd.DataFrame(results)

def simulate_live_inference(df, feat_cols, models, lookahead_days=10):
    df = df.copy()
    results = []

    for i in range(lookahead_days, len(df) - lookahead_days):
        today = df.index[i]
        row = df.iloc[i]

        # Infer regime for the current day only
        vix = row["vix_close"]
        if vix > 30:
            model = models["high"]
            threshold = 0.2
        elif vix > 20:
            model = models["caution"]
            threshold = 0.35
        else:
            model = models["low"]
            threshold = 0.4

        # Skip if any required feature is missing
        if row[feat_cols].isnull().any():
            continue

        x = df.iloc[i:i+1][feat_cols]
        proba = model.predict_proba(x)[0, 1]
        if threshold and proba < threshold:
            continue

        # Forward simulate
        future = df.iloc[i+1:i+1+lookahead_days]
        entry_price = row["Close"]
        max_high = future["High"].max()
        min_low = future["Low"].min()
        final_close = future.iloc[-1]["Close"]

        results.append({
            "entry_date": today,
            "exit_date": df.index[i + lookahead_days],
            "entry_price": entry_price,
            "final_return": (final_close - entry_price) / entry_price,
            "mfe": (max_high - entry_price) / entry_price,
            "mae": (min_low - entry_price) / entry_price,
            "proba": proba,
            "regime": "high" if vix > 30 else "caution" if vix > 20 else "low"
        })

    return pd.DataFrame(results)



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

def analyze_entry_quality_by_ticker(combined_entries: pd.DataFrame, show_charts: bool = True, top_n_examples: int = 1):
    """
    Evaluate entry quality for each ticker in combined_entries.
    Optionally show charts for high-confidence entries.
    """

    tickers = combined_entries["ticker"].unique()
    all_summaries = []

    for t in tickers:
        sub = combined_entries[combined_entries["ticker"] == t].copy()
        sub["win"] = sub["final_return"] > 0
    

        stats = {
        "ticker": t,
        "count": len(sub),
        "win_rate": sub["win"].mean(),
        "avg_return": sub["final_return"].mean(),
        "std_return": sub["final_return"].std(),
        "net_return": sub["final_return"].sum()
        }

        all_summaries.append(stats)

    # Final Summary Table
    summary_df = pd.DataFrame(all_summaries).sort_values("win_rate", ascending=False)
    print("\n=== OVERALL SUMMARY ===")
    print(summary_df.to_string(index=False))

    return summary_df
# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    yr = args.test_year

# ─── Feature columns ─────────────────────────────────────────────────────
    feat_cols = [
        "atr", "vix_close", "macd_diff",
        "rsi", "volume_surge", "price_vs_ema50", "stoch_d",
        "volatility_regime_low", "volatility_regime_neutral", "volatility_regime_high"
    ]

    dip_feat_cols = [
         "atr", "vix_close", "macd_diff",
        "rsi", "volume_surge", "price_vs_ema50", "stoch_d",
        "volatility_regime_low", "volatility_regime_neutral", "volatility_regime_high"
    ]

    # ─── Define training vs. testing windows ─────────────────────────────────
    train_start = (datetime.date(yr, 1, 1) - datetime.timedelta(days=365 * 16)).isoformat()
    train_end   = f"{yr-1}-12-31"
    test_start  = f"{yr}-01-01"
    test_end    = f"{yr}-12-31"

    print("Downloading macro data…")
    fear = get_fear_greed()
    vix = get_vix(train_start, test_end)
    # print(f"[VIX] Downloaded VIX from {train_start} to {test_end} → {len(vix)} rows")
    # print(vix.head())
    # print(vix.tail())
    # print("VIX Max:", vix.max())

    # ─── Build regime-aware training sets ─────────────────────────────────────
    print("Building training set…")
    trains = []
    dip_rsi_labeled = []
    confirmed_caution_labeled = []

    for t in training_tickers:
        # print(f"[{t}] requesting from {train_start} to {test_end}")
        all_data    = download_data(t, train_start, test_end)
        
        all_feats   = compute_features(all_data, fear, vix)
        # print(f"[{t}] all_feats date range: {all_feats.index.min()} → {all_feats.index.max()}")
        train_feats = all_feats.loc[:train_end].copy()
        #print(f"[{t}] train_feats rows: {len(train_feats)}")


        train_feats["regime"] = train_feats["vix_close"].apply(
            lambda x: (
                "high" if x > 30 else
                "caution" if x > 20 else
                "low"
            )
        )

        labeled_parts = []
        for regime, sub in train_feats.groupby("regime"):
            if regime == "high":
                labeled = label_simple_rsi_reversal(sub, lookahead_days=10)
            elif regime == "caution":
                labeled = label_stage2_confirmed(sub, lookahead_days=10)
            else:
                labeled = label_stage2_breakout(sub, lookahead_days=10)


            labeled["label_type"] = regime
            labeled_parts.append(labeled)

        lbl = pd.concat(labeled_parts)
        trains.append(lbl)

        # Extract only high-VIX RSI reversal labels for dip-only model
        high_vol = train_feats[train_feats["vix_close"] > 30]

        dip_labeled = label_simple_rsi_reversal(sub, lookahead_days=10)

        dip_rsi_labeled.append(dip_labeled)

        # Extract only VIX 20–30 range for confirmed breakout training
        caution_vol = train_feats[(train_feats["vix_close"] > 20) & (train_feats["vix_close"] <= 30)]
        confirmed_labeled = label_stage2_confirmed(caution_vol, lookahead_days=10)
        confirmed_caution_labeled.append(confirmed_labeled)


    train_df = pd.concat(trains).dropna(subset=feat_cols)
    dip_train_df = pd.concat(dip_rsi_labeled).dropna(subset=dip_feat_cols)
    caution_train_df = pd.concat(confirmed_caution_labeled).dropna(subset=feat_cols)

    # Oversample dip cases in main model (optional, keep or remove)
    dip_cases = train_df[
        (train_df["rsi"] < 35) &
        (train_df["price_vs_ema50"] < -0.05) &
        (train_df["vix_close"] > 25)
    ]
    train_df = pd.concat([train_df, dip_cases, dip_cases])  # 2x boost



    # ─── Train models ────────────────────────────────────────────────────────
    print(f"Training main model on {len(train_df)} rows…")
    model = train_model(train_df, feat_cols)
    train_df["proba"] = model.predict_proba(train_df[feat_cols])[:, 1]

    print(f"Training dip model on {len(dip_train_df)} rows…")
    dip_model = train_model(dip_train_df, dip_feat_cols)
    dip_train_df["proba"] = dip_model.predict_proba(dip_train_df[dip_feat_cols])[:, 1]

    print(f"Training caution model on {len(caution_train_df)} rows…")
    caution_model = train_model(caution_train_df, feat_cols)
    caution_train_df["proba"] = caution_model.predict_proba(caution_train_df[feat_cols])[:, 1]


    # Print stats by class
    print("\n=== Dip Model Confidence by Class ===")
    print(dip_train_df.groupby("target")["proba"].describe())

    print("\n=== Main (Low VIX) Model Confidence by Class ===")
    print(train_df.groupby("target")["proba"].describe())

    print("\n=== Caution (VIX 20–30) Model Confidence by Class ===")
    print(caution_train_df.groupby("target")["proba"].describe())

    # ─── Out-of-sample simulation by regime ──────────────────────────────────
    all_entries = []
    for t in args.test_tickers:
        print(f"\n=== TESTING {t} on {yr} ===")
        all_data   = download_data(t, train_start, test_end)
        all_feats  = compute_features(all_data, fear, vix)
        test_feats = all_feats.loc[test_start:]

        # Create regime-specific model mapping
        models = {
            "high": dip_model,
            "caution": caution_model,
            "low": model
        }

        # Dynamically simulate one day at a time with regime detection
        entries = simulate_live_inference(
            df=test_feats,
            feat_cols=feat_cols,
            models=models,
            lookahead_days=10
        )

        if not entries.empty:
            entries["ticker"] = t
            all_entries.append(entries)
        else:
            print(f"{t}: no trades made")

    if all_entries:
        combined_entries = pd.concat(all_entries).reset_index(drop=True)
        combined_entries.to_csv("trade_log.csv", index=False)
        summary = analyze_entry_quality_by_ticker(combined_entries, show_charts=False, top_n_examples=1)



if __name__ == "__main__":
    main()
