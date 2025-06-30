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
    vix_p20 = np.percentile(data["vix_close"], 20)
    vix_p80 = np.percentile(data["vix_close"], 80)
    data["volatility_regime"] = pd.cut(
        data["vix_close"],
        bins=[-np.inf, vix_p20, vix_p80, np.inf],
        labels=["low", "neutral", "high"]
    )

    # One-hot encode and join
    regime_dummies = pd.get_dummies(data["volatility_regime"], prefix="volatility_regime")
    data = pd.concat([data, regime_dummies], axis=1)
    
    return data.dropna()

# ─── LABELING ───────────────────────────────────────────────────────────────────
def label_data(df: pd.DataFrame, profit_target: float, stop_loss: float, lookahead_days: int) -> pd.DataFrame:
    d = df.copy()
    labels = []

    for i in range(len(d)):
        if i + lookahead_days >= len(d):
            labels.append(np.nan)
            continue

        entry_price = d.iloc[i]["Close"]
        future = d.iloc[i+1:i+1+lookahead_days]

        hit_target = False
        stopped_out = False

        for _, row in future.iterrows():
            low = row["Low"]
            high = row["High"]

            if low <= entry_price * (1 - stop_loss):
                stopped_out = True
                break  # stop loss hit before profit
            if high >= entry_price * (1 + profit_target):
                hit_target = True
                break  # profit hit first

        labels.append(1 if hit_target else 0)

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

# ─── TRADE SIMULATION WITH ALL EXITS ────────────────────────────────────────────
def simulate_trades(
    model,
    df: pd.DataFrame,
    feature_cols: List[str],
    profit_target: float,
    stop_loss: float,
    max_duration: int,
    enable_mom: bool,
    enable_ind: bool,
    enable_trail: bool,
    trail_pct: float,
    min_hold_days: int,
    train_df: pd.DataFrame,
    meta_model, 
    meta_features
) -> pd.DataFrame:
    df = df.copy()
    df["proba"] = model.predict_proba(df[feature_cols])[:, 1]
    # ────────────────────────────────────────────────────────────────
    train_probas = model.predict_proba(train_df[feature_cols])[:, 1]
    dynamic_thresh = np.percentile(train_probas, 90)
    print(f"Dynamic threshold (top 20%): {dynamic_thresh:.4f}")
    # 1) base signals by model confidence
    high_conf = df[df["proba"] >= dynamic_thresh].index

    # 2) universal “oversold” or “deep‐dip” signals
    oversold = df[df["rsi"] < 30].index
    deep_dip = df[df["price_vs_ema50"] < -0.05].index

    # 3) union them
    signals = high_conf.union(oversold).union(deep_dip)
    trades = []

    for entry in signals:
        row = df.loc[entry]
        proba = row["proba"]
        # ─── Meta-model prediction to validate trade ─────────────────────────────
        row_meta_input = row[meta_features].values.reshape(1, -1)
        meta_prob = meta_model.predict_proba(row_meta_input)[0, 1]



        # if meta_prob < 0.5:
        #     continue  # suppress trades that meta-model flags as likely to fail
        # ─── Meta-model prediction only in high-vol regime ───────────────────────
        regime = (
            "high_vol" if row.get("volatility_regime_high", 0) == 1 else
            "low_vol" if row.get("volatility_regime_low", 0) == 1 else
            "neutral"
        )

        if regime == "high_vol":
            row_meta_input = row[meta_features].values.reshape(1, -1)
            meta_prob = meta_model.predict_proba(row_meta_input)[0, 1]
            if meta_prob < 0.5:
                continue  # suppress low-quality trades only in turbulent regimes
            
        # Determine regime from one-hot columns
        if row.get("volatility_regime_high", 0) == 1 and proba >= dynamic_thresh:
            # Only allow if it's a deep oversold panic dip
            if not (row.rsi < 30 and row.price_vs_ema50 < -0.1):
                continue  # skip this false high-confidence trade

        # if below threshold, skip—except allow oversold/dip in quiet (low-vol) regimes
        if proba < dynamic_thresh:
            is_dip      = row["price_vs_ema50"] < -0.05
            is_oversold = row["rsi"] < 30
            if not (regime == "low_vol" and (is_dip or is_oversold)):
                continue   # normal filter

        score = 0
        if row["price_vs_ema50"] > 0:
            score += 1
        if row["vix_close"] < 25 and row["fear_greed"] > 30:
            score += 1
        if row["macd_diff"] > 0 and row["rsi"] > 50:
            score += 1
        if row["is_hammer"] or row["is_bullish_engulfing"]:
            score += 1
        if row["volume_surge"] > 0.2:
            score += 1

        if score < 3:
            is_dip_oversold = (row.rsi < 30 and row.price_vs_ema50 < -0.05)
            # only let dip/oversold entries through in high-volatility regimes
            if not (is_dip_oversold and regime == "high_vol"):
                continue

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

            if low <= entry_price * (1 - stop_loss):
                exit_price, exit_type, exit_date = entry_price * (1 - stop_loss), "stop_loss", dt
                break

            if days_held < min_hold_days:
                continue

            if high >= entry_price * (1 + profit_target):
                exit_price, exit_type, exit_date = entry_price * (1 + profit_target), "profit_target", dt
                break

            if rsi > 70 and close > entry_price:
                exit_price, exit_type, exit_date = close, "rsi_exit", dt
                break

            if prev_macd > 0 and macd < 0 and close > entry_price:
                exit_price, exit_type, exit_date = close, "macd_reversal", dt
                break

            if enable_trail and close <= highest * (1 - trail_pct):
                exit_price, exit_type, exit_date = close, "trailing_stop", dt
                break

            if enable_ind and (prev_macd > 0 and macd < 0):
                exit_price, exit_type, exit_date = close, "indicator_exit", dt
                break

            if enable_mom and high <= highest and vol < prev_vol:
                exit_price, exit_type, exit_date = close, "momentum_exit", dt
                break

            if days_held >= max_duration:
                exit_price, exit_type, exit_date = close, "max_duration", dt
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
    yr = args.test_year

    # ─── Define training vs. testing windows ─────────────────────────────────
    train_start = (datetime.date(yr, 1, 1) - datetime.timedelta(days=365 * 16)).isoformat()
    train_end   = f"{yr-1}-12-31"      # end training the day before test year
    test_start  = f"{yr}-01-01"
    test_end    = f"{yr}-12-31"

    print("Downloading macro data…")
    fear = get_fear_greed()
    # VIX needs to cover both periods for feature alignment
    vix = get_vix(train_start, test_end)


    fear_train = fear.loc[:train_end]
    vix_train  = vix.loc[:train_end]
    fear_test  = fear.loc[test_start:]
    vix_test   = vix.loc[test_start:]

    # ─── Build training set ─────────────────────────────────────────────────────
    print("Building training set…")
    trains = []
    for t in training_tickers:
        # 1) Download full price history for this ticker
        all_data    = download_data(t, train_start, test_end)

        # 2) Compute features on entire span
        all_feats   = compute_features(all_data, fear, vix)

        # 3) Slice into train vs test
        train_feats = all_feats.loc[:train_end]
        # (we’ll use test_feats later in the simulation loop)
        # test_feats  = all_feats.loc[test_start:]  

        # 4) Label only the training portion
        lbl = label_data(
            train_feats,
            profit_target=args.profit_target,
            stop_loss=args.stop_loss,
            lookahead_days=args.max_duration
        )
        trains.append(lbl)

    train_df = pd.concat(trains).dropna()

    # Filter out high-volatility regime rows from training
    train_df = train_df[train_df["volatility_regime_high"] != 1]



    # Emphasize dip buying in training
    dip_cases = train_df[
        (train_df["rsi"] < 35) &
        (train_df["price_vs_ema50"] < -0.05) &
        (train_df["vix_close"] > 25)
    ]
    train_df = pd.concat([train_df, dip_cases, dip_cases])

    feat_cols = [
        "atr", "vix_close", "fear_greed", "macd_diff",
        "rsi", "volume_surge", "price_vs_ema50", "stoch_d",
         "volatility_regime_low", "volatility_regime_neutral", "volatility_regime_high"
    ]

    print(f"Training on {len(train_df)} rows…")
    model = train_model(train_df, feat_cols)
    meta_features = [
    "proba", "rsi", "price_vs_ema50", "price_vs_ema200", "macd_diff",
    "volatility_regime_low", "volatility_regime_neutral", "volatility_regime_high"
]

    print("Training meta-model…")
    meta_train_df = train_df.copy()
    meta_train_df["proba"] = model.predict_proba(meta_train_df[feat_cols])[:, 1]
    meta_train_df["meta_label"] = meta_train_df["target"]

    meta_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
    meta_model.fit(meta_train_df[meta_features], meta_train_df["meta_label"])

    # ─── Out-of-sample simulation ────────────────────────────────────────────────
    all_trades = []
    for t in args.test_tickers:
        print(f"\n=== TESTING {t} on {yr} ===")

        # Download & feature-engineer once
        all_data   = download_data(t, train_start, test_end)
        all_feats  = compute_features(all_data, fear, vix)

        # Now slice only the test portion
        test_feats = all_feats.loc[test_start:].dropna(subset=feat_cols)

        trades = simulate_trades(
            model, test_feats, feat_cols,
            args.profit_target, args.stop_loss,
            args.max_duration,
            args.enable_momentum, args.enable_indicator,
            args.enable_trailing, args.trailing_stop,
            args.min_hold_days, train_df, meta_model, meta_features
        )
        expand_trade_metrics(trades)

        if not trades.empty:
            trades["ticker"] = t
            all_trades.append(trades)

    # ─── Summary ────────────────────────────────────────────────────────────────
    if all_trades:
        combined = pd.concat(all_trades).reset_index(drop=True)
           # ─── Merge entry‐day features into the combined trade log ─────────────
    # 1) Rebuild a single feature table for your test period
        feat_list = []
        for t in args.test_tickers:
            # download & feature-engineer
            all_price = download_data(t, train_start, test_end)
            feats = compute_features(all_price, fear, vix)

            # Name the index so reset_index() creates a "date" column
            feats.index.name = "date"
            feats = feats.reset_index()

            feats["ticker"] = t
            feats["proba"]  = model.predict_proba(feats[feat_cols])[:, 1]
            feat_list.append(feats)

        features_df = pd.concat(feat_list, ignore_index=True)

    # 3) merge into your combined DataFrame on ticker + entry_date
        combined = combined.merge(
            features_df[[
                "ticker", "date",
                "rsi", "price_vs_ema50", "vix_close", "fear_greed",
                "proba"
            ]],
            left_on=["ticker", "entry_date"],
            right_on=["ticker", "date"],
            how="left"
        ).drop(columns=["date"])

        combined['regime'] = combined['vix_close'].apply(lambda x:
        'high_vol' if x > 25 else 'low_vol' if x < 15 else 'neutral'
        )
        print(combined.groupby('regime')['return'].agg(['count', 'mean', 'std', lambda x: (x > 0).mean()]))

        df = combined.copy()
        df["regime"] = df["vix_close"].apply(lambda x: "high_vol" if x > 25 else "low_vol" if x < 15 else "neutral")
        df["confidence_bin"] = pd.qcut(df["proba"], q=4, labels=["low", "mid", "high", "very_high"])
        print(df.groupby(["regime", "confidence_bin"])["return"].agg(["count", "mean", lambda x: (x>0).mean()]))

        # Now `combined` has your entry‐day features alongside return, exit_type, etc.
        combined.to_csv("trade_log_with_features.csv", index=False)

        summary = combined.groupby("ticker").agg(
            total_trades=("return", "count"),
            win_rate=("return", lambda x: (x >= 0).mean()),
            net_return=("return", "sum"),
            sharpe_ratio=("return", lambda x: np.nan if x.std() == 0 else x.mean() / x.std())
        ).sort_values(by="net_return", ascending=False)

        print("\n=== Trade Summary by Ticker ===")
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
        results = []
        total_trades = len(combined)

         # → insert percentile‐band logic here:
        combined['pct_rank'] = combined['proba'].rank(pct=True) * 100
        avg_pct = combined['pct_rank'].mean()
        low_cut, high_cut = avg_pct - 10, avg_pct + 10
        in_band = combined[
            (combined['pct_rank'] >= low_cut) &
            (combined['pct_rank'] <= high_cut)
        ]
        print(f"Average confidence percentile: {avg_pct:.1f}")
        print(f"  Filtering to {low_cut:.0f}–{high_cut:.0f} band → {len(in_band)} trades")
        print(f"  Win‐rate in band: {(in_band['return']>0).mean():.2%}")

        for pct in range(50, 99, 5):
            cutoff = np.percentile(combined['proba'], pct)
            sub = combined[combined['proba'] >= cutoff]
            n = len(sub)
            if n == 0:
                continue
            win_rate = (sub['return'] >= 0).mean()
            avg_ret  = sub['return'].mean()
            # e.g. expected return per trade
            exp_ret = avg_ret * win_rate
            results.append({
                'percentile': pct,
                'n_trades': n,
                'win_rate': win_rate,
                'avg_return': avg_ret,
                'exp_return': exp_ret
            })

        df = pd.DataFrame(results)

        # Plot Win Rate
        plt.figure()
        plt.plot(df['percentile'], df['win_rate'], marker='o')
        plt.xlabel('Proba Percentile Threshold')
        plt.ylabel('Win Rate')
        plt.title('Win Rate vs Confidence Threshold')
        plt.show()

        # Plot # Trades
        plt.figure()
        plt.plot(df['percentile'], df['n_trades'], marker='o')
        plt.xlabel('Proba Percentile Threshold')
        plt.ylabel('Number of Trades')
        plt.title('Trade Count vs Confidence Threshold')
        plt.show()

        # Plot Expected Return
        plt.figure()
        plt.plot(df['percentile'], df['exp_return'], marker='o')
        plt.xlabel('Proba Percentile Threshold')
        plt.ylabel('Expected Return')
        plt.title('Expected Return vs Threshold')
        plt.show()
    else:
        print("\n✓ Done → no trades to log")


if __name__ == "__main__":
    main()
