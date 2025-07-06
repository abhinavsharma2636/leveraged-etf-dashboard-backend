import pandas as pd
from .simulate_helpers import passes_entry_filters, check_exit_condition 

class TradeSimulator:
    def __init__(self, feat_cols, models):
        self.feat_cols = feat_cols
        self.models = models

    def simulate(self, df: pd.DataFrame):

        df = df.copy()
        results = []

        df["ema50_slope"] = df["ema50"].diff(3)
        df["vix_5d_slope"] = df["vix_close"].diff(5)
        df["fear_greed_slope"] = df["fear_greed"].diff(5)
        df["vix_low_20d"] = df["vix_close"].rolling(20).min()

        last_proba_by_regime = {"high": 0, "caution": 0, "low": 0}
        delta_by_regime = {"high": 0, "caution": 0, "low": 0}
        threshold_by_regime = {"high": 0.02, "caution": 0.2, "low": 0.2}

        i = 0
        while i < len(df):
            row = df.iloc[i]
            if row[self.feat_cols].isnull().any():
                i += 1
                continue

            vix = row["vix_close"]
            regime = "high" if vix > 30 else "caution" if vix > 20 else "low"
            model = self.models[regime]
            threshold = threshold_by_regime[regime]
            delta = delta_by_regime[regime]

            if not passes_entry_filters(row, regime):
                i += 1
                continue

            x = df.iloc[i:i+1][self.feat_cols]
            proba = model.predict_proba(x)[0, 1]
            if proba < threshold or proba < last_proba_by_regime[regime] + delta:
                i += 1
                continue

            entry_price = row["Close"]
            entry_date = df.index[i]
            entry_regime = regime
            entry_vix = vix
            entry_rsi = row["rsi"]

            exit_idx, exit_type = check_exit_condition(df, i, entry_price, entry_regime, i)
            exit_row = df.iloc[exit_idx]
            exit_date = df.index[exit_idx]
            exit_price = exit_row["Close"]

            results.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "final_return": (exit_price - entry_price) / entry_price,
                "mfe": (df.iloc[i:exit_idx+1]["High"].max() - entry_price) / entry_price,
                "mae": (df.iloc[i:exit_idx+1]["Low"].min() - entry_price) / entry_price,
                "proba": proba,
                "regime": entry_regime,
                "vix_at_entry": entry_vix,
                "vix_at_exit": exit_row["vix_close"],
                "regime_at_exit": "high" if exit_row["vix_close"] > 30 else "caution" if exit_row["vix_close"] > 20 else "low",
                "rsi_at_entry": entry_rsi,
                "rsi_at_exit": exit_row["rsi"],
                "macd_at_exit": exit_row["macd_diff"],
                "bb_pct_at_exit": exit_row["bb_pct"],
                "ema50_slope_exit": exit_row["ema50_slope"],
                "price_vs_ema50_entry": row["price_vs_ema50"],
                "price_vs_ema200_entry": row["price_vs_ema200"],
                "price_vs_ema50_exit": exit_row["price_vs_ema50"],
                "price_vs_ema200_exit": exit_row["price_vs_ema200"],
                "hold_days": (exit_date - entry_date).days,
                "exit_type": exit_type,
            })

            last_proba_by_regime[regime] = 0
            i = exit_idx + 1

        return pd.DataFrame(results)
