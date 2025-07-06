# labeling_functions.py
# Contains actual implementations of labeling logic

import numpy as np
import pandas as pd

class LabelingFunctions:
    def label_rsi_reversal_enhanced(self, df: pd.DataFrame, lookahead_days: int = 10) -> pd.DataFrame:
        df = df.copy()
        df["target"] = 0
        df["mfe"] = np.nan
        df["mae"] = np.nan

        df["volume_ma"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_ma"]
        df["price_change_5d"] = df["Close"].pct_change(5)
        df["vix_percentile"] = df["vix_close"].rolling(252).rank(pct=True)

        for i in range(20, len(df) - lookahead_days):
            row = df.iloc[i]
            if row["rsi"] > 45:
                continue
            if not (row["vix_close"] >= 30 and row["vix_rsi2"] >= 65) and row["vix_percentile"] < 0.8:
                continue
            if row.get("volume_ratio", 0) < 1.2:
                continue
            if row.get("price_change_5d", 0) > -0.05:
                continue
            recent_rsi = df.iloc[i - 5:i]["rsi"]
            if len(recent_rsi) >= 3 and row["rsi"] > recent_rsi.min():
                continue

            entry_price = row["Close"]
            future = df.iloc[i + 1 : i + 1 + lookahead_days]
            max_high = future["High"].max()
            min_low = future["Low"].min()

            mfe = (max_high - entry_price) / entry_price
            mae = (min_low - entry_price) / entry_price

            df.at[df.index[i], "mfe"] = mfe
            df.at[df.index[i], "mae"] = mae

            if mfe >= 0.012 and mae >= -0.18:
                df.at[df.index[i], "target"] = 1

        return df.dropna(subset=["target"])

    def label_stage2_breakout_with_exit_logic(self, df: pd.DataFrame, min_gain: float = 0.05, max_hold: int = 180) -> pd.DataFrame:
        d = df.copy()
        labels = []

        for i in range(len(d)):
            if i + 1 >= len(d):
                labels.append(np.nan)
                continue

            row = d.iloc[i]
            price = row["Close"]
            if price <= row["ema200"] or row["price_vs_ema200"] < 0:
                labels.append(0)
                continue

            label = 0
            max_high = price

            for j in range(i + 1, min(i + 1 + max_hold, len(d))):
                next_row = d.iloc[j]
                max_high = max(max_high, next_row["High"])
                gain = (max_high - price) / price

                if gain >= min_gain:
                    label = 1
                    break

                if (next_row["price_vs_ema200"] > 0 and next_row["ema50"] - d.iloc[j - 3]["ema50"] < 0.1 and
                    (next_row["macd_diff"] < 0 or next_row["rsi"] < 50)):
                    break

            labels.append(label)

        d["target"] = labels
        return d.dropna(subset=["target"])

    def label_stage2_confirmed_with_exit_logic(self, df: pd.DataFrame, min_gain: float = 0.05, max_hold: int = 180) -> pd.DataFrame:
        d = df.copy()
        labels = []

        for i in range(len(d)):
            if i + 1 >= len(d):
                labels.append(np.nan)
                continue

            row = d.iloc[i]
            price = row["Close"]

            if price <= row["ema200"] or row["price_vs_ema200"] < 0:
                labels.append(0)
                continue

            confirm = (
                row.get("volume_surge", 0) > 0.3 or
                row.get("macd_diff", 0) > 0 or
                row.get("is_hammer", 0) == 1 or
                row.get("is_bullish_engulfing", 0) == 1
            )

            if not confirm:
                labels.append(0)
                continue

            label = 0
            max_high = price

            for j in range(i + 1, min(i + 1 + max_hold, len(d))):
                next_row = d.iloc[j]
                max_high = max(max_high, next_row["High"])
                gain = (max_high - price) / price

                if gain >= min_gain:
                    label = 1
                    break

                if (next_row["price_vs_ema200"] > 0 and next_row["ema50"] - d.iloc[j - 3]["ema50"] < 0.1 and
                    (next_row["macd_diff"] < 0 or next_row["rsi"] < 50)):
                    break

            labels.append(label)

        d["target"] = labels
        return d.dropna(subset=["target"])
