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

        df["entry_date"] = df.index
        df["label_type"] = "high"
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
        d["entry_date"] = d.index
        d["label_type"] = "caution"

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
        d["entry_date"] = d.index
        d["label_type"] = "low"

        # Now it's safe to print — 'target' column exists
        return d.dropna(subset=["target"])
    
    def label_stage2_breakout_triple_barrier(self, df: pd.DataFrame,
                                         lookahead_days: int = 180,
                                         profit_thresh: float = 0.20,
                                         stop_thresh: float = -0.10) -> pd.DataFrame:
        d = df.copy()
        d.index = pd.to_datetime(d.index)

        d["target"] = np.nan
        d["entry_date"] = pd.NaT
        d["exit_date"] = pd.NaT
        d["exit_type"] = "not_triggered"
        d["mfe"] = np.nan
        d["mae"] = np.nan
        d["final_return"] = np.nan
        d["label_type"] = pd.Series(dtype="object")

        entry_start = d.index.min().replace(day=1)
        entry_end = (entry_start + pd.offsets.MonthEnd(0)).normalize()
        # print(f"[🔍] Breakout Entry Window: {entry_start.date()} → {entry_end.date()}")

        for i in range(len(d)):
            row = d.iloc[i]
            row_date = row.name

            if not (entry_start <= row_date <= entry_end):
                continue
            if (
                row["Close"] > row["ema200"] and
                (d.iloc[i - 5]["ema200"] < row["ema200"] if i >= 5 else True) and
                row["rsi"] > 50 and
                row["macd_diff"] > 0
            ):
                entry_price = row["Close"]
                entry_date = row_date
                future = d.iloc[i + 1:i + 1 + lookahead_days]
                if future.empty:
                    continue

                high_prices = future["High"]
                low_prices = future["Low"]
                close_prices = future["Close"]

                hit_tp = hit_sl = False
                exit_day = future.index[-1]
                exit_type = "final_close"

                for j, (high, low) in enumerate(zip(high_prices, low_prices)):
                    if (high - entry_price) / entry_price >= profit_thresh:
                        hit_tp = True
                        exit_day = future.index[j]
                        exit_type = "profit_hit"
                        break
                    if (low - entry_price) / entry_price <= stop_thresh:
                        hit_sl = True
                        exit_day = future.index[j]
                        exit_type = "stop_hit"
                        break

                exit_price = d.loc[exit_day, "Close"] if exit_day in d.index else close_prices.iloc[-1]
                final_ret = (exit_price - entry_price) / entry_price
                max_ret = (high_prices.max() - entry_price) / entry_price
                min_ret = (low_prices.min() - entry_price) / entry_price

                d.at[entry_date, "target"] = 1 if hit_tp else 0
                d.at[entry_date, "label_type"] = "low"
                d.at[entry_date, "entry_date"] = entry_date
                d.at[entry_date, "exit_date"] = exit_day
                d.at[entry_date, "exit_type"] = exit_type
                d.at[entry_date, "final_return"] = final_ret
                d.at[entry_date, "mfe"] = max_ret
                d.at[entry_date, "mae"] = min_ret

        return d.dropna(subset=["target"])


    
    def label_stage2_confirmed_triple_barrier(self, df: pd.DataFrame,
                                          lookahead_days: int = 180,
                                          profit_thresh: float = 0.20,
                                          stop_thresh: float = -0.10) -> pd.DataFrame:
        d = df.copy()
        d.index = pd.to_datetime(d.index)

        d["target"] = np.nan
        d["entry_date"] = pd.NaT
        d["exit_date"] = pd.NaT
        d["exit_type"] = "not_triggered"
        d["mfe"] = np.nan
        d["mae"] = np.nan
        d["final_return"] = np.nan
        d["label_type"] = pd.Series(dtype="object")

        entry_start = d.index.min().replace(day=1)
        entry_end = (entry_start + pd.offsets.MonthEnd(0)).normalize()
        # print(f"[🔍] Confirmed Entry Window: {entry_start.date()} → {entry_end.date()}")

        for i in range(len(d)):
            row = d.iloc[i]
            row_date = row.name

            if not (entry_start <= row_date <= entry_end):
                continue
            if (
                row["Close"] > row["ema200"] and
                row["price_vs_ema50"] > 0.01 and
                row["ema50"] > row["ema200"] and
                (d.iloc[i - 3]["ema50"] < row["ema50"] if i >= 3 else True) and
                row["rsi"] > 55 and
                row["macd_diff"] > 0
            ):
                entry_price = row["Close"]
                entry_date = row_date
                future = d.iloc[i + 1:i + 1 + lookahead_days]
                if future.empty:
                    continue

                high_prices = future["High"]
                low_prices = future["Low"]
                close_prices = future["Close"]

                hit_tp = hit_sl = False
                exit_day = future.index[-1]
                exit_type = "final_close"

                for j, (high, low) in enumerate(zip(high_prices, low_prices)):
                    if (high - entry_price) / entry_price >= profit_thresh:
                        hit_tp = True
                        exit_day = future.index[j]
                        exit_type = "profit_hit"
                        break
                    if (low - entry_price) / entry_price <= stop_thresh:
                        hit_sl = True
                        exit_day = future.index[j]
                        exit_type = "stop_hit"
                        break

                exit_price = d.loc[exit_day, "Close"] if exit_day in d.index else close_prices.iloc[-1]
                final_ret = (exit_price - entry_price) / entry_price
                max_ret = (high_prices.max() - entry_price) / entry_price
                min_ret = (low_prices.min() - entry_price) / entry_price

                d.at[entry_date, "target"] = 1 if hit_tp else 0
                d.at[entry_date, "label_type"] = "caution"
                d.at[entry_date, "entry_date"] = entry_date
                d.at[entry_date, "exit_date"] = exit_day
                d.at[entry_date, "exit_type"] = exit_type
                d.at[entry_date, "final_return"] = final_ret
                d.at[entry_date, "mfe"] = max_ret
                d.at[entry_date, "mae"] = min_ret

        return d.dropna(subset=["target"])



    
    def label_rsi_reversal_triple_barrier(self, df: pd.DataFrame,
                                      lookahead_days: int = 180,
                                      profit_thresh: float = 0.08,
                                      stop_thresh: float = -0.20) -> pd.DataFrame:
        d = df.copy()
        d.index = pd.to_datetime(d.index)

        d["target"] = np.nan
        d["entry_date"] = pd.NaT
        d["exit_date"] = pd.NaT
        d["exit_type"] = "not_triggered"
        d["mfe"] = np.nan
        d["mae"] = np.nan
        d["final_return"] = np.nan
        d["label_type"] = pd.Series(dtype="object")

        d["volume_ma"] = d["Volume"].rolling(20).mean()
        d["volume_ratio"] = d["Volume"] / d["volume_ma"]
        d["price_change_5d"] = d["Close"].pct_change(5)
        d["vix_percentile"] = d["vix_close"].rolling(252).rank(pct=True)

        total_rows = len(d)
        if total_rows < 40:
            print(f"[⚠️] Skipping: only {total_rows} rows available")
            return d.dropna(subset=["target"])

        eval_rows = 0
        passed_all = 0
        rsi_fail = vix_fail = vol_fail = price_fail = trend_fail = 0

        # ── Step 1: restrict entry to the current month ──
        entry_cutoff_start = d.index.min().replace(day=1)
        entry_cutoff_end = (entry_cutoff_start + pd.offsets.MonthEnd(0)).normalize()

        entry_mask = (d.index >= entry_cutoff_start) & (d.index <= entry_cutoff_end)
        entry_candidates = d[entry_mask].copy()

        # print(f"[🔍] Entry window: {entry_cutoff_start.date()} → {entry_cutoff_end.date()} | {len(entry_candidates)} rows")

        # ── Step 2: run labeling logic only for those rows ──
        for i in range(5, total_rows - 1):
            row = d.iloc[i]
            row_date = row.name

            if not (entry_cutoff_start <= row_date <= entry_cutoff_end):
                continue  # skip rows outside the entry month

            if row["rsi"] > 50:
                rsi_fail += 1
                continue
            if row["vix_close"] < 25 and row["vix_percentile"] < 0.75:
                vix_fail += 1
                continue
            if row["volume_ratio"] < 1.0:
                vol_fail += 1
                continue
            if row["price_change_5d"] > -0.03:
                price_fail += 1
                continue
            if i < 5 or d.iloc[i - 5:i]["rsi"].min() < row["rsi"]:
                trend_fail += 1
                continue

            passed_all += 1
            eval_rows += 1
            entry_price = row["Close"]
            entry_idx = row_date

            future_available = len(d) - (i + 1)
            this_lookahead = min(lookahead_days, future_available)
            future = d.iloc[i + 1: i + 1 + this_lookahead]

            if future.empty:
                continue

            hit_tp = hit_sl = False
            exit_day = future.index[-1]
            exit_type = "final_close"

            for j, (high, low) in enumerate(zip(future["High"], future["Low"])):
                if (high - entry_price) / entry_price >= profit_thresh:
                    hit_tp = True
                    exit_day = future.index[j]
                    exit_type = "profit_hit"
                    break
                if (low - entry_price) / entry_price <= stop_thresh:
                    hit_sl = True
                    exit_day = future.index[j]
                    exit_type = "stop_hit"
                    break

            exit_price = d.loc[exit_day, "Close"] if exit_day in d.index else future["Close"].iloc[-1]
            final_ret = (exit_price - entry_price) / entry_price
            max_ret = (future["High"].max() - entry_price) / entry_price
            min_ret = (future["Low"].min() - entry_price) / entry_price

            d.at[entry_idx, "target"] = 1 if hit_tp else 0
            d.at[entry_idx, "label_type"] = "high"
            d.at[entry_idx, "entry_date"] = entry_idx
            d.at[entry_idx, "exit_date"] = exit_day
            d.at[entry_idx, "exit_type"] = exit_type
            d.at[entry_idx, "final_return"] = final_ret
            d.at[entry_idx, "mfe"] = max_ret
            d.at[entry_idx, "mae"] = min_ret

        # print("\n[High-VIX Labeler Debug Summary]")
        # print(f"Total rows in df:             {total_rows}")
        # print(f"Rows evaluated:               {eval_rows}")
        # print(f"Passed all filters:           {passed_all}")
        # print(f"Filtered by RSI > 50:         {rsi_fail}")
        # print(f"Filtered by low VIX:          {vix_fail}")
        # print(f"Filtered by low volume:       {vol_fail}")
        # print(f"Filtered by weak dip:         {price_fail}")
        # print(f"Filtered by RSI not lowest:   {trend_fail}")


        return d.dropna(subset=["target"])





    
    def label_rsi_reversal_debug(self, df: pd.DataFrame, lookahead_days: int = 180) -> pd.DataFrame:
        d = df.copy()
        d.index = pd.to_datetime(d.index)  # ✅ Ensure datetime index

        d["target"] = np.nan
        d["entry_date"] = pd.NaT
        d["exit_date"] = pd.NaT
        d["exit_type"] = "dummy"
        d["mfe"] = np.nan
        d["mae"] = np.nan
        d["final_return"] = np.nan
        d["label_type"] = "high"

        label_attempts = 0
        labeled_rows = 0

        for i in range(10, len(d) - 10, 10):  # label every 10th row
            entry_idx = d.index[i]
            d.at[entry_idx, "target"] = 1
            d.at[entry_idx, "entry_date"] = entry_idx
            d.at[entry_idx, "exit_date"] = d.index[i + 10]
            d.at[entry_idx, "exit_type"] = "dummy"
            d.at[entry_idx, "final_return"] = 0.01
            d.at[entry_idx, "mfe"] = 0.015
            d.at[entry_idx, "mae"] = -0.005

            label_attempts += 1
            labeled_rows += 1

        print(f"[🧪 Dummy Labeler] Attempts: {label_attempts}, Labels assigned: {labeled_rows}")
        return d.dropna(subset=["target"])
