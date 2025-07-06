def passes_entry_filters(row, regime):
    if regime == "low":
        return not (
            (row["price_vs_ema200"] > 0 and row["ema50_slope"] < 0.1 and (row["macd_diff"] < 0 or row["rsi"] < 50)) or
            (row["price_vs_ema200"] < 0 and row["macd_diff"] < 0 and row["rsi"] < 50) or
            (row["rsi"] >= 65 or row["bb_pct"] >= 0.95) or
            (row["vix_5d_slope"] > 0) or
            (row["fear_greed"] < 50) or
            (row["ema50_slope"] < 0.05)
        )

    elif regime == "caution":
        if not row.get("macro_trend_ok", True):
            return False
        return not (
            (row["price_vs_ema200"] > 0 and row["ema50_slope"] < 0.05 and row["macd_diff"] < 0 and row["rsi"] < 50) or
            (row["price_vs_ema200"] < 0 and row["macd_diff"] < -0.5 and row["rsi"] < 40) or
            (row["rsi"] >= 65 or row["bb_pct"] >= 0.95) or
            (row["price_vs_ema50"] > 0.05 or row["rsi"] > 60) or
            (row["price_vs_ema200"] > 0 and row["macd_diff"] < -0.3) or
            (row["bb_pct"] > 0.85)
        )

    elif regime == "high":
        return not (
            row["rsi"] > 45 or
            row["macd_diff"] > 0.3 or
            row["price_vs_ema50"] > -0.005 or
            row["volume_surge"] < 0.05
        )

    return True


def check_exit_condition(df, entry_idx, entry_price, entry_regime, i):
    exit_type = "max_hold"
    max_high = entry_price
    min_low = entry_price
    rsi_peaked = False
    min_hold_days = 10

    for j in range(i + 1, len(df)):
        row = df.iloc[j]
        max_high = max(max_high, row["High"])
        min_low = min(min_low, row["Low"])
        hold_days = j - i
        mfe = (max_high - entry_price) / entry_price
        mae = (min_low - entry_price) / entry_price
        current_regime = "high" if row["vix_close"] > 30 else "caution" if row["vix_close"] > 20 else "low"

        stop_loss_thresh = {"low": -0.05, "caution": -0.05, "high": -0.20}[entry_regime]
        if mae < stop_loss_thresh:
            return j, "stop_loss"

        if entry_regime == "low":
            if hold_days < min_hold_days:
                continue
            if current_regime == "caution":
                if (hold_days >= 7 and mfe < 0.05 and (row["Close"] - entry_price) / entry_price < -0.03 and row["macd_diff"] < -0.5):
                    return j, "caution_exit_loss_macd_from_low"
                if row["ema50_slope"] < 0.03 and row["macd_diff"] < -0.5 and row["rsi"] < 45 and mfe > 0.04:
                    return j, "caution_exit_macd_rsi_v2_from_low"
                continue
            if row["price_vs_ema50"] > -0.01 and row["ema50"] > row["ema200"] and row["macd_diff"] > -0.3 and row["rsi"] > 50:
                continue
            if row["price_vs_ema200"] < 0 and row["price_vs_ema50"] < -0.02 and row["macd_diff"] < -0.6 and row["rsi"] < 45:
                return j, "low_exit_confirmed_breakdown"

        elif entry_regime == "caution":
            if hold_days < min_hold_days:
                continue
            if current_regime == "low":
                ema50_slope_3day = df.iloc[j]["ema50"] - df.iloc[max(j - 3, 0)]["ema50"]
                if row["price_vs_ema200"] > 0 and ema50_slope_3day < 0.05 and row["macd_diff"] < -0.5 and row["rsi"] < 45 and mfe > 0.03:
                    return j, "low_exit_macd_rsi_from_caution"
                continue
            if hold_days >= 7 and mfe < 0.05 and (row["Close"] - entry_price) / entry_price < -0.03 and row["macd_diff"] < -0.5:
                return j, "caution_exit_loss_macd"
            if row["ema50_slope"] < 0.03 and row["macd_diff"] < -0.5 and row["rsi"] < 45 and mfe > 0.04:
                return j, "caution_exit_macd_rsi_v2"

        elif entry_regime == "high":
            if not rsi_peaked and row["rsi"] >= 70:
                rsi_peaked = True
            uptrend = row["ema20"] > row["ema50"] > row["ema200"]
            if rsi_peaked and row["rsi"] < 40 and row["Close"] < row["ema20"] and row["macd_diff"] < 0 and not uptrend:
                return j, "high_exit_rsi_macd"

    return len(df) - 1, exit_type
