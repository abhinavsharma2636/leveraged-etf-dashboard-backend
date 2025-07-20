def is_flat_price_zone(row):
    flat_price = abs(row.get("price_vs_ema200", 0)) < 0.02
    flat_slope = abs(row.get("ema50_slope", 0)) < 0.05
    flat_rsi = 45 <= row.get("rsi", 50) <= 55
    return flat_price and flat_slope and flat_rsi


def is_choppy_consolidation(row):
    low_volatility = 0.4 <= row.get("bb_pct", 0.5) <= 0.6
    no_volume = row.get("volume_surge", 1.0) < 0.05
    flat_slope = abs(row.get("ema50_slope", 0)) < 0.05
    return low_volatility and no_volume and flat_slope



def passes_entry_filters(row, regime):
    """
    Applies regime-specific filters to determine whether an entry should be allowed.
    Returns:
        bool: True if the entry passes all filters, False otherwise.
    """

    # ─── Low Volatility Regime: Bull trend or strong setup only ───
    if regime == "low":
        return not (
            (row["price_vs_ema200"] > 0 and row["ema50_slope"] <= 0 and row["macd_diff"] < -0.3 and row["rsi"] > 55) or
            (row["price_vs_ema200"] < 0 and row["macd_diff"] < 0 and row["rsi"] > 50) or
            (row["rsi"] >= 68 or row["bb_pct"] >= 0.95) or
            (row["vix_5d_slope"] > 5) or
            is_flat_price_zone(row) or
            is_choppy_consolidation(row)
        )

    # ─── Caution Regime: Only trade with confirmed macro alignment ───
    elif regime == "caution":
        if not row.get("macro_trend_ok", False):
            return False

        return not (
            (row["price_vs_ema200"] > 0 and row["ema50_slope"] < 0.05 and row["macd_diff"] < -0.3 and row["rsi"] < 50) or
            (row["price_vs_ema200"] < 0 and row["macd_diff"] < -0.5 and row["rsi"] < 40) or
            (row["rsi"] >= 65 or row["bb_pct"] >= 0.9) or
            (row["price_vs_ema50"] > 0.03 and row["volume_surge"] < 0.05) or
            is_flat_price_zone(row) or
            is_choppy_consolidation(row)
        )

    # ─── High Volatility Regime: Flush only ───
    elif regime == "high":
        return not (
            row["rsi"] > 45 or
            row["macd_diff"] > 0.5 or
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

def is_protected_early_entry(row, entry_features, day_count, entry_price):
    rsi_entry = entry_features.get("rsi", 50)
    macd_entry = entry_features.get("macd_diff", 0)
    price = row["Close"]
    ema50 = row.get("ema50", price)
    gain = (price - entry_price) / entry_price

    return (
        rsi_entry < 40 and
        macd_entry < 0 and
        gain > 0.03 and
        price > ema50 and
        day_count < 60
    )


def is_reversal_entry(entry_features):
    return entry_features.get("rsi", 50) < 40 and entry_features.get("macd_diff", 0) < 0


def check_exit_today(row, entry_features, day_count, entry_price, entry_regime):
    price = row["Close"]
    ema50 = row.get("ema50", price)
    ema200 = row.get("ema200", price)
    ema50_slope = row.get("ema50_slope", 0)
    macd = row.get("macd_diff", 0)
    rsi = row.get("rsi", 50)
    vix = row.get("vix", 20)

    gain = (price - entry_price) / entry_price
    max_close = entry_features.get("max_close", entry_price)

    # 🛑 1. EMERGENCY STOP
    if day_count < 20 and gain < -0.20:
        return "exit_emergency_stop"

    # 🔒 2. MIN HOLD PERIOD
    if day_count < 20:
        return None

    # 🛡️ 3. UNDER EMA200 GRACE WINDOW
    if entry_price < ema200 and price > entry_price and day_count < 60:
        return None

    # 🦍 4. SUPERWINNER HOLD
    if gain > 2.0 and price > ema200 and ema50_slope > 0:
        return None

    # ⛔ 5. REVERSAL TRUST
    if is_protected_early_entry(row, entry_features, day_count, entry_price):
        return None

    # 💣 6. FAILING REVERSAL EARLY EXIT (new)
    if is_reversal_entry(entry_features) and day_count < 60:
        drawdown = (price - entry_price) / entry_price
        if drawdown < -0.25:
            return "exit_failed_reversal"

    # 🧨 7. MACRO PANIC EXIT
    if vix > 25 and price < ema50 and rsi < 40:
        return "exit_macro_breakdown"

    # 📉 8. STRUCTURAL BREAKDOWN
    if price < ema200 and ema50 < ema200 and macd < 0 and rsi < 50 and ema50_slope < 0:
        return "exit_structural_breakdown"

    # 🏔️ 9. TRAILING STOP
    if price < max_close * 0.75:
        return "exit_trailing_stop"

    return None  # ✅ Default: HOLD
