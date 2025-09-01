import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# ==== Feature flags & knobs (defaults OFF unless you turn them on elsewhere) ====
ENABLE_STRICT_ENTRY_FILTERS = False   # stronger entry vetoes (below-200 + weak momentum; falling 50d + weak tape)
ENABLE_REENTRY_COOLDOWN     = True   # block re-entries for N bars after early exits
USE_RATIO_P200              = True    # True => price_vs_ema200 is a ratio (1.0 = on 200d). False => pct-from (0.0 = on 200d)
REENTRY_BARS                = 15      # default cooldown length after early exits

# ==== 200d relation helpers (honor USE_RATIO_P200) ====
def _p200_near(p200_value, eps=0.02):
    """
    Returns True if price is within 'eps' of the 200d.
    Ratio mode (USE_RATIO_P200=True): near 1.0
    Pct-from mode (USE_RATIO_P200=False): near 0.0
    """
    anchor = 1.0 if USE_RATIO_P200 else 0.0
    try:
        x = float(p200_value)
    except Exception:
        return False
    return abs(x - anchor) < float(eps)

def _p200_below(p200_value):
    """True if price is below the 200d (mode-aware)."""
    anchor = 1.0 if USE_RATIO_P200 else 0.0
    try:
        return float(p200_value) < anchor
    except Exception:
        return False

def _p200_above(p200_value):
    """True if price is above the 200d (mode-aware)."""
    anchor = 1.0 if USE_RATIO_P200 else 0.0
    try:
        return float(p200_value) > anchor
    except Exception:
        return False


def _gnum(row, key, default=None):
    """Safe numeric getter: returns float or default if missing/NaN/None."""
    v = row.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    # treat NaN as missing
    if v != v:  # NaN check
        return default
    return v

# ==== Lightweight re-entry cooldown registry ====
# Keyed by ticker; decremented once per market day by calling decay_reentry_blocks()
_REENTRY_BLOCKS = {}  # dict[str, int]

def register_reentry_block(ticker: str, bars: int = REENTRY_BARS) -> None:
    """
    Register a cooldown window for 'ticker' to avoid immediate re-entries after an early exit.
    Use when an early fuse triggers in simulate_one_day().
    """
    if not ticker:
        return
    try:
        bars = int(bars)
    except Exception:
        bars = REENTRY_BARS
    prev = _REENTRY_BLOCKS.get(ticker, 0)
    _REENTRY_BLOCKS[ticker] = max(prev, bars)

def decay_reentry_blocks() -> None:
    """
    Decrement all cooldown counters by 1 bar. Call once per market day
    (e.g., at the start of simulate_one_day() or in your outer daily loop).
    """
    if not _REENTRY_BLOCKS:
        return
    to_delete = []
    for tkr, remaining in _REENTRY_BLOCKS.items():
        new_val = int(remaining) - 1
        _REENTRY_BLOCKS[tkr] = new_val
        if new_val <= 0:
            to_delete.append(tkr)
    for tkr in to_delete:
        _REENTRY_BLOCKS.pop(tkr, None)

def is_reentry_blocked(row) -> bool:
    """
    Returns True if 'row' (candidate entry) is currently blocked by cooldown.
    Expects row to have 'ticker' (or 'symbol' / 'TICKER') present.
    """
    if row is None:
        return False
    tkr = (
        row.get("ticker")
        or row.get("symbol")
        or row.get("TICKER")
        or row.get("Symbol")
        or row.get("Ticker")
    )
    if not tkr:
        return False
    return bool(_REENTRY_BLOCKS.get(tkr, 0) > 0)



def is_flat_price_zone(row, regime=None):
    """
    'Flat' = price hugging long trend, MA50 slope near zero, RSI mid-range.
    Mode-aware wrt price_vs_ema200:
      - USE_RATIO_P200=True  -> anchor is 1.0 (ratio; 1.0 = on 200d)
      - USE_RATIO_P200=False -> anchor is 0.0 (pct-from; 0.0 = on 200d)
    """
    price_vs_ema200 = _gnum(row, "price_vs_ema200", None)
    ema50_slope     = _gnum(row, "ema50_slope", None)
    rsi             = _gnum(row, "rsi", None)

    # Fail-safe: if any missing, do NOT auto-block
    if price_vs_ema200 is None or ema50_slope is None or rsi is None:
        return False

    # base thresholds
    thr_price_eps = 0.02  # "near 200d" envelope (±2% in ratio mode; ±0.02 abs in pct-from mode)
    thr_slope     = 0.05
    rsi_low, rsi_high = 45.0, 55.0

    if regime == "low":
        thr_slope = 0.04
    elif regime == "high":
        rsi_low, rsi_high = 43.0, 57.0

    # Mode-aware 200d "hugging" check
    flat_price = _p200_near(price_vs_ema200, eps=thr_price_eps)
    flat_slope = (abs(float(ema50_slope)) < float(thr_slope))
    flat_rsi   = (float(rsi) >= float(rsi_low) and float(rsi) <= float(rsi_high))

    return bool(flat_price and flat_slope and flat_rsi)



def is_choppy_consolidation(row, regime=None):
    """
    'Chop' = mid Bollinger, low participation, flat slope.
    Regime-aware volume bar:
      - Caution: raise the volume requirement (fake breakouts are common).
      - High: keep volume bar low to allow capitulation reversals elsewhere.
    """
    bb_pct       = _gnum(row, "bb_pct", 0.5)
    vol_surge    = _gnum(row, "volume_surge", 1.0)
    ema50_slope  = _gnum(row, "ema50_slope", 0.0)

    # base thresholds
    low_volatility = (0.40 <= bb_pct <= 0.60)
    no_volume_thr  = 0.05
    if regime == "caution":
        no_volume_thr = 0.10   # need more participation to trust breaks in chop
    elif regime == "high":
        no_volume_thr = 0.05   # unchanged; we use capitulation logic elsewhere

    no_volume = (vol_surge is not None and vol_surge < no_volume_thr)
    flat_slope = (abs(ema50_slope) < 0.05)

    # If any key input is missing, fail-safe to False (don’t auto-block)
    if bb_pct is None or vol_surge is None or ema50_slope is None:
        return False

    return bool(low_volatility and no_volume and flat_slope)


def passes_entry_filters(row, regime=None):
    """
    Entry gate with optional strict vetoes and re-entry cooldown.
    Defaults keep behavior unchanged unless flags are enabled:
      - ENABLE_REENTRY_COOLDOWN: blocks if ticker is on cooldown
      - ENABLE_STRICT_ENTRY_FILTERS: activates below-200 + weak-momentum vetoes
    Structural vetoes:
      - is_flat_price_zone(...)
      - is_choppy_consolidation(...)
    """
    import math

    # ---- regime resolution (use row override if present) ----
    regime = (row.get("volatility_regime") or regime or "low")

    # ---- safe getter (float-casting; returns math.nan on failure) ----
    def g(k, default=math.nan):
        v = row.get(k, default)
        try:
            return float(v)
        except Exception:
            return default

    # ---- 0) re-entry cooldown (optional) ----
    if ENABLE_REENTRY_COOLDOWN and is_reentry_blocked(row):
        return False

    # ---- core pulls (use g(); 200d checks are via helpers below) ----
    p200        = g("price_vs_ema200")
    rsi         = g("rsi")
    macd_diff   = g("macd_diff")
    ema50_slope = g("ema50_slope")
    bb_pct      = g("bb_pct")

    # ---- 1) structural tape vetoes ----
    if is_flat_price_zone(row, regime=regime):
        return False
    if is_choppy_consolidation(row, regime=regime):
        return False

    # ---- 2) optional strict entry vetoes (default OFF) ----
    if ENABLE_STRICT_ENTRY_FILTERS:
        below200_mom_weak = (_p200_below(p200) and (rsi < 48.0) and (macd_diff < 0.0))
        falling50_weak    = ((ema50_slope < 0.0) and (rsi < 50.0) and (bb_pct < 0.25))
        if below200_mom_weak or falling50_weak:
            return False

    # ---- 3) light safety veto (always-on; conservative) ----
    if (
        (bb_pct < 0.20) and
        (macd_diff < -0.30) and
        (ema50_slope < 0.0) and
        _p200_near(p200, eps=0.02)  # mode-aware: 1.0 in ratio mode; 0.0 in pct-from mode
    ):
        return False

    # ---- 4) regime-specific rules (kept as provided) ----
    if regime == "low":
        bad_low_bleed = (
            (g("ema50_slope") < 0) and
            (g("macd_diff") < -0.30) and
            _p200_below(g("price_vs_ema200"))
        )
        return not (
            bad_low_bleed or
            (_p200_above(g("price_vs_ema200")) and g("ema50_slope") <= 0 and g("macd_diff") < -0.3 and g("rsi") > 55) or
            (_p200_below(g("price_vs_ema200")) and g("macd_diff") < 0 and g("rsi") > 50) or
            (g("rsi") >= 68 or g("bb_pct") >= 0.95) or
            (g("vix_5d_slope") > 5) or
            is_flat_price_zone(row, regime=regime) or
            is_choppy_consolidation(row, regime=regime)
        )

    elif regime == "caution":
        if not row.get("macro_trend_ok", False):
            return False
        weak_chop = (
            (g("macd_diff") < 0.0) and
            (_p200_above(g("price_vs_ema200")) is False or _p200_near(g("price_vs_ema200"), 0.02)) and
            (g("bb_pct") < 0.20)
        )
        return not (
            weak_chop or
            (_p200_above(g("price_vs_ema200")) and g("ema50_slope") < 0.05 and g("macd_diff") < -0.3 and g("rsi") < 50) or
            (_p200_below(g("price_vs_ema200")) and g("macd_diff") < -0.5 and g("rsi") < 40) or
            (g("rsi") >= 65 or g("bb_pct") >= 0.90) or
            (g("price_vs_ema50") > 0.03 and g("volume_surge") < 0.10) or
            is_flat_price_zone(row, regime=regime) or
            is_choppy_consolidation(row, regime=regime)
        )

    elif regime == "high":
        # mode-aware: delta from the 200d anchor (1.0 if ratio; 0.0 if pct-from)
        p200_delta = g("price_vs_ema200") - (1.0 if USE_RATIO_P200 else 0.0)
        extreme_weak_no_capit = (
            (g("rsi") < 30.0) and
            (p200_delta < -0.12) and   # ≤ 0.88*200d in ratio mode, ≤ -12% in pct-from mode
            (g("macd_diff") < -2.0) and
            (g("volume_surge") < 0.10)
        )
        return not (
            (g("rsi") > 45) or
            (g("macd_diff") > 0.5) or
            (g("price_vs_ema50") > -0.005) or
            (g("volume_surge") < 0.05) or
            extreme_weak_no_capit
        )

    # default allow
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



def label_meta_candidates_ranking(
    trade,
    min_hold_days=5,
    min_gain_thresh=0.20,
    fwd_drawdown_thresh=0.20,
    peak_threshold=0.05,      # within 5% of trailing peak = full proximity
    alpha=1.5,
    beta=1.0,
    gamma=1.5,
    momentum_weight=1.5,      # penalize decay
    peak_bonus=0.15,          # convex boost near peak
    regret_weight=0.1,
    decay_lambda=0.03,
    top_k_pct=0.10,           # fallback band if floor yields 0 positives
    absolute_utility_thresh=1.70,  # lowered slightly from 1.98 for healthier coverage
    near_best_1=0.010,        # Δ from best utility for rel=3
    near_best_2=0.020,        # rel=2
    near_best_3=0.035,        # rel=1
    jitter_eps=1e-4           # tiny deterministic tiebreaker scale
):
    """
    Label exits with graded relevance around the trade's utility peak.

    Improvements (quant-focused):
    - Compute utility/components for *all* candidates (no row drops).
    - Add tiny, deterministic tie-breaking 'jitter' from secondary signals to avoid utility plateaus.
    - Produce a *continuous* relevance 'rel_cont' in [0,1] for smoother training.
    - Apply floors only to discrete labels; continuous stays smooth.
    - Mark absolute best by adjusted utility (is_best_candidate=1).
    """
    import numpy as np
    import pandas as pd
    import hashlib

    entry_price = float(trade.get("entry_price", 0.0))
    entry_date  = pd.to_datetime(trade.get("entry_date")).normalize()
    price_df    = trade.get("price_series")
    ticker      = str(trade.get("ticker"))

    if price_df is None or price_df.empty or entry_price <= 0:
        return
    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index).normalize()

    candidates = trade.get("meta_candidates", [])
    if not candidates:
        return

    # Stable group_id: date-only to avoid '00:00:00' variants
    group_id = f"{entry_date.strftime('%Y-%m-%d')}_{ticker}"
    for cand in candidates:
        cand["group_id"] = group_id
        cand.setdefault("rel", 0)
        cand.setdefault("target", 0)
        cand.setdefault("is_best_candidate", 0)

    # Trailing max (no look-ahead)
    max_close_so_far = price_df["Close"].cummax()

    valid = []
    for cand in candidates:
        exit_date  = pd.to_datetime(cand.get("exit_date")).normalize()
        exit_price = cand.get("exit_price")
        if exit_price is None or exit_date not in price_df.index:
            continue
        exit_price = float(exit_price)

        days_held     = int((exit_date - entry_date).days)
        captured_gain = (exit_price - entry_price) / entry_price

        # Available gain up to exit
        available_gain  = (max_close_so_far.loc[:exit_date].max() - entry_price) / entry_price
        gain_lock_ratio = captured_gain / (available_gain + 1e-12)

        # Forward 5d window (exclude same day)
        fwd = price_df.loc[exit_date:].iloc[1:6]
        if not fwd.empty:
            low_after  = float(fwd["Low"].min())
            high_after = float(fwd["High"].max())
            future_drawdown = max(0.0, (exit_price - low_after) / exit_price)
            future_regret   = max(0.0, (high_after - exit_price) / exit_price)
        else:
            future_drawdown = 0.0
            future_regret   = 0.0

        # Momentum: penalize decay only
        momentum_signal = 0.0
        if "rsi" in price_df.columns:
            rsi_series = price_df.loc[:exit_date].tail(10)["rsi"]
            if len(rsi_series) >= 2:
                momentum_signal += np.tanh(float(rsi_series.iloc[-1]) - float(rsi_series.iloc[0]))
        if "macd_hist" in price_df.columns:
            macd_series = price_df.loc[:exit_date].tail(10)["macd_hist"]
            if len(macd_series) >= 2:
                momentum_signal += np.tanh(float(macd_series.iloc[-1]) - float(macd_series.iloc[0]))
        momentum_penalty = max(0.0, -momentum_signal)

        # Peak proximity reward
        historical_high = float(price_df.loc[entry_date:exit_date]["High"].max())
        pct_from_peak   = (abs(exit_price - historical_high) / historical_high) if historical_high > 0 else 1.0
        peak_proximity  = max(0.0, 1.0 - (pct_from_peak / peak_threshold))
        near_peak_boost = peak_bonus * (peak_proximity ** 1.5)

        # Time + regret
        time_penalty   = 1 - np.exp(-decay_lambda * max(days_held, 0))
        regret_penalty = regret_weight * future_regret

        # Base utility (continuous)
        utility = (
            alpha * gain_lock_ratio
            - beta * future_drawdown
            - gamma * time_penalty
            - momentum_weight * momentum_penalty
            + near_peak_boost
            - regret_penalty
        )

        # ---- Deterministic tiny jitter to break ties ----
        # Use stable hash of (group_id, exit_date) so runs are reproducible.
        h = hashlib.blake2b(f"{group_id}|{exit_date:%Y-%m-%d}".encode(), digest_size=8).hexdigest()
        uhash = int(h, 16) / 2**64  # in [0,1)
        # Secondary signal: momentum & forward outcomes — very small influence
        sec = 0.50 * momentum_signal + 0.30 * captured_gain - 0.20 * future_drawdown
        jitter = jitter_eps * (sec + (uhash - 0.5))  # centered ~0

        utility_adj = utility + jitter

        cand.update({
            "days_held": days_held,
            "captured_gain": captured_gain,
            "utility": utility,          # raw
            "utility_adj": utility_adj,  # tie-broken
            "jitter": jitter,
            "gain_lock_ratio": gain_lock_ratio,
            "future_drawdown": future_drawdown,
            "future_regret": future_regret,
            "momentum_signal": momentum_signal,
            "momentum_penalty": momentum_penalty,
            "pct_from_peak": pct_from_peak,
            "peak_proximity": peak_proximity,
            "time_penalty": time_penalty,
            "regret_penalty": regret_penalty,
            "exit_date": exit_date,
            "exit_price": exit_price,
        })
        valid.append(cand)

    if not valid:
        return

    # Rank by adjusted utility
    sorted_cands = sorted(valid, key=lambda x: x["utility_adj"], reverse=True)
    best_util    = sorted_cands[0]["utility_adj"]
    worst_util   = sorted_cands[-1]["utility_adj"]

    # Continuous relevance in [0,1] (groupwise min-max on adjusted utility)
    span = max(best_util - worst_util, 1e-12)
    for cand in sorted_cands:
        cand["rel_cont"] = max(0.0, (cand["utility_adj"] - worst_util) / span)

    # ---- Discrete graded labels with floors (for ranking) ----
    positives = 0
    for i, cand in enumerate(sorted_cands):
        delta = best_util - cand["utility_adj"]

        meets_floor = (
            (cand["days_held"] >= min_hold_days)
            and (cand["captured_gain"] >= min_gain_thresh)
            and (cand["future_drawdown"] <= fwd_drawdown_thresh)
            and (cand["utility_adj"] >= absolute_utility_thresh)
        )

        if meets_floor:
            if   delta <= near_best_1: cand["rel"] = 3
            elif delta <= near_best_2: cand["rel"] = 2
            elif delta <= near_best_3: cand["rel"] = 1
            else:                      cand["rel"] = 0
        else:
            cand["rel"] = 0

        cand["target"] = int(cand["rel"] > 0)
        cand["is_best_candidate"] = int(i == 0)  # absolute best by adjusted utility
        cand["delta_to_best"] = delta
        positives += cand["target"]

    # Fallback if no positives by the absolute floor
    if positives == 0:
        band_n = max(3, int(len(sorted_cands) * top_k_pct))
        for i, cand in enumerate(sorted_cands[:band_n]):
            delta = cand["delta_to_best"]
            if   delta <= near_best_1: cand["rel"] = 3
            elif delta <= near_best_2: cand["rel"] = 2
            elif delta <= near_best_3: cand["rel"] = 1
            else:                      cand["rel"] = 1
            cand["target"] = 1
        sorted_cands[0]["is_best_candidate"] = 1

def compute_exit_features(row, trade, history_window):
    close = history_window["Close"]
    vix = history_window.get("vix_close", pd.Series(index=history_window.index, data=np.nan))
    rsi = history_window.get("rsi", pd.Series(index=history_window.index, data=np.nan))
    macd = history_window.get("macd_diff", pd.Series(index=history_window.index, data=np.nan))

    price = row["Close"]
    entry_price = trade["entry_price"]
    max_close = trade["entry_features"].get("max_close", close.max())
    min_close = close.min()
    core_proba = trade.get("core_proba", 0)
    day_count = trade.get("day_count", 0)

    # Return-based
    pct_of_peak = (price - entry_price) / (max_close - entry_price) if max_close > entry_price else 0
    vol_5d = close.pct_change().rolling(5).std().iloc[-1] if len(close) >= 5 else 0
    vol_adj_ret = ((price - entry_price) / entry_price) / vol_5d if vol_5d else 0

    # Trend decay
    ema5_slope = row.get("ema5", price) - history_window.get("ema5", pd.Series([price])).iloc[-5] if len(history_window) >= 5 else 0
    price_trend_slope_5 = (close.iloc[-1] - close.iloc[-5]) / 5 if len(close) >= 5 else 0
    price_vs_max_last_5 = close.iloc[-1] / close.rolling(5).max().iloc[-1] - 1 if len(close) >= 5 else 0

    # RSI / MACD
    rsi_now = row.get("rsi", 50)
    rsi_slope_3d = (rsi.iloc[-1] - rsi.iloc[-4]) / 3 if len(rsi) >= 4 else 0
    rsi_above_70 = int(rsi_now >= 70)
    macd_now = row.get("macd_diff", 0)
    macd_rolling_3 = macd.iloc[-3:].mean() if len(macd) >= 3 else 0
    macd_rollover = int(macd_now < 0 and macd_rolling_3 > 0)

    # Trailing stop
    trailing_stop_pct = 0.75
    price_vs_trailing_stop = price / max_close - trailing_stop_pct if max_close else 0
    price_near_trailing_peak = int(abs(price - max_close) / max_close < 0.03) if max_close else 0

    # Exit regime
    vix_now = row.get("vix_close", 20)
    exit_regime_low = int(vix_now <= 20)
    exit_regime_caution = int(20 < vix_now <= 30)
    exit_regime_high = int(vix_now > 30)

    # Entry regime
    entry_regime = trade.get("entry_regime", "")
    entry_regime_low = int(entry_regime == "low")
    entry_regime_caution = int(entry_regime == "caution")
    entry_regime_high = int(entry_regime == "high")

    return {
        # --- Returns & path ---
        "pct_of_peak_captured": pct_of_peak,
        "vol_adj_return": vol_adj_ret,

        # --- Trend decay ---
        "price_trend_slope_5": price_trend_slope_5,
        "price_vs_max_last_5": price_vs_max_last_5,
        "price_vs_ema5": price / row.get("ema5", price) - 1,
        "price_vs_ema20": price / row.get("ema20", price) - 1,
        "price_vs_ema50": price / row.get("ema50", price) - 1,
        "ema5_slope": ema5_slope,
        "macd_diff": macd_now,
        "macd_rolling_3": macd_rolling_3,
        "macd_rollover": macd_rollover,
        "rsi": rsi_now,
        "rsi_slope_3d": rsi_slope_3d,
        "rsi_above_70": rsi_above_70,
        "stoch_d": row.get("stoch_d", np.nan),
        "stoch_k": row.get("stoch_k", np.nan),

        # --- Lifecycle / duration ---
        "entry_age_norm": day_count / 20 if day_count else 0,
        "days_since_max_price": len(close) - close.argmax(),
        "exit_efficiency_days": abs((row.name - pd.to_datetime(trade.get("peak_date"))) // pd.Timedelta("1D"))
            if trade.get("peak_date") else 0,

        # --- Drawdown & risk ---
        "rolling_max_drawdown": ((close.cummax() - close) / close.cummax()).max(),
        "entry_to_peak_drawdown": (max_close - price) / max_close if max_close else 0,
        "price_vs_trailing_stop": price_vs_trailing_stop,
        "price_near_trailing_peak": price_near_trailing_peak,
        "volatility_5d": vol_5d,

        # --- Regimes ---
        "entry_regime_low": entry_regime_low,
        "entry_regime_caution": entry_regime_caution,
        "entry_regime_high": entry_regime_high,
        "exit_regime_low": exit_regime_low,
        "exit_regime_caution": exit_regime_caution,
        "exit_regime_high": exit_regime_high,
        "vix_close": vix_now,
        "vix_spike": vix.iloc[-1] / vix.iloc[-3] - 1 if len(vix) >= 3 else 0,

        # --- Core signal ---
        "core_proba": core_proba,
        "core_proba_high": int(core_proba > 0.75)
    }


def finalize_trade(trade, meta_end_month, core_trade_state):
    """
    Finalize an open trade at the end of the meta simulation.
    Adds a timeout meta candidate and delegates ranking/labeling to the zone-aware ranker.
    """

    ticker = trade["ticker"]
    entry_date = pd.to_datetime(trade["entry_date"]).normalize()
    end_date = pd.to_datetime(meta_end_month).normalize()

    trade["exit_date"] = end_date
    trade["exit_price"] = trade.get("last_seen_price", trade["entry_price"])
    trade["exit_reason"] = "forced_timeout"
    trade["final_return"] = (trade["exit_price"] - trade["entry_price"]) / trade["entry_price"]

    price_df = core_trade_state.get(ticker)
    if price_df is not None and not price_df.empty:
        price_df.index = pd.to_datetime(price_df.index).normalize()
    else:
        price_df = None

    # ─── Capture Stats ───
    if price_df is not None and "High" in price_df.columns:
        try:
            exit_window = price_df.loc[entry_date:end_date]
            highs = exit_window["High"].dropna()
            if not highs.empty:
                max_price = highs.max()
                peak_date = highs.idxmax()
                trade["max_possible_return"] = (max_price - trade["entry_price"]) / trade["entry_price"]
                trade["capture_ratio"] = (
                    trade["final_return"] / trade["max_possible_return"]
                    if trade["final_return"] > 0 and trade["max_possible_return"] > 0
                    else 0
                )
                trade["peak_date"] = peak_date
                trade["exit_efficiency_days"] = abs((end_date - peak_date).days)
            else:
                raise ValueError("No highs in window")
        except Exception as e:
            print(f"[⚠️] Capture stats failed for {ticker}: {e}")
            trade.update({
                "max_possible_return": None,
                "capture_ratio": None,
                "peak_date": None,
                "exit_efficiency_days": None
            })
    else:
        trade.update({
            "max_possible_return": None,
            "capture_ratio": None,
            "peak_date": None,
            "exit_efficiency_days": None
        })

    # ─── Optional binary tags for diagnostics ───
    trade["meta_label"] = 1 if trade["final_return"] > 0.20 else 0
    capture = trade.get("capture_ratio")
    trade["capture_label"] = 1 if capture is not None and capture >= 0.6 else 0

    # ─── Exit Features ───
    if price_df is not None:
        try:
            last_row = price_df.loc[:end_date].iloc[-1]
            history_window = price_df.loc[:end_date].tail(10)
            exit_features = compute_exit_features(last_row, trade, history_window)
        except Exception as e:
            print(f"[⚠️] Feature extraction failed for {ticker}: {e}")
            exit_features = {}
    else:
        exit_features = {}

    trade["exit_features"] = exit_features

    if "meta_candidates" not in trade:
        trade["meta_candidates"] = []

    # ─── Final Timeout Candidate ───
    candidate = {
        "exit_date": end_date,
        "exit_price": trade["exit_price"],
        "features": exit_features,
        "group_id": f"{trade['entry_date']}_{ticker}"
        # ⛔ No target/utility here — will be assigned by labeler
    }
    trade["meta_candidates"].append(candidate)

    # ─── Apply Smart Exit Zone Labeling ───
    label_meta_candidates_ranking(trade)

    # ─── Diagnostics: return delta vs final exit ───
    for cand in trade["meta_candidates"]:
        if "exit_price" in cand:
            cand_return = (cand["exit_price"] - trade["entry_price"]) / trade["entry_price"]
            cand["return_delta"] = cand_return - trade["final_return"]


def compute_feature_auc_scores(df, label_col="label"):
    """
    Compute AUC score for each feature vs binary label.
    Returns a sorted Series (high AUC = better separation).
    """
    aucs = {}
    for col in df.columns:
        if col in ["label", "ticker", "entry_date", "exit_date", "entry_price", "exit_price", "final_return"]:
            continue
        try:
            auc = roc_auc_score(df[label_col], df[col])
            # Flip if < 0.5 (lower values for label=1)
            auc = max(auc, 1 - auc)
            aucs[col] = auc
        except Exception:
            continue  # skip constant or NaN columns
    return pd.Series(aucs).sort_values(ascending=False)
