import numpy as np
import pandas as pd
from .simulate_helpers import (
    check_exit_today, compute_exit_features, label_meta_candidates_ranking,
    passes_entry_filters, check_exit_condition,
    register_reentry_block, _p200_below, _p200_near, _p200_above, USE_RATIO_P200
)
from .simulate_helpers import decay_reentry_blocks


class TradeSimulator:
    def __init__(self, feat_cols, models):
        self.feat_cols = feat_cols
        self.models = models


    @staticmethod
    def simulate_one_day(
        day,
        daily_data,
        feat_cols,                 # ENTRY feature list (for your core entry classifier)
        models,
        open_positions,
        results,
        core_trade_state,
        meta_model=None,           # optional entry meta filter
        exit_meta_model=None,      # exit ranker (LightGBM)
        EXIT_FEATURE_COLS=None,    # trained exit feature list (exact order)
    ):
        import numpy as np
        import pandas as pd

        # ---- Meta-exit policy (kept for meta score percentile only) ----
        WIN              = 60
        EXIT_PCT_TAU     = 0.85
        EXIT_TOPK_PCT    = 0.10
        EXIT_TOPK_MIN    = 3
        MIN_HOLD_DAYS    = 5
        MIN_GAIN_PCT     = 0.10
        NEAR_PEAK_WITHIN = 0.02

        # ---- Anti-churn guards (NEW) ----
        GRACE_DAYS_BASE    = 5     # no normal exits for first N bars
        EXCEPTIONAL_PROBA  = 0.80  # if entry proba >= this => longer grace
        GRACE_DAYS_EXCEPT  = 8

        TWO_BAR_CONFIRM      = True  # require 2 consecutive trail breaks
        NEED_2_CLOSES        = True
        TRAIL_PENETRATION_ATR = 0.50 # or 0.5*ATR penetration in one bar

        # ==== Exit feature flags (defaults OFF) ====
        ENABLE_NOPROG_FUSE             = True
        ENABLE_EARLY_GIVEUP            = False
        ENABLE_LATE_PERSIST_BELOW200D  = False

        # Evaluate exits using the state *before* daily transitions
        USE_PRETRANSITION_STATE_FOR_EXIT = True

        # ==== Mode & anchors ====
        USE_RATIO_P200_LOCAL = USE_RATIO_P200  # honor helpers' 200d semantics (True: ratio 1.0 anchor; False: pct-from 0.0 anchor)

        # ==== Early fuses (no-progress / early give-up) ====
        # EARLY_MIN_DAYS  = 12      # earliest day to evaluate no-progress
        # NOPROG_MFE_MAX  = 0.08    # exit if peak_pnl < 8% by EARLY_MIN_DAYS in weak context
        # GIVEUP_MIN_DAYS = 18      # earliest day to evaluate early give-up
        # GIVEUP_MFE_MAX  = 0.05    # exit if peak_pnl < 5% by GIVEUP_MIN_DAYS in weak context

        # ==== Early fuses (no-progress / early give-up) ====
        EARLY_MIN_DAYS  = 16      # E1 (tuned)
        NOPROG_MFE_MAX  = 0.05    # E1 (tuned)

        # Conservative E2 (later, stricter, low-churn)
        GIVEUP_MIN_DAYS   = 22      # fire later than E1
        GIVEUP_MFE_MAX    = 0.04    # only if peak < +4%
        E2_META_MIN       = 0.4     # skip if meta is strong (meta_pct < 0.35)
        E2_NEAR_PEAK_WITHIN = 0.01  # skip if within 1% of 5d max
        E2_PNL_NOW_MAX    = 0.02    # skip if PnL_now > +2%
        E2_REENTRY_BARS   = 20      # longer cooldown to cut churn

        # ==== Late persist-below-200d (L1_v2 strict) ====
        LATE_PERSIST_DAYS = 2  # keep

        # ==== Late persist-below-200d (L1_v3 stricter) ====
        L1_MIN_BELOW200_STREAK  = 4     # require 4+ consecutive bars below 200d
        L1_REQUIRE_TRAIL_BROKE  = True  # must break trail or have 2 closes
        L1_REQUIRE_CROSS_DOMAIN = True  # structure + momentum weakness
        L1_NEAR_PEAK_WITHIN     = 0.04  # skip if within 4% of 5d max
        L1_META_BLOCK           = 0.25  # block if meta strong
        L1_REENTRY_BARS         = 20    # longer cooldown
        L1_PEAK_CAP_FOR_EXIT    = 0.10  # only if peak_pnl < +10%
        L1_MOM_RSI_MAX          = 48.0  # RSI < 48 or falling MA50
        L1_REQUIRE_WEAK_TH_DAYS = 2     # TH below late threshold for ≥2 days

        # ==== Raised-floor guard (only if the run never improved) ====
        RAISED_FLOOR_MAX_LOSS = -0.15  # cap loss if peak never > +15%
        RAISED_FLOOR_MIN_PEAK = 0.10   # only apply raised floor if peak_pnl < +15%

        EARLY_K_BONUS = +0.8  # loosen trail for early bars
        EARLY_K_DAYS  = 5

        # ---- Defaults for positional trading ----
        k_loose_min, k_loose_max = 3.2, 4.6
        k_tight_floor            = 2.4
        k_default                = 3.8

        ENTRY_FEATURE_COLS = list(feat_cols) if feat_cols is not None else []

        # ensure exit feature list
        if exit_meta_model is not None:
            try:
                EXIT_FEATURE_COLS = list(exit_meta_model.booster_.feature_name())
            except Exception:
                EXIT_FEATURE_COLS = list(EXIT_FEATURE_COLS) if EXIT_FEATURE_COLS is not None else []
        else:
            EXIT_FEATURE_COLS = list(EXIT_FEATURE_COLS) if EXIT_FEATURE_COLS is not None else []

        # === 1) Manage open positions (positional exit controller v2 – tightened gates) ===
        for ticker, row in daily_data.items():
            if ticker not in open_positions:
                continue

            trade = open_positions[ticker]
            trade["day_count"] = trade.get("day_count", 0) + 1

            # ---- PATCH: normalize OHLC/indicators to adjusted close ----
            def _normalize_to_adj(row):
                adjc = float(row.get("Adj Close", row.get("AdjClose", row["Close"])))
                if adjc <= 0:
                    return row
                factor = adjc / float(row["Close"])
                row["Close"] = adjc
                for col in ("Open", "High", "Low", "ema_20", "ema_50", "ema_200", "LL10_price"):
                    if col in row and row[col] is not None:
                        row[col] = float(row[col]) * factor
                return row

            row = _normalize_to_adj(dict(row))  # make a local, adjusted copy

            # ---------------- Utilities ----------------
            import math
            import numpy as np
            import pandas as pd

            def _sigmoid(x):
                try:
                    return 1.0 / (1.0 + math.exp(-x))
                except OverflowError:
                    return 0.0 if x < 0 else 1.0

            # ------------- Config -------------
            # States: 0=Early, 1=Trending, 2=Late
            N_EARLY, TH_TREND, TH_LATE = 7, 0.35, 0.50
            META_VETO, META_LOOSEN, META_TIGHTEN = 0.15, 0.25, 0.75

            # Trail policy (ATR chandelier)
            K_BASE_LOW, K_BASE_MID, K_BASE_HIGH = 3.6, 4.0, 4.4
            K_BONUS_EARLY, K_BONUS_TREND = 0.60, 0.40
            K_TIGHTEN_LATE, K_TIGHT_FLOOR = 0.60, 2.6
            K_META_NUDGE, K_MAX_STEP, K_COOLDOWN_DAYS = 0.30, 0.30, 3

            # Confirmation ladder (base)
            PEN_TREND_ATR, PEN_LATE_ATR = 0.35, 0.75

            # ---- Giveback (state-aware; peak-anchored) ----
            MIN_HOLD_FOR_GB, MIN_GAIN_FOR_GB = 10, 0.10
            GB_START_T1, GB_CAP_T1_TREND, GB_CAP_T1_LATE = 0.20, 0.25, 0.20
            GB_START_T2, GB_CAP_T2_TREND, GB_CAP_T2_LATE = 0.40, 0.20, 0.15
            GB_META_TIGHTEN = 0.03

            # Dynamic giveback tuning (UPDATED)
            GB_STRONG_TREND_BONUS = 0.05   # (unchanged) allow +5% more giveback in strong trends
            GB_WEAK_TREND_TIGHTEN = 0.08   # (was 0.05) tighten by 8% in weak/rolling-over trends

            # NEW: hard cap in weak context (peak-anchored drawdown)
            GB_HARD_CAP_WEAK = 0.28        # exit if drawdown from peak >= 28% in weak context


            # Catastrophic floor guard
            SPLIT_ARTIFACT_RET = 0.40

            # --- Late-state stricter knobs (NEW) ---
            # PATCH START
            LATE_PEN_ATR_STRONG        = 1.25  # deeper one-shot needed to bypass persistence
            LATE_CONFIRM_PERSIST_DAYS  = 3     # else require 2-day persistence of confirm in Late
            META_VETO_LATE             = 0.20  # stronger meta veto specifically for Late
            # PATCH END

            # --- Context guards & hi-vol knobs (NEW) ---
            # Catastrophic floor should respect trend context
            TH_CATA, META_CATA = 0.40, 0.25

            # Late depth should scale up in high volatility
            PEN_LATE_ATR_HIVOL = 0.90



            P200_ANCHOR = (1.0 if USE_RATIO_P200_LOCAL else 0.0)

            # ------------- Book-keeping & features -------------
            price_df = core_trade_state.get(ticker, None)
            # ---- PATCH: normalize index tz ----
            if price_df is not None:
                price_df = price_df.copy()
                price_df.index = pd.to_datetime(price_df.index).tz_localize(None)

            day = pd.to_datetime(day).tz_localize(None)

            history_window = price_df.loc[:day].tail(10) if price_df is not None else None
            exit_features  = compute_exit_features(row, trade, history_window)

            c_list = trade.setdefault("meta_candidates", [])
            c_list.append({"exit_date": day, "exit_price": row["Close"], "features": exit_features})

            trade["peak_close"] = max(trade.get("peak_close", trade["entry_price"]), row["Close"])
            pnl_now   = (row["Close"] - trade["entry_price"]) / trade["entry_price"]
            trade["peak_pnl"] = max(trade.get("peak_pnl", 0.0), pnl_now)

            # Meta percentile (read-only)
            meta_pct = None
            if exit_meta_model is not None and len(c_list) >= 2:
                try:
                    X_full = (
                        pd.DataFrame([c["features"] for c in c_list])
                        .reindex(columns=EXIT_FEATURE_COLS, fill_value=0.0)
                        .astype(float)
                    )
                    scores = exit_meta_model.predict(X_full).astype(float)
                    scores = scores[-min(len(scores), 60):]
                    meta_s = float(scores[-1])
                    meta_pct = float((scores <= meta_s).mean())
                except Exception as _e:
                    print(f"[⚠️] meta pct calc failed for {ticker}: {_e}")

            # Shorthand pulls
            f   = exit_features
            atr = float(f.get("atr", row.get("atr", 0.0)) or 0.0)
            if not atr or atr <= 0:
                atr = 0.015 * float(row["Close"])

            rsi           = float(f.get("rsi", row.get("rsi", 50.0)))
            rsi_med3      = float(f.get("rsi_med3", rsi))
            rsi_slope_3d  = float(f.get("rsi_slope_3d", 0.0))
            macd_rollover = int(f.get("macd_rollover", 0))
            macd_roll_p2  = int(f.get("macd_rollover_persist2", macd_rollover))
            price_vs_max5 = float(f.get("price_vs_max_last_5", 0.0))
            ema20         = float(row.get("ema_20", row["Close"]))
            ema50         = float(row.get("ema_50", row["Close"]))
            ema50_slope   = float(f.get("ema50_slope", row.get("ema50_slope", 0.0)))
            price_vs_ema200 = float(f.get("price_vs_ema200", row.get("price_vs_ema200", 1.0)))
            donch_ll_10     = int(f.get("donch_ll_10", 0))
            ll10_price      = float(f.get("LL10_price", row["Close"]))

            # --- Track persistence below the 200d (mode-aware) ---
            anchor_p200 = (1.0 if USE_RATIO_P200_LOCAL else 0.0)
            p200_val = row.get("price_vs_ema200")
            try:
                p200_f = float(p200_val)
            except Exception:
                p200_f = None

            below200 = (p200_f is not None) and (p200_f <= anchor_p200)
            trade["below200_streak"] = (trade.get("below200_streak", 0) + 1) if below200 else 0

            # ------------- Trade state / init -------------
            state = trade.get("state", 0)
            TH_hist = trade.get("TH_hist", [])
            late_votes = trade.get("late_votes", 0)
            closes_below_trail_count = trade.get("closes_below_trail_count", 0)
            weak_TH_streak = trade.get("weak_TH_streak", 0)  # NEW: persistence counter

            # Keep the pre-transition state for exit gating if enabled
            state_prev = trade.get("state_pre", state)
            trade["state_pre"] = state  # store today's pre-transition state
            state_for_exit = state_prev if USE_PRETRANSITION_STATE_FOR_EXIT else state

            # ------------- Vol bucket → k_base -------------
            sigma_pctl = trade.get("sigma_pctl", None)
            if sigma_pctl is None and price_df is not None and "atr" in price_df.columns:
                look = price_df.loc[:day].tail(252)
                if not look.empty:
                    sigma_series = (look["atr"] / look["Close"]).dropna()
                    if len(sigma_series) >= 30:
                        current_sigma = atr / float(row["Close"])
                        sigma_pctl = float((sigma_series <= current_sigma).mean())
            if sigma_pctl is None:
                sigma_pctl = 0.5

            if   sigma_pctl <= 0.33: k_base = K_BASE_LOW
            elif sigma_pctl <= 0.66: k_base = K_BASE_MID
            else:                    k_base = K_BASE_HIGH

            # ------------- Trend-Health (TH) -------------
            if meta_pct is None:
                TH_meta = 0.5
            elif meta_pct <= 0.20:
                TH_meta = 1.0
            elif meta_pct >= 0.80:
                TH_meta = 0.0
            elif meta_pct < 0.50:
                TH_meta = max(0.0, min(1.0, (0.50 - meta_pct) / 0.30))
            else:
                TH_meta = 0.5

            TH_mom = (
                0.5 * _sigmoid((rsi - 55.0) / 5.0) +
                0.3 * _sigmoid(rsi_slope_3d / 0.5) +
                0.2 * (1 - int(macd_rollover == 1))
            )
            TH_mom = max(0.0, min(1.0, TH_mom))

            thr_slope = max(0.05, abs(ema50_slope))
            TH_struct = (
                0.35 * _sigmoid(price_vs_ema200 - 1.0) +
                0.35 * _sigmoid(ema50_slope / thr_slope) +
                0.30 * (1.0 if price_vs_max5 >= -0.03 else 0.0)
            )
            if donch_ll_10:
                TH_struct -= 0.20
            TH_struct = max(0.0, min(1.0, TH_struct))

            TH = 0.35 * TH_meta + 0.35 * TH_mom + 0.30 * TH_struct
            TH_hist = (TH_hist[-2:] + [TH]) if TH_hist else [TH]
            TH_ma3 = float(np.mean(TH_hist)) if len(TH_hist) > 0 else TH

            # Weak-TH persistence counter (NEW)
            if TH_ma3 < TH_TREND:
                weak_TH_streak += 1
            else:
                weak_TH_streak = 0

            trade["TH_hist"], trade["TH_ma3"] = TH_hist, TH_ma3
            trade["sigma_pctl"], trade["weak_TH_streak"] = sigma_pctl, weak_TH_streak

            # ------------- State transitions -------------
            if state == 0 and trade["day_count"] >= N_EARLY:
                state = 1
            if state == 1 and ((TH_ma3 < TH_LATE) or (meta_pct is not None and meta_pct >= 0.80)):
                late_votes = late_votes + 1
                if late_votes >= 2:
                    state = 2
            else:
                late_votes = max(0, late_votes - 1)
            trade["state"], trade["late_votes"] = state, late_votes

            # ------------- k policy -------------
            k_target = (
                k_base +
                (K_BONUS_EARLY if state == 0 else K_BONUS_TREND * TH_ma3 if state == 1 else -K_TIGHTEN_LATE)
            )
            if state == 2:
                k_target = max(K_TIGHT_FLOOR, k_target)
            if meta_pct is not None:
                if (meta_pct <= META_LOOSEN) and (TH_ma3 >= TH_TREND) and state == 1:
                    k_target += K_META_NUDGE
                if (meta_pct >= META_TIGHTEN) and (TH_ma3 < TH_LATE):
                    k_target -= K_META_NUDGE

            if "trail_k" not in trade:
                trade["trail_k"] = k_target
            if "trail_cooldown" not in trade:
                trade["trail_cooldown"] = 0
            if trade["trail_cooldown"] > 0:
                trade["trail_cooldown"] -= 1
            else:
                delta = k_target - float(trade["trail_k"])
                trade["trail_k"] += (
                    K_MAX_STEP if delta > K_MAX_STEP else
                    (-K_MAX_STEP if delta < -K_MAX_STEP else delta)
                )
                if delta < -1e-6:
                    trade["trail_cooldown"] = K_COOLDOWN_DAYS
            k_use = float(trade["trail_k"])

            # >>> Strong-context trail loosening (DYNAMIC; Upside Capture)
            # In clearly strong trends, give a small extra cushion so runners aren't clipped.
            if state == 1:  # Trending only
                strong_ctx = (TH_ma3 >= 0.60) or ((meta_pct is not None) and (meta_pct >= 0.75))
                if strong_ctx:
                    # nudge the effective k just a touch for this bar
                    k_use = max(k_use, float(trade["trail_k"]) + 0.15)


            # ------------- Trail -------------
            if price_df is not None and "Close" in price_df.columns:
                closes = price_df.loc[trade["entry_date"]:day]["Close"].dropna()
                highest_close = closes.max() if not closes.empty else row["Close"]
            else:
                highest_close = max(trade["entry_price"], row["Close"])

            active_trail = float(highest_close - k_use * atr)

            # ------------- Confirm ladder (tightened) -------------
            trail_broke     = row["Close"] < active_trail
            penetration_atr = (active_trail - float(row["Close"])) / max(atr, 1e-6)

            # A: structure
            A  = 0
            A += int(donch_ll_10 == 1 or float(row["Close"]) < ll10_price - 0.2 * atr)
            A += int((ema20 < ema50) and (ema50_slope <= 0))
            A += int(price_vs_max5 < -0.03)

            # B: momentum
            B  = 0
            B += int((rsi < 50.0) and (rsi_med3 < 52.0))
            B += int(rsi_slope_3d < -0.5)
            B += int(macd_roll_p2 == 1)

            closes_below_trail_count = (closes_below_trail_count + 1) if trail_broke else 0
            trade["closes_below_trail_count"] = closes_below_trail_count

            # High-vol stricter rules
            high_vol      = sigma_pctl > 0.66
            pen_trend_req = 0.45 if high_vol else PEN_TREND_ATR
            votes_req     = 3 if high_vol else 2

            if state == 0:
                confirm_ok, confirm_tag = False, "early_no_confirm"
            elif state == 1:
                # Need cross-domain: at least 1A & 1B, and total votes
                ok_votes = ((A >= 1) and (B >= 1) and ((A + B) >= votes_req))
                ok_depth = (penetration_atr >= pen_trend_req) or (NEED_2_CLOSES and closes_below_trail_count >= 2)
                confirm_ok, confirm_tag = (trail_broke and ok_votes and ok_depth), "trend_confirm"
            else:  # Late  ------- CHANGED -------
                # Cross-domain requirement in Late (noise filter)
                late_cross = (A >= 1) and (B >= 1) and ((A + B) >= 2)

                # Strong one-shot penetration can qualify without persistence — but must show some structure signal
                late_strong_pen = (penetration_atr >= LATE_PEN_ATR_STRONG)

                # A "regular" late confirm flag for this bar
                pen_late_req    = PEN_LATE_ATR_HIVOL if high_vol else PEN_LATE_ATR
                late_flag_today = trail_broke and (late_cross or (A >= 2) or (B >= 2) or (penetration_atr >= pen_late_req))

                # Persistence counter for Late confirms
                confirm_streak = trade.get("late_confirm_streak", 0)
                confirm_streak = (confirm_streak + 1) if late_flag_today else 0
                trade["late_confirm_streak"] = confirm_streak

                # PATCH START: require some structure on strong-penetration path
                if late_strong_pen:
                    strong_cross_ok = (A >= 1) or ((float(row["Close"]) < ema50) and (ema50_slope <= 0))
                    confirm_ok, confirm_tag = (strong_cross_ok, "late_confirm_strong")
                else:
                    # Otherwise: need persistence AND cross-domain
                    confirm_ok  = (confirm_streak >= LATE_CONFIRM_PERSIST_DAYS) and late_cross and (weak_TH_streak >= 2)
                    confirm_tag = "late_confirm_persist"
                # PATCH END
                    
                # >>> Late strong-context persistence bump (DYNAMIC; Upside Capture)
                # In strong late context, require one extra day of confirm persistence to avoid clipping benign pullbacks.
                if not late_strong_pen:
                    strong_ctx_late = (TH_ma3 >= 0.60) and (ema50_slope > 0) and (float(row["Close"]) >= ema50)
                    if confirm_ok and strong_ctx_late and (trade.get("late_confirm_streak", 0) < (LATE_CONFIRM_PERSIST_DAYS + 1)):
                        confirm_ok = False
                        confirm_tag = "late_confirm_wait_strong_ctx"


            # ------------- Giveback (state-aware; refined) -------------
            gb_active = (trade["day_count"] >= MIN_HOLD_FOR_GB) and (pnl_now >= MIN_GAIN_FOR_GB)
            gb_cap = None
            if gb_active and trade.get("peak_pnl", 0.0) >= GB_START_T1:
                if state == 1 and trade["peak_pnl"] < GB_START_T2:
                    # NEW: no T1 giveback in Trending to avoid mid-trend caps
                    gb_cap = None
                else:
                    if trade["peak_pnl"] < GB_START_T2:
                        gb_cap = (GB_CAP_T1_TREND if state == 1 else GB_CAP_T1_LATE)
                    else:
                        gb_cap = (GB_CAP_T2_TREND if state == 1 else GB_CAP_T2_LATE)
                    if meta_pct is not None and meta_pct >= META_TIGHTEN:
                        gb_cap = max(0.10, (gb_cap - GB_META_TIGHTEN))

            # ---- Dynamic giveback adjustment (NEW) ----
            if gb_cap is not None:
                strong_trend = (TH_ma3 >= 0.55) or ((meta_pct is not None) and (meta_pct >= 0.70) and (ema50_slope > 0))
                weak_trend   = (TH_ma3 <= 0.30) or ((float(row["Close"]) < ema50) and (ema50_slope <= 0))
                if strong_trend:
                    gb_cap = min(gb_cap + GB_STRONG_TREND_BONUS, gb_cap + 0.05)
                elif weak_trend:
                    gb_cap = max(0.10, gb_cap - GB_WEAK_TREND_TIGHTEN)


            # >>> Early-runner profit ratchet (DYNAMIC; Upside Capture)
            # If a trade runs hard early, ensure we don’t round-trip too much profit—even if Trend state disables T1.
            if gb_active:
                peak = float(trade.get("peak_pnl", 0.0))
                early = int(trade.get("day_count", 0))

                # Helper to set/clip a cap even if it was None
                def _cap_to(v):
                    nonlocal gb_cap
                    gb_cap = min((gb_cap if gb_cap is not None else 1.0), float(v))

                # Case A: fast +25% within first ~25 bars and context weakening → cap drawdown to ~18% of peak
                if (early <= 25) and (peak >= 0.25) and ((rsi_slope_3d < 0.0) or (TH_ma3 < TH_TREND)):
                    _cap_to(0.18)

                # Case B: fast +50% within first ~40 bars and momentum rolls over → cap drawdown to ~25% of peak
                if (early <= 40) and (peak >= 0.50) and ((rsi_slope_3d < -0.2) or (macd_roll_p2 == 1)):
                    _cap_to(0.25)


            # >>> Extra tightening in distinctly weak context (NEW)
            if gb_cap is not None:
                very_weak_ctx = (weak_TH_streak >= 3) or ((meta_pct is not None) and (meta_pct <= 0.25))
                if very_weak_ctx:
                    gb_cap = max(0.10, gb_cap - 0.03)


            drawdown_from_peak = 0.0
            if trade.get("peak_close", None):
                drawdown_from_peak = (trade["peak_close"] - float(row["Close"])) / max(1e-6, trade["peak_close"])
            cap_hit = (gb_cap is not None) and (drawdown_from_peak >= gb_cap)

            # ------------- Catastrophic floor (artifact guard) -------------
            ema200 = row.get("ema_200", None)
            hard_floor_price = trade["entry_price"] * 0.90
            if ema200 is not None and not pd.isna(ema200):
                hard_floor_price = min(hard_floor_price, float(ema200) * 0.92)
            catastrophic = float(row["Close"]) <= hard_floor_price
            if price_df is not None and "Close" in price_df.columns and len(price_df.loc[:day].tail(2)) == 2:
                prev = price_df.loc[:day].tail(2).iloc[0]["Close"]
                ret  = abs(float(row["Close"]) / float(prev) - 1.0)
                if ret >= SPLIT_ARTIFACT_RET:
                    catastrophic = False

            # ------------- Exit gating (tightened) -------------
            exit_now, exit_tag = False, None

            # 0) Raised catastrophic floor (only if the run never improved)
            if not exit_now and (trade.get("peak_pnl", 0.0) < RAISED_FLOOR_MIN_PEAK):
                # Context override: if trend health or meta is strong, skip raised-floor
                skip_raised_floor = (TH_ma3 >= 0.55) or (meta_pct is not None and meta_pct >= 0.70)

                if not skip_raised_floor:
                    close_px = row.get("Close", row.get("close"))
                    if close_px is not None and trade.get("entry_price") not in (None, 0):
                        pnl_now = (float(close_px) - float(trade["entry_price"])) / float(trade["entry_price"])
                        if pnl_now <= RAISED_FLOOR_MAX_LOSS:
                            exit_now = True
                            exit_tag = "raised_floor_weak_run"


            # 1) Catastrophic
            if catastrophic:
                # Require weak context or MA50 break to honor catastrophic exit
                cata_context = (
                    ((meta_pct is not None) and (meta_pct <= META_CATA) and (TH_ma3 < TH_CATA)) or
                    ((float(row["Close"]) < ema50) and (ema50_slope <= 0))
                )
                if cata_context:
                    exit_now, exit_tag = True, "catastrophic_floor"

            # 1b) Early fuses (run before confirm exits) — tuned to reduce over-trigger
            if not exit_now:
                # Weak context signals (mode-aware). Require 2-of-3, and a short persistence if below 200d.
                ema50_slope_val = row.get("ema50_slope")
                rsi_val         = row.get("rsi")

                sig_below200 = (p200_f is not None and p200_f <= anchor_p200)
                sig_ma50_neg = (ema50_slope_val is not None and float(ema50_slope_val) <= 0.0)
                sig_rsi_weak = (rsi_val is not None and float(rsi_val) < 50.0)

                weak_count    = int(sig_below200) + int(sig_ma50_neg) + int(sig_rsi_weak)
                # require 2-of-3 to call it "weak context"
                weak_ctx_2of3 = (weak_count >= 2)

                # if the below-200d signal is used, require brief persistence to avoid one-day blips
                below200_ok = (not sig_below200) or (int(trade.get("below200_streak", 0)) >= 2)
                weak_ctx    = weak_ctx_2of3 and below200_ok

                # safety valves: skip fuse if near a recent max, trend health is fine, or momentum is improving
                near_peak_5d  = (price_vs_max5 >= -NEAR_PEAK_WITHIN)  # within ~2% of last-5-day max
                trend_ok_now  = (TH_ma3 >= TH_TREND)
                improving_mom = (rsi_slope_3d > 0.0)

                # TUNED thresholds (gentler than raw E1)
                EARLY_MIN_DAYS = 16  # was 12
                NOPROG_MFE_MAX = 0.05  # was 0.08

                # No-progress fuse (gentler + extra guards)
                if (
                    ENABLE_NOPROG_FUSE and weak_ctx and
                    not near_peak_5d and
                    not trend_ok_now and
                    not improving_mom and
                    int(trade.get("day_count", 0)) >= int(EARLY_MIN_DAYS) and
                    float(trade.get("peak_pnl", 0.0)) < float(NOPROG_MFE_MAX)
                ):
                    ticker_for_cooldown = trade.get("ticker") or row.get("ticker")
                    if ticker_for_cooldown:
                        register_reentry_block(ticker_for_cooldown, bars=15)
                    exit_now = True
                    exit_tag = "early_no_progress_fuse"

                # Early give-up (conservative zombie cutter)
                elif (
                    ENABLE_EARLY_GIVEUP and
                    int(trade.get("day_count", 0)) >= int(GIVEUP_MIN_DAYS) and
                    float(trade.get("peak_pnl", 0.0)) < float(GIVEUP_MFE_MAX) and
                    float(pnl_now) <= float(E2_PNL_NOW_MAX) and
                    (meta_pct is None or float(meta_pct) >= float(E2_META_MIN)) and
                    (price_vs_max5 < -E2_NEAR_PEAK_WITHIN) and
                    # weak context: 2-of-3 (below200 / ema50_slope<=0 / RSI<50) with below200 persistence if used
                    (
                        ((int(sig_below200) + int(sig_ma50_neg) + int(sig_rsi_weak)) >= 2) and
                        ((not sig_below200) or (int(trade.get("below200_streak", 0)) >= 2))
                    )
                ):
                    ticker_for_cooldown = trade.get("ticker") or row.get("ticker")
                    if ticker_for_cooldown:
                        register_reentry_block(ticker_for_cooldown, bars=int(E2_REENTRY_BARS))
                    exit_now = True
                    exit_tag = "early_giveup_weak_context"

            # 2) Early: only catastrophic or extreme giveback
            if not exit_now and state == 0:
                if cap_hit and trade.get("peak_pnl", 0.0) >= GB_START_T2:
                    exit_now, exit_tag = True, "max_giveback_cap"

            # >>> Weak-context hard cap from peak (NEW)
            if (not exit_now
                and gb_active
                and (state in (1, 2))
                and ((weak_TH_streak >= 3) or ((meta_pct is not None) and (meta_pct <= 0.25)))
                and (drawdown_from_peak >= GB_HARD_CAP_WEAK)):
                exit_now, exit_tag = True, "max_giveback_cap_weak"


            # 3) Trending: need BOTH confirm and persistent weak TH (2d) — and no meta veto
            #     Allow single-day exit if TH plunges far below threshold (safety)
            if not exit_now and state_for_exit == 1 and confirm_ok:
                weak_ok = (weak_TH_streak >= 2) or (TH_ma3 < (TH_TREND - 0.05))
                if weak_ok and not (meta_pct is not None and meta_pct <= META_VETO):
                    exit_now, exit_tag = True, "protect_trail_confirmed"

            # 3.9) Late persist-below-200d (L1_v3 strict; honors pre-transition state)
            if not exit_now:
                st = state_for_exit
                if ENABLE_LATE_PERSIST_BELOW200D and st == 2:
                    below200_ok = int(trade.get("below200_streak", 0)) >= int(L1_MIN_BELOW200_STREAK)
                    trail_ok    = bool(trail_broke) or (NEED_2_CLOSES and int(trade.get("closes_below_trail_count", 0)) >= 2)

                    # Cross-domain weakness: structure + momentum OR price < MA50 & MA50 falling
                    cross_ok = ((A >= 1) and (B >= 1)) or ((float(row["Close"]) < ema50) and (ema50_slope <= 0.0))

                    # Persistently weak Trend Health in Late
                    weak_th_persist = (weak_TH_streak >= int(L1_REQUIRE_WEAK_TH_DAYS))

                    # Don’t clip benign pullbacks near prior highs; don’t fight strong meta
                    not_near_peak = (price_vs_max5 <= -float(L1_NEAR_PEAK_WITHIN))
                    meta_blocks   = (meta_pct is not None) and (meta_pct <= float(L1_META_BLOCK))

                    # Only punish weak-late runs with little progress + soft momentum
                    peak_ok = float(trade.get("peak_pnl", 0.0)) < float(L1_PEAK_CAP_FOR_EXIT)
                    mom_soft = (rsi < float(L1_MOM_RSI_MAX)) or (ema50_slope <= 0.0)

                    if below200_ok and trail_ok and cross_ok and weak_th_persist and not meta_blocks and not_near_peak and peak_ok and mom_soft:
                        exit_now = True
                        exit_tag = "late_persist_below200d"
                        tkr = trade.get("ticker") or row.get("ticker")
                        if tkr:
                            register_reentry_block(tkr, bars=int(L1_REENTRY_BARS))

            # 4) Late: confirm + weak trend required; meta veto (Late) applies  ------- CHANGED -------
            if not exit_now and state_for_exit == 2 and trail_broke:
                # Weak TH today or persisting from prior day(s)
                weak_ok = (weak_TH_streak >= 2) or (TH_ma3 < (TH_LATE - 0.03))

                # Stronger meta veto in Late
                meta_blocks = (meta_pct is not None) and (meta_pct <= META_VETO_LATE)

                # PATCH START: benign pullback guard in Late
                benign_hold = (
                    (float(row["Close"]) >= ema50) and
                    (rsi >= 45.0) and
                    (price_vs_max5 >= -0.06) and
                    (penetration_atr < LATE_PEN_ATR_STRONG)
                )
                # PATCH END

                if confirm_ok and weak_ok and not meta_blocks and not benign_hold:
                    exit_now, exit_tag = True, "late_distribution_confirmed"

            # ---- PATCH: context-aware giveback gate ----
            if not exit_now and cap_hit:
                strong_ctx = (
                    (TH_ma3 >= 0.55) and
                    (float(row["Close"]) >= ema20) and
                    (ema50_slope > 0) and
                    (state == 2)
                )
                if not strong_ctx:
                    exit_now, exit_tag = True, "max_giveback_cap"

            # ------------- Commit / log -------------
            trade.setdefault("debug_logs", []).append({
                "date": str(day),
                "state": state,
                "k_use": float(trade["trail_k"]),
                "k_base": k_base,
                "TH": TH,
                "TH_ma3": TH_ma3,
                "TH_meta": TH_meta,
                "TH_mom": TH_mom,
                "TH_struct": TH_struct,
                "A_votes": A,
                "B_votes": B,
                "confirm_tag": confirm_tag,
                "trail": active_trail,
                "trail_broke": bool(trail_broke),
                "penetration_atr": penetration_atr,
                "closes_below_trail_count": closes_below_trail_count,
                "meta_pct": meta_pct,
                "sigma_pctl": sigma_pctl,
                "gb_cap": gb_cap,
                "drawdown_from_peak": drawdown_from_peak,
                "catastrophic": catastrophic,
                "pnl_now": pnl_now,
                "weak_TH_streak": weak_TH_streak,
            })

            if exit_now:
                trade["exit_date"], trade["exit_price"]  = day, row["Close"]
                trade["exit_reason"], trade["final_return"] = exit_tag, pnl_now

                if price_df is not None and "High" in price_df.columns:
                    exit_window = price_df.loc[trade["entry_date"]:day]
                    # ---- PATCH: per-trade max drawdown ----
                    if price_df is not None and "Close" in price_df.columns:
                        path = price_df.loc[trade["entry_date"]:day]["Close"].astype(float)
                        if len(path) >= 2:
                            cum = path / float(trade["entry_price"])
                            roll_max = cum.cummax()
                            dd = (cum / roll_max - 1.0)
                            trade["trade_max_drawdown"] = float(dd.min())  # negative number

                    highs = exit_window["High"].dropna()
                    if not highs.empty:
                        max_price  = highs.max()
                        peak_date  = highs.idxmax()
                        trade["max_possible_return"] = (max_price - trade["entry_price"]) / trade["entry_price"]
                        trade["capture_ratio"] = (
                            trade["final_return"] / trade["max_possible_return"]
                            if trade["final_return"] > 0 and trade["max_possible_return"] > 0 else 0
                        )
                        trade["peak_date"] = peak_date
                        trade["exit_efficiency_days"] = abs((pd.Timestamp(day) - peak_date).days)

                print(
                    f"[POS] {ticker} EXIT @{row['Close']:.2f} "
                    f"reason={exit_tag} state={state} k={float(trade['trail_k']):.2f} "
                    f"pnl_now={pnl_now:.3f} peak_pnl={trade['peak_pnl']:.3f} "
                    f"TH={TH:.2f}/{TH_ma3:.2f} weakStk={weak_TH_streak} A={A} B={B} "
                    f"penATR={penetration_atr:.2f} meta_pct={(meta_pct if meta_pct is not None else float('nan')):.3f}"
                )


                results.append(trade)
                del open_positions[ticker]
                continue

        # === end manage open positions ===


        # === 2) Evaluate new entries (once per day) ===
        warned_bad_featlist = False
        for ticker, row in daily_data.items():
            if ticker in open_positions:
                continue

            entry_feats = [c for c in ENTRY_FEATURE_COLS if c in row.index]
            if len(entry_feats) != len(ENTRY_FEATURE_COLS) and not warned_bad_featlist:
                missing = [c for c in ENTRY_FEATURE_COLS if c not in row.index]
                print(f"[warn] {len(missing)} entry feature(s) missing; using intersection. Missing (first 10): {missing[:10]}")
                warned_bad_featlist = True
            if len(entry_feats) == 0:
                continue
            if row[entry_feats].isnull().any():
                continue

            vix = row["vix_close"]
            regime = "high" if vix > 30 else "caution" if vix > 20 else "low"
            model = models[regime]

            if not passes_entry_filters(row, regime):
                continue

            x = row.reindex(entry_feats).astype(float).to_numpy().reshape(1, -1)
            proba = model.predict_proba(x)[0, 1]
            threshold = {"low": 0.6, "caution": 0.5, "high": 0.30}[regime]
            if proba < threshold:
                continue

            # optional entry meta filter
            if meta_model is not None:
                meta_features = {
                    "proba": proba,
                    "core_proba_squared": proba ** 2,
                    "core_proba_high": int(proba > 0.7),
                    "core_proba_superhigh": int(proba > 0.85),
                    "rsi": row["rsi"],
                    "macd_diff": row["macd_diff"],
                    "price_vs_ema50": row["price_vs_ema50"],
                    "volume_surge": row["volume_surge"],
                    "stoch_d": row["stoch_d"],
                    "atr": row["atr"],
                    "volatility_regime_low": int(regime == "low"),
                    "volatility_regime_neutral": int(regime == "caution"),
                    "volatility_regime_high": int(regime == "high"),
                }
                x_meta = pd.DataFrame([meta_features])
                meta_proba = meta_model.predict_proba(x_meta)[0, 1]
                if meta_proba < 0.4 and proba <= 0.70:
                    continue
            else:
                meta_proba = None

            # open the position
            entry_row_slim = {
                "rsi": row["rsi"],
                "macd_diff": row["macd_diff"],
                "bb_pct": row["bb_pct"],
                "ema50_slope": row["ema50_slope"],
                "price_vs_ema200": row["price_vs_ema200"],
                "vix_close": row["vix_close"],
                "fear_greed": row["fear_greed"],
                "volume_surge": row["volume_surge"],
                "macro_trend_ok": row.get("macro_trend_ok", None),
                "volatility_regime": regime,
                "max_close": row["Close"],
            }

            open_positions[ticker] = {
                "ticker": ticker,
                "entry_date": day,
                "entry_price": row["Close"],
                "entry_regime": regime,
                "entry_features": entry_row_slim,
                "proba": proba,
                "meta_proba": meta_proba,
                "day_count": 0,
                # anti-churn state
                "grace_days_left": GRACE_DAYS_EXCEPT if proba >= EXCEPTIONAL_PROBA else GRACE_DAYS_BASE,
                "trail_break_count": 0,
            }

        # === 3) End of day update ===
        for trade in list(open_positions.values()):
            tkr = trade["ticker"]
            if tkr in daily_data:
                trade["last_seen_price"] = daily_data[tkr]["Close"]


    @staticmethod
    def simulate_one_day_meta(day, daily_data, feat_cols, models, active_trades, results, core_trade_state):
        still_open = []

        for trade in active_trades:
            ticker = trade["ticker"]
            if ticker not in daily_data:
                still_open.append(trade)
                continue

            row = daily_data[ticker]
            trade["day_count"] += 1

            trade["entry_features"]["max_close"] = max(trade["entry_features"].get("max_close", row["Close"]), row["Close"])
            trade["entry_features"]["min_close"] = min(trade["entry_features"].get("min_close", row["Close"]), row["Low"])

            if "meta_candidates" not in trade:
                trade["meta_candidates"] = []

            # === Candidate Sampling ===
            if trade["day_count"] >= 5:
                last_cand = trade["meta_candidates"][-1] if trade["meta_candidates"] else {}
                last_price = last_cand.get("exit_price", trade["entry_price"])
                price_move = abs(row["Close"] - last_price) / last_price

                if price_move > 0.01 or trade["day_count"] % 5 == 0:
                    try:
                        price_df = core_trade_state.get(ticker)
                        if price_df is None or price_df.empty:
                            print(f"[🚫] Missing or empty price_df for candidate sampling → {ticker}")
                            raise ValueError

                        price_df = price_df.copy()
                        price_df.index = pd.to_datetime(price_df.index).normalize()
                        day_norm = pd.to_datetime(day).normalize()

                        history_window = price_df.loc[:day_norm].tail(10)
                        if history_window.empty:
                            print(f"[⚠️] Empty history window for {ticker} on {day_norm}")
                            raise ValueError

                        exit_features = compute_exit_features(row, trade, history_window)

                        trade["meta_candidates"].append({
                            "exit_date": day,
                            "exit_price": row["Close"],
                            "features": exit_features
                        })

                    except Exception as e:
                        print(f"[⚠️] Candidate sampling failed for {ticker} on {day}: {e}")

            # === Exit Check ===
            vix = row["vix_close"]
            regime = "high" if vix > 30 else "caution" if vix > 20 else "low"

            exit_reason = check_exit_today(
                row,
                trade["entry_features"],
                trade["day_count"],
                trade["entry_price"],
                regime
            )

            if exit_reason:
                trade["exit_date"] = day
                trade["exit_price"] = row["Close"]
                trade["exit_reason"] = exit_reason
                trade["final_return"] = (row["Close"] - trade["entry_price"]) / trade["entry_price"]

                try:
                    price_df = core_trade_state.get(ticker)
                    if price_df is None or price_df.empty:
                        print(f"[🚫] Missing or empty price_df for EXIT labeling → {ticker}")
                        raise ValueError

                    price_df = price_df.copy()
                    price_df.index = pd.to_datetime(price_df.index).normalize()
                    entry_date = pd.to_datetime(trade["entry_date"]).normalize()
                    day_norm = pd.to_datetime(day).normalize()

                    # High watermark features
                    exit_window = price_df.loc[entry_date:day_norm]
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
                        trade["exit_efficiency_days"] = abs((pd.Timestamp(day) - peak_date).days)
                    else:
                        print(f"[⚠️] No highs for {ticker} from {entry_date} to {day_norm}")
                        trade.update({
                            "max_possible_return": None,
                            "capture_ratio": None,
                            "peak_date": None,
                            "exit_efficiency_days": None,
                        })

                    # Add series for utility label
                    trade["price_series"] = price_df.loc[entry_date:day_norm]

                except Exception as e:
                    print(f"[‼️] Error loading price_df for {ticker} on exit: {e}")
                    trade["price_series"] = pd.DataFrame()

                # Label summary
                trade["meta_label"] = 1 if trade["final_return"] > 0.20 else 0
                trade["capture_label"] = 1 if trade.get("capture_ratio", 0) >= 0.6 else 0

                # Final exit candidate
                try:
                    history_window = price_df.loc[:day_norm].tail(10)
                    exit_features = compute_exit_features(row, trade, history_window)
                    trade["exit_features"] = exit_features
                    trade["meta_candidates"].append({
                        "exit_date": day,
                        "exit_price": row["Close"],
                        "features": exit_features
                    })
                except Exception as e:
                    print(f"[⚠️] Final feature extraction failed for {ticker}: {e}")

                # Label meta candidates
                try:
                    label_meta_candidates_ranking(trade)
                except Exception as e:
                    print(f"[💥] Utility labeling failed for {ticker}: {e}")

                # Clean memory
                trade.pop("price_series", None)
                results.append(trade)
            else:
                still_open.append(trade)

        active_trades[:] = still_open

        # === New Entry Detection ===
        for ticker, row in daily_data.items():
            if row[feat_cols].isnull().any():
                continue

            vix = row["vix_close"]
            regime = "high" if vix > 30 else "caution" if vix > 20 else "low"
            model = models[regime]

            if not passes_entry_filters(row, regime):
                continue

            x = row[feat_cols].values.reshape(1, -1)
            proba = model.predict_proba(x)[0, 1]
            threshold = {"low": 0.6, "caution": 0.5, "high": 0.30}[regime]
            if proba < threshold:
                continue

            active_trades.append({
                "ticker": ticker,
                "entry_date": day,
                "entry_price": row["Close"],
                "entry_regime": regime,
                "proba": proba,
                "core_proba": proba,
                "day_count": 0,
                "entry_features": {
                    "max_close": row["Close"],
                    "min_close": row["Low"]
                },
                "features": {
                    "proba": proba,
                    "core_proba_squared": proba ** 2,
                    "core_proba_high": int(proba > 0.6),
                    "core_proba_superhigh": int(proba > 0.8),
                    "rsi": row.get("rsi"),
                    "macd_diff": row.get("macd_diff"),
                    "price_vs_ema50": row.get("price_vs_ema50"),
                    "volume_surge": row.get("volume_surge"),
                    "stoch_d": row.get("stoch_d"),
                    "atr": row.get("atr"),
                    "volatility_regime_low": int(regime == "low"),
                    "volatility_regime_neutral": int(regime == "caution"),
                    "volatility_regime_high": int(regime == "high")
                }
            })
