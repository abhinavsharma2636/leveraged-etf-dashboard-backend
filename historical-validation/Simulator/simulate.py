import pandas as pd
from .simulate_helpers import check_exit_today, passes_entry_filters, check_exit_condition 

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


    # @staticmethod
    # def simulate_one_day(day, daily_data, feat_cols, models, open_positions, results, core_trade_state):
    #     for ticker, row in daily_data.items():
    #         # if ticker not in core_trade_state:
    #         #     continue

    #         # 🔓 Disable core trade locking so we log every possible trade
    #         # Skip NaN inputs
    #         if row[feat_cols].isnull().any():
    #             print(f"[⚠️] {ticker} has missing values — skipping")
    #             continue

    #         vix = row["vix_close"]
    #         regime = "high" if vix > 30 else "caution" if vix > 20 else "low"
    #         model = models[regime]

    #         if not passes_entry_filters(row, regime):
    #             continue

    #         x = row[feat_cols].values.reshape(1, -1)
    #         proba = model.predict_proba(x)[0, 1]

    #         # ✅ Print all regime+proba signals
    #         print(f"📈 {day.date()} | {ticker} | Proba: {proba:.3f} | Regime: {regime}")

    #         threshold = {"low": 0.7, "caution": 0.6, "high": 0.55}[regime]
    #         if proba < threshold:
    #             continue

    #         # 🟢 Log raw buy indication only (do not lock)
    #         results.append({
    #             "ticker": ticker,
    #             "entry_date": day,
    #             "entry_price": row["Close"],
    #             "entry_regime": regime,
    #             "proba": proba
    #         })

    #         # Optional: suppress this line if you want to allow repeated buys
    #         # core_trade_state[ticker] = day

    @staticmethod
    def simulate_one_day(day, daily_data, feat_cols, models, open_positions, results, core_trade_state, meta_model=None):
        for ticker, row in daily_data.items():
            # Check existing position
            if ticker in open_positions:
                trade = open_positions[ticker]
                trade["day_count"] += 1

                # Update trailing stop
                trade["entry_features"]["max_close"] = max(
                    trade["entry_features"]["max_close"],
                    row["Close"]
                )

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
                    trade["final_return"] = (row["Close"] - trade["entry_price"]) / trade["entry_price"]  # ✅ add return here too
                    results.append(trade)
                    del open_positions[ticker]
                continue

            # Check for missing features
            if row[feat_cols].isnull().any():
                print(f"[⚠️] {ticker} has missing values — skipping")
                continue

            # Model & regime
            vix = row["vix_close"]
            regime = "high" if vix > 30 else "caution" if vix > 20 else "low"
            model = models[regime]

            # Entry filters
            if not passes_entry_filters(row, regime):
                continue

            x = row[feat_cols].values.reshape(1, -1)
            proba = model.predict_proba(x)[0, 1]
            print(f"📈 {day.date()} | {ticker} | Proba: {proba:.3f} | Regime: {regime}")

            threshold = {"low": 0.6, "caution": 0.5, "high": 0.30}[regime]
            if proba < threshold:
                continue

            # ✅ Meta model check (if provided)
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
                    "volatility_regime_neutral": int(regime == "neutral"),
                    "volatility_regime_high": int(regime == "high"),
                }

                x_meta = pd.DataFrame([meta_features])
                meta_proba = meta_model.predict_proba(x_meta)[0, 1]
                print(f"Meta Proba: {meta_proba:.3f} | Core Proba: {proba:.3f} | Ticker: {ticker} | Regime: {regime}")

                if meta_proba < 0.4 and proba <= 0.70:
                    continue


            # ✅ Entry accepted
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
                "max_close": row["Close"]
            }

            open_positions[ticker] = {
                "ticker": ticker,
                "entry_date": day,
                "entry_price": row["Close"],
                "entry_regime": regime,
                "entry_features": entry_row_slim,
                "proba": proba,
                "meta_proba": meta_proba,
                "day_count": 0
            }

        # ✅ After all processing, update last_seen_price
        for trade in open_positions.values():
            ticker = trade["ticker"]
            if ticker in daily_data:
                trade["last_seen_price"] = daily_data[ticker]["Close"]




    @staticmethod
    def simulate_one_day_meta(day, daily_data, feat_cols, models, active_trades, results):
        # === 1. Check exits for all active trades ===
        still_open = []

        for trade in active_trades:
            ticker = trade["ticker"]
            if ticker not in daily_data:
                still_open.append(trade)
                continue

            row = daily_data[ticker]
            trade["day_count"] += 1
            trade["entry_features"]["max_close"] = max(
                trade["entry_features"]["max_close"],
                row["Close"]
            )

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
                trade["final_return"] = (trade["exit_price"] - trade["entry_price"]) / trade["entry_price"]

                # ✅ Apply your meta labeling rule
                if trade["final_return"] > 0.20 and exit_reason not in ["exit_emergency_stop", "exit_failed_reversal"]:
                    trade["meta_label"] = 1
                else:
                    trade["meta_label"] = 0

                results.append(trade)
            else:
                still_open.append(trade)

        active_trades[:] = still_open  # update in-place

        # === 2. Evaluate new entries ===
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

            # ✅ Create meta features (clean + consistent with META_FEATURE_COLS)
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
                "volatility_regime_neutral": int(regime == "neutral"),
                "volatility_regime_high": int(regime == "high"),
            }

            active_trades.append({
                "ticker": ticker,
                "entry_date": day,
                "entry_price": row["Close"],
                "entry_regime": regime,
                "features": meta_features,
                "entry_features": {
                    "max_close": row["Close"]  # ✅ Needed for exit check
                },
                "proba": proba,
                "day_count": 0
            })








