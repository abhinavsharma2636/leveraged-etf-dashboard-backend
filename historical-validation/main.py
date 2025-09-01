# main.py
# Entry point for regime-aware ML trading pipeline

import argparse
import datetime
import json as json_mod
import os
from dateutil.relativedelta import relativedelta
import pandas as pd
from sklearn.utils import resample
from Data_Manager.data import DataManager
from Labeler.labeling import Labeler
from Labeler.labeling_functions import LabelingFunctions
from Model.model import ModelTrainer
from Simulator.simulate import TradeSimulator
import numpy as np
from Evaluate.evaluate import Evaluator
from collections import deque
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score








from Simulator.simulate_helpers import check_exit_today, compute_exit_features, compute_feature_auc_scores, finalize_trade, label_meta_candidates_ranking

monthly_label_stats = []
def print_label_distribution_by_regime(df: pd.DataFrame):
    if "label_type" not in df.columns or "target" not in df.columns:
        print("   ↪︎ [Missing regime labels]")
        return

    labeled = df.dropna(subset=["target"])

    regimes = labeled["label_type"].unique()
    for regime in sorted(regimes):
        subset = labeled[labeled["label_type"] == regime]
        counts = subset["target"].value_counts(normalize=True).to_dict()
        total = len(subset)
        print(f"   ↪︎ Regime: {regime} ({total} samples)")
        for label in [0.0, 1.0]:
            pct = round(counts.get(label, 0.0) * 100, 2)
            print(f"     Label {int(label)}: {pct}%")

# ─── USER-CONFIGURED TICKERS ─────────────────────────────────────────────────────
training_tickers = [
    "AAPL", "MSFT", "INTC", "CSCO", "IBM", "ORCL",
    "TXN", "ADI", "JPM", "BAC", "WFC", "GS", "AXP",
    "PG", "KO", "PEP", "WMT", "COST", "GE", "CAT",
    "MMM", "HON", "JNJ", "PFE", "MRK", "ABBV", "UNH",
    "SPY", "QQQ", "DIA", "IWM", "XOM", "CVX", "DUK",
    "NEE", "HD", "LOW", "TGT", "MCD", "NKE", "SBUX"
]


meta_training_tickers = [
    # Breakout-from-base
    "NVDA", "AMD", "MU", "NFLX", "BIIB", "MA", "V", "ADSK", "NOW", "EBAY", "CRM",
    
    # Stable structure names
    "AAPL", "MSFT", "KO", "PG", "UNH", "JNJ", "WMT", "COST", "HD",
    
    # Trap/failure-prone (confirmed 2010–2019)
    "GE", "FSLR", "F", "BBY", "WU", "TAP", "ET"
]

# meta_training_tickers = [
#     # Breakout-from-base
#     "NVDA", 
    
#     # Stable structure names
#     "AAPL", 
    
#     # Trap/failure-prone (confirmed 2010–2019)
#     "FSLR", "WU"
# ]



LABEL_DIR = "Labeler/data_labeling"
def parse_args():
    p = argparse.ArgumentParser(description="High-Confidence Entry Detection with Extended Exit Rules")
    p.add_argument("--test_year", type=int, required=True)
    p.add_argument("--test_tickers", nargs="+", default=training_tickers)
    return p.parse_args()

from collections import deque

def main():
    args = parse_args()
    start_year = args.test_year

    start_date = datetime.date(start_year, 1, 1)             # when simulation starts
    today = datetime.date.today()
    end_month = datetime.date(today.year, today.month, 1)    # simulation end (rounded to month start)

    # ✅ Pull 16 years of data before the test start year (for training)
    data_start_date = (start_date - datetime.timedelta(days=365 * 16)).strftime("%Y-%m-%d")
    data_end_date = today.strftime("%Y-%m-%d")
    feat_cols = [
        "atr", "vix_close", "macd_diff", "rsi", "volume_surge",
        "price_vs_ema50", "stoch_d",
        "volatility_regime_low", "volatility_regime_neutral", "volatility_regime_high"
    ]

    META_FEATURE_COLS = [
    "proba",
    "core_proba_squared",
    "core_proba_high",
    "core_proba_superhigh",
    "rsi",
    "macd_diff",
    "price_vs_ema50",
    "volume_surge",
    "stoch_d",
    "atr",
    "volatility_regime_low",
    "volatility_regime_neutral",
    "volatility_regime_high"
]
    
    EXIT_FEATURE_COLS = [
    # --- Returns & Path ---
    "pct_of_peak_captured", "vol_adj_return",

    # --- Trend Decay ---
    "price_trend_slope_5", "price_vs_max_last_5",
    "price_vs_ema5", "price_vs_ema20", "ema5_slope",
    "macd_diff", "macd_rolling_3", "rsi", "rsi_slope_3d",
    "rsi_above_70", "stoch_d", "stoch_k",

    # --- Lifecycle / Duration ---
     "days_since_max_price", "exit_efficiency_days",

    # --- Drawdown / Risk ---
    "rolling_max_drawdown", "entry_to_peak_drawdown", "price_vs_ema50", "volatility_5d",

    # --- Regime & Sentiment ---
    "entry_regime_low", "entry_regime_caution", "entry_regime_high",
    "exit_regime_low", "exit_regime_caution", "exit_regime_high",
    "vix_close", "vix_spike",

    # --- Core Signal ---
    "core_proba", "core_proba_high",

    # --- Breakdown Signal ---
    "macd_rollover", "price_near_trailing_peak"
    ]


    dm = DataManager(data_start_date, data_end_date)
    dm.download_all_macro()

    labeler = Labeler(label_dir=LABEL_DIR)
    labeling = LabelingFunctions()
    rolling_train_df = deque(maxlen=180)

    import pandas as pd

    print("Initializing 15-year training window...")

    train_start = datetime.date(start_year - 15, 1, 1)
    months = pd.date_range(start=train_start, end=start_date - pd.offsets.MonthBegin(1), freq="MS")


    ticker_feats = {}
    for ticker in training_tickers:
        raw = dm.download_stock_data(ticker)
        feats = dm.compute_features(raw)

        feats["ema50_slope"] = feats["ema50"].diff(3)
        feats["vix_5d_slope"] = feats["vix_close"].diff(5)
        feats["fear_greed_slope"] = feats["fear_greed"].diff(5)
        feats["macro_trend_ok"] = (
            (feats["price_vs_ema200"] > 0.01) &
            (feats["rsi"] > 50) &
            (feats["vix_5d_slope"] < 0)
        )

        ticker_feats[ticker] = feats

    # ✅ Step 2: Loop month-by-month and use cached features
    monthly_label_stats = []

    for month_start in months:
        month_start = month_start.date()
        month_end = (month_start + pd.offsets.MonthEnd(0)).date()
        monthly_labeled_dfs = []

        for ticker in training_tickers:
            feats = ticker_feats[ticker]

            feats_start = feats.index.min().date()
            feats_end = feats.index.max().date()

            if month_end < feats_start or month_start > feats_end:
                continue

            # Slice current month
            month_feats = feats.loc[str(month_start):str(month_end)].copy()
            if month_feats.empty:
                continue

            # Slice window for labeling (month + 180d lookahead)
            label_window_end = (month_end + datetime.timedelta(days=180))
            window_feats = feats.loc[str(month_start):str(label_window_end)].copy()
            if len(window_feats) < 30:
                continue

            # Label and cache to parquet
            labeled = labeler.load_or_label_monthly(
                window_feats,
                ticker,
                month_start.year,
                month_start.month,
                {
                    "high": labeling.label_rsi_reversal_triple_barrier,
                    "caution": labeling.label_stage2_confirmed_triple_barrier,
                    "low": labeling.label_stage2_breakout_triple_barrier,
                },
            )

            if "entry_date" not in labeled.columns:
                continue

            # Filter entries to current month only
            labeled = labeled[
                (labeled["entry_date"] >= pd.Timestamp(month_start)) &
                (labeled["entry_date"] <= pd.Timestamp(month_end))
            ]

            if not labeled.empty:
                monthly_labeled_dfs.append(labeled)

        # Combine and append results
        if monthly_labeled_dfs:
            combined_month = pd.concat(monthly_labeled_dfs)
            monthly_label_stats.append({
                "month": pd.Timestamp(month_start),
                "num_rows": len(combined_month),
                "label_1_pct": combined_month["target"].mean(),
                "regime_counts": combined_month["label_type"].value_counts().to_dict()
            })

            label_dist = combined_month["target"].value_counts(normalize=True).to_dict()
            print(f"     ↪︎ Label distribution (pre-oversampling): {label_dist}")
            print_label_distribution_by_regime(combined_month)
            rolling_train_df.append(combined_month)
            print(f"[📊] Month {month_start.strftime('%Y-%m')} → {len(combined_month)} rows")
        else:
            print(f"[📭] Month {month_start.strftime('%Y-%m')} → No labeled data from any ticker")

    train_df = pd.concat(list(rolling_train_df)).dropna(subset=feat_cols)

    print("[📊] Label distribution before initial training:")
    print(train_df["target"].value_counts(normalize=True))

    print(f"Training initial models on {len(train_df)} rows…")
    main_model = ModelTrainer(feat_cols).train(train_df)
    dip_model = ModelTrainer(feat_cols).train(train_df[train_df["label_type"] == "high"])
    caution_model = ModelTrainer(feat_cols).train(train_df[train_df["label_type"] == "caution"])

    current_models = {
        "low": main_model,
        "caution": caution_model,
        "high": dip_model
    }

    # ─── META Live Simulation ─────────────────────────────────────

    meta_start = datetime.date(start_year - 10, 1, 1)
    meta_months = pd.date_range(start=meta_start, end=start_date - pd.offsets.MonthBegin(1), freq="MS")
    meta_end_month = datetime.date(start_year, 1, 1)

    # ─── Preload all features for meta simulation tickers ─────────────────────────────
    print("Preloading feature data for meta training tickers...")

    meta_ticker_feats = {}
    for t in meta_training_tickers:
        raw = dm.download_stock_data(t)
        feats = dm.compute_features(raw)

        # Add derived columns once (not per day)
        feats["ema50_slope"] = feats["ema50"].diff(3)
        feats["vix_5d_slope"] = feats["vix_close"].diff(5)
        feats["fear_greed_slope"] = feats["fear_greed"].diff(5)
        feats["macro_trend_ok"] = (
            (feats["price_vs_ema200"] > 0.01) &
            (feats["rsi"] > 50) &
            (feats["vix_5d_slope"] < 0)
        )

        meta_ticker_feats[t] = feats

    latest_possible_date = min(
    feats.index.max().date()
    for feats in meta_ticker_feats.values()
    if feats is not None and not feats.empty
)

    # Clip end month to avoid going beyond available data
    meta_end_month = min(meta_end_month, latest_possible_date)
    print(f"🧭 Adjusted meta_end_month to {meta_end_month} based on available data.")
        

    # ─── Run Meta Simulation Month by Month ──────────────────────────────────────
    # === Check for cached simulation/model results ===
    # === Load cached simulation results & trained models ===
    if os.path.exists("meta_results.json") and os.path.exists("exit_meta_candidates.feather") \
    and os.path.exists("exit_meta_model.pkl") and os.path.exists("exit_meta_features.json"):

        print("🔁 Loading cached simulation + exit meta model...")

        # Load meta_results (for context if needed)
        with open("meta_results.json", "r") as f:
            meta_results = json_mod.load(f)

        # Load candidate DataFrame
        exit_meta_df = pd.read_feather("exit_meta_candidates.feather")

        # Load model + trained feature order
        exit_meta_model = joblib.load("exit_meta_model.pkl")
        with open("exit_meta_features.json", "r") as f:
            trained_features = json_mod.load(f)

        print(f"[ℹ️] Loaded {len(exit_meta_df)} exit candidates, "
            f"{len(trained_features)} trained features.")

        # === Prepare for evaluation ===
        # Align features exactly to trained feature set
        X = exit_meta_df.reindex(columns=trained_features, fill_value=0.0).astype(float)

        # Drop rows with NaNs in model features
        na_mask = X.isna().any(axis=1)
        if na_mask.any():
            print(f"[warn] Dropping {na_mask.sum()} rows with NaNs in model features.")
            X = X.loc[~na_mask]
            exit_meta_df = exit_meta_df.loc[~na_mask]

        # Predict scores
        exit_meta_df["score"] = exit_meta_model.predict(X)

        # Pick truth column
        truth_col = "rel" if ("rel" in exit_meta_df.columns and exit_meta_df["rel"].sum() > 0) else "target"
        exit_meta_df[truth_col] = pd.to_numeric(exit_meta_df[truth_col], errors="coerce").fillna(0).astype(int)

        # Drop singleton groups
        grp_sizes = exit_meta_df.groupby("group_id").size()
        exit_meta_df = exit_meta_df[exit_meta_df["group_id"].isin(grp_sizes[grp_sizes >= 2].index)].copy()

        # Ranks
        exit_meta_df["pred_rank"] = exit_meta_df.groupby("group_id")["score"].rank("first", ascending=False)
        exit_meta_df["true_rank"] = exit_meta_df.groupby("group_id")[truth_col].rank("first", ascending=False)

        # === Accuracy metrics ===
        def topk_overlap(g, k):
            pred_top = g.nlargest(k, "score").index
            true_top = g.nlargest(k, truth_col).index
            return len(set(pred_top) & set(true_top)) / max(1, len(set(true_top)))

        top1_acc = exit_meta_df.groupby("group_id").apply(topk_overlap, k=1).mean()
        top3_acc = exit_meta_df.groupby("group_id").apply(topk_overlap, k=3).mean()
        top5_acc = exit_meta_df.groupby("group_id").apply(topk_overlap, k=5).mean()

        print(f"\n🎯 Top-1 Accuracy: {top1_acc:.2%}")
        print(f"🎯 Top-3 Accuracy: {top3_acc:.2%}")
        print(f"🎯 Top-5 Accuracy: {top5_acc:.2%}")

        # === Preview ===
        model_top_idx = exit_meta_df.groupby("group_id")["score"].idxmax()
        true_top_idx  = exit_meta_df.groupby("group_id")[truth_col].idxmax()

        n_show = min(5, len(model_top_idx))
        if n_show > 0:
            print("\n🔍 Sample Predicted Top Exits:")
            print(exit_meta_df.loc[model_top_idx, ["group_id", "score", truth_col, "pred_rank", "exit_price"]]
                    .sample(n_show, replace=False))

            print("\n🔍 Sample True Best Exits:")
            print(exit_meta_df.loc[true_top_idx, ["group_id", "score", truth_col, "true_rank", "exit_price"]]
                    .sample(n_show, replace=False))

        # === Feature Importances ===
        # try:
        #     gains = exit_meta_model.booster_.feature_importance(importance_type="gain")
        #     names = exit_meta_model.booster_.feature_name()
        #     s = pd.Series(gains, index=names).sort_values()
        #     s.plot.barh(figsize=(8, 6), title="EXIT Meta Ranker Feature Importances (gain)")
        #     plt.tight_layout()
        #     plt.show()
        # except Exception as e:
        #     print(f"[warn] Could not plot feature importances: {e}")

        if os.path.exists("entry_meta_model.pkl"):
            meta_model = joblib.load("entry_meta_model.pkl")
            print("🧠 Loaded entry_meta_model.pkl")
        else:
            meta_model = None
            print("🧠 entry_meta_model.pkl not found (meta_model=None)")

    else:
        print("🚀 Running meta simulation and training from scratch...")

        # === Run simulation as you already had ===
        current_month = meta_start
        meta_results = []
        active_trades = []
        core_trade_state = {t: meta_ticker_feats[t] for t in meta_training_tickers}

        while current_month <= meta_end_month:
            print(f"\n=== META SIMULATING {current_month.strftime('%Y-%m')} ===")
            month_start = current_month
            month_end = (current_month + relativedelta(months=1)) - datetime.timedelta(days=1)
            date_range = pd.date_range(month_start, month_end, freq="B")

            for current_day in date_range:
                daily_data = {}
                for t in meta_training_tickers:
                    feats = meta_ticker_feats[t]
                    if current_day not in feats.index:
                        continue
                    daily_data[t] = feats.loc[current_day]

                TradeSimulator.simulate_one_day_meta(
                    current_day,
                    daily_data,
                    feat_cols,
                    current_models,
                    active_trades,
                    meta_results,
                    core_trade_state=core_trade_state
                )
            current_month += relativedelta(months=1)

        # === Finalize open trades ===
        for trade in active_trades:
            finalize_trade(trade, meta_end_month, core_trade_state)

        # === Save meta_results ===
        with open("meta_results.json", "w") as f:
            json_mod.dump(meta_results, f, default=str)

        # === Prepare exit candidates for model training ===
        # === Build exit_meta_df from meta_results ===
        # ==== Assemble candidate rows ====
        exit_meta_candidate_rows = []

        for trade in meta_results:
            for cand in trade.get("meta_candidates", []) or []:
                feat = (cand.get("features", {}) or {}).copy()
                row = {
                    **feat,

                    # ── labels (robust) ──
                    "group_id": cand.get("group_id") or f"{trade.get('entry_date')}_{trade.get('ticker')}",
                    "target":   cand.get("target", 0),
                    "rel":      cand.get("rel", 0),
                    "utility":  cand.get("utility", np.nan),

                    # ── utility components ──
                    "gain_lock_ratio": cand.get("gain_lock_ratio"),
                    "future_drawdown": cand.get("future_drawdown"),
                    "future_regret":   cand.get("future_regret"),
                    "momentum_signal": cand.get("momentum_signal"),
                    "momentum_penalty": cand.get("momentum_penalty"),
                    "pct_from_peak":   cand.get("pct_from_peak"),
                    "peak_proximity":  cand.get("peak_proximity"),
                    "time_penalty":    cand.get("time_penalty"),
                    "regret_penalty":  cand.get("regret_penalty"),
                    "is_best_candidate": int(cand.get("is_best_candidate", 0)),

                    # ── context ──
                    "exit_date":  cand.get("exit_date"),
                    "exit_price": cand.get("exit_price"),
                    "ticker":     trade.get("ticker"),
                    "entry_date": trade.get("entry_date"),
                    "entry_price": trade.get("entry_price"),
                    "final_return": trade.get("final_return", 0.0),
                    "regime":       trade.get("entry_regime"),
                }
                exit_meta_candidate_rows.append(row)

        # ── Build DF ──
        exit_meta_df = pd.DataFrame(exit_meta_candidate_rows)

        # Core fields must exist
        exit_meta_df = exit_meta_df.dropna(subset=["group_id", "entry_price", "exit_price"]).copy()

        # Return delta
        exit_meta_df["return_delta"] = (
            (exit_meta_df["exit_price"] - exit_meta_df["entry_price"]) / exit_meta_df["entry_price"]
        ) - exit_meta_df["final_return"]

        # Coerce dates & make stable group_id
        exit_meta_df["entry_date"] = pd.to_datetime(exit_meta_df["entry_date"], errors="coerce")
        exit_meta_df["exit_date"]  = pd.to_datetime(exit_meta_df["exit_date"],  errors="coerce")
        exit_meta_df["group_id"]   = exit_meta_df["entry_date"].dt.strftime("%Y-%m-%d") + "_" + exit_meta_df["ticker"].astype(str)

        # Persist raw candidates (optional)
        exit_meta_df.to_feather("exit_meta_candidates.feather")

        # ── Choose labels robustly ──
        for c in ("rel", "target"):
            if c in exit_meta_df.columns:
                exit_meta_df[c] = pd.to_numeric(exit_meta_df[c], errors="coerce").fillna(0).astype(int)

        use_rel   = ("rel" in exit_meta_df.columns) and (exit_meta_df["rel"].sum() > 0)
        label_col = "rel" if use_rel else "target"

        # ── Keep only features we actually have ──
        train_feats = [c for c in EXIT_FEATURE_COLS if c in exit_meta_df.columns]
        if not train_feats:
            raise ValueError("No training features found in exit_meta_df. Check EXIT_FEATURE_COLS vs feature engineering.")

        # Drop NaNs only on what we strictly need
        ranker_data = exit_meta_df.dropna(subset=train_feats + ["group_id", label_col]).copy()

        # Drop singleton groups (ranker needs >1 candidate per group)
        grp_sizes = ranker_data.groupby("group_id").size()
        keep_groups = grp_sizes[grp_sizes >= 2].index
        ranker_data = ranker_data[ranker_data["group_id"].isin(keep_groups)].copy()

        # Final safety: ensure at least some positives
        if ranker_data[label_col].sum() == 0:
            # last resort: promote each group's best-utility candidate to rel=1
            ranker_data["utility"] = pd.to_numeric(ranker_data.get("utility"), errors="coerce")
            top_idx = ranker_data.groupby("group_id")["utility"].idxmax()
            ranker_data.loc[top_idx, label_col] = 1
            print("[warn] No positives in labels; promoted group-utility maxima to positives.")

        # ── Train Exit Ranker ──
        exit_trainer = ModelTrainer(feature_cols=train_feats)
        exit_meta_model = exit_trainer.train_exit_ranker(ranker_data)
        joblib.dump(exit_meta_model, "exit_meta_model.pkl")
        print("[✅] Saved exit_meta_model.pkl")

        # Persist the EXACT trained feature order for later prediction
        trained_features = list(exit_meta_model.booster_.feature_name())
        with open("exit_meta_features.json", "w") as f:
            json_mod.dump(trained_features, f)
        print(f"[✅] Saved exit feature list ({len(trained_features)}) to exit_meta_features.json")

        print("\n📈 Predicting scores and computing rank...")

        # === PREDICT WITH THE EXACT TRAINED FEATURE LIST ===
        with open("exit_meta_features.json", "r") as f:
            trained_features = json_mod.load(f)

        # Strict align: reindex to trained features; ignore extras, fill missing with 0.0
        X = exit_meta_df.reindex(columns=trained_features, fill_value=0.0).astype(float)

        extras = [c for c in exit_meta_df.columns if c not in trained_features]
        if extras:
            print(f"[info] Ignoring {len(extras)} extra columns not in model: {extras[:10]}{'...' if len(extras) > 10 else ''}")

        na_mask = X.isna().any(axis=1)
        if na_mask.any():
            print(f"[warn] Dropping {na_mask.sum()} rows with NaNs in model features for eval.")
            X = X.loc[~na_mask]
            exit_meta_df = exit_meta_df.loc[~na_mask]

        exit_meta_df["score"] = exit_meta_model.predict(X)

        # choose truth
        truth_col = "rel" if ("rel" in exit_meta_df.columns and exit_meta_df["rel"].sum() > 0) else "target"
        exit_meta_df[truth_col] = pd.to_numeric(exit_meta_df[truth_col], errors="coerce").fillna(0).astype(int)

        # drop singleton groups
        grp_sizes = exit_meta_df.groupby("group_id").size()
        exit_meta_df = exit_meta_df[exit_meta_df["group_id"].isin(grp_sizes[grp_sizes >= 2].index)].copy()

        # ranks and overlap
        exit_meta_df["pred_rank"] = exit_meta_df.groupby("group_id")["score"].rank("first", ascending=False)
        exit_meta_df["true_rank"] = exit_meta_df.groupby("group_id")[truth_col].rank("first", ascending=False)

        def topk_overlap(g, k):
            pred_top = g.nlargest(k, "score").index
            true_top = g.nlargest(k, truth_col).index
            return len(set(pred_top) & set(true_top)) / max(1, len(set(true_top)))

        top1_acc = exit_meta_df.groupby("group_id").apply(topk_overlap, k=1).mean()
        top3_acc = exit_meta_df.groupby("group_id").apply(topk_overlap, k=3).mean()
        top5_acc = exit_meta_df.groupby("group_id").apply(topk_overlap, k=5).mean()

        print(f"\n🎯 Top-1 Accuracy: {top1_acc:.2%}")
        print(f"🎯 Top-3 Accuracy: {top3_acc:.2%}")
        print(f"🎯 Top-5 Accuracy: {top5_acc:.2%}")

        # Preview
        model_top_idx = exit_meta_df.groupby("group_id")["score"].idxmax()
        true_top_idx  = exit_meta_df.groupby("group_id")[truth_col].idxmax()

        n_show = min(5, len(model_top_idx))
        if n_show > 0:
            print("\n🔍 Sample Predicted Top Exits:")
            print(exit_meta_df.loc[model_top_idx, ["group_id", "score", truth_col, "pred_rank", "exit_price"]]
                    .sample(n_show, replace=False))
            print("\n🔍 Sample True Best Exits:")
            print(exit_meta_df.loc[true_top_idx, ["group_id", "score", truth_col, "true_rank", "exit_price"]]
                    .sample(n_show, replace=False))

        # Importances
        try:
            gains = exit_meta_model.booster_.feature_importance(importance_type="gain")
            names = exit_meta_model.booster_.feature_name()
            s = pd.Series(gains, index=names).sort_values()
            s.plot.barh(figsize=(8,6), title="EXIT Meta Ranker Feature Importances (gain)")
            plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"[warn] importances not available: {e}")

        # === Train entry meta model (unchanged logic) ===
        print("\n🧠 Training entry meta model...")
        meta_df = pd.DataFrame(meta_results).dropna(subset=["features", "meta_label"])
        features_df = pd.json_normalize(meta_df["features"])
        features_df = features_df[META_FEATURE_COLS]
        # Force numeric
        features_df = features_df.apply(pd.to_numeric, errors="coerce")
        meta_data = pd.concat([features_df, meta_df["meta_label"]], axis=1)
        meta_trainer = ModelTrainer(feature_cols=META_FEATURE_COLS)
        meta_model = meta_trainer.train_meta(meta_data)
        joblib.dump(meta_model, "entry_meta_model.pkl")
        print("[✅] Saved entry_meta_model.pkl")

    # ──────────────────────────────────────────────────────────────────────────────
    # Monthly Live Simulation (feature alignment enforced)
    print("Preloading feature data for simulation tickers...")

    sim_ticker_feats = {}
    for t in args.test_tickers:
        raw = dm.download_stock_data(t)
        feats = dm.compute_features(raw)

        # Add derived columns once (not per day)
        feats["ema50_slope"] = feats["ema50"].diff(3)
        feats["vix_5d_slope"] = feats["vix_close"].diff(5)
        feats["fear_greed_slope"] = feats["fear_greed"].diff(5)
        feats["macro_trend_ok"] = (
            (feats["price_vs_ema200"] > 0.01) &
            (feats["rsi"] > 50) &
            (feats["vix_5d_slope"] < 0)
        )
        sim_ticker_feats[t] = feats

    # Load trained exit model + exact feature order for live scoring
    exit_meta_model = joblib.load("exit_meta_model.pkl")
    exit_trained_features = list(exit_meta_model.booster_.feature_name())
    print(f"[exit] Using {len(exit_trained_features)} trained exit features for live sim.")

    # ─── Monthly Live Simulation ─────────────────────────────────────
    current_month = start_date
    all_results = []
    open_positions = {}
    closed_keys = set()  # (ticker, entry_date) we've already recorded
    core_trade_state = {t: sim_ticker_feats[t] for t in args.test_tickers}

    while current_month <= end_month:
        print(f"\n=== SIMULATING {current_month.strftime('%Y-%m')} ===")

        month_start = current_month
        month_end = (current_month + relativedelta(months=1)) - datetime.timedelta(days=1)
        date_range = pd.date_range(month_start, month_end, freq="B")

        for current_day in date_range:
            daily_data = {}
            for t in args.test_tickers:
                feats = sim_ticker_feats[t]
                if current_day not in feats.index:
                    continue
                daily_data[t] = feats.loc[current_day]

            # simulate one day
            TradeSimulator.simulate_one_day(
                current_day,
                daily_data,
                feat_cols=feat_cols,
                models=current_models,
                open_positions=open_positions,
                results=all_results,
                core_trade_state=core_trade_state,
                meta_model=meta_model,
                exit_meta_model=exit_meta_model,
                EXIT_FEATURE_COLS=EXIT_FEATURE_COLS
            )

            # SAFETY: if any trade got appended today multiple times, keep only first
            if all_results:
                seen = set()
                deduped = []
                for tr in all_results:
                    k = (tr["ticker"], pd.Timestamp(tr["entry_date"]))
                    if k in seen: 
                        continue
                    seen.add(k)
                    deduped.append(tr)
                all_results = deduped
                closed_keys |= seen  # everything in results is considered closed

            # remove closed from open_positions
            for k in list(open_positions.keys()):
                kk = (open_positions[k]["ticker"], pd.Timestamp(open_positions[k]["entry_date"]))
                if kk in closed_keys:
                    del open_positions[k]

        current_month += relativedelta(months=1)

    # ─── Finalize any positions still open (RUNS ONCE, after loop) ─────────────────
    final_day = end_month
    for ticker, trade in list(open_positions.items()):
        key = (ticker, pd.Timestamp(trade["entry_date"]))
        if key in closed_keys:
            del open_positions[ticker]
            continue

        last_price = trade.get("last_seen_price", trade["entry_price"])
        trade["exit_date"] = final_day
        trade["exit_price"] = last_price
        trade["exit_reason"] = "forced_timeout"
        trade["final_return"] = (last_price - trade["entry_price"]) / trade["entry_price"]

        price_df = core_trade_state.get(ticker, None)
        if price_df is not None and "High" in price_df.columns:
            exit_window = price_df.loc[trade["entry_date"]:final_day]
            highs = exit_window["High"].dropna()
            if not highs.empty:
                max_price = highs.max()
                peak_date = highs.idxmax()
                trade["max_possible_return"] = (max_price - trade["entry_price"]) / trade["entry_price"]
                trade["capture_ratio"] = (
                    trade["final_return"] / trade["max_possible_return"]
                    if trade["final_return"] > 0 and trade["max_possible_return"] > 0 else 0
                )
                trade["peak_date"] = peak_date
                trade["exit_efficiency_days"] = abs((pd.Timestamp(final_day) - peak_date).days)
            else:
                trade["max_possible_return"] = None
                trade["capture_ratio"] = None
                trade["peak_date"] = None
                trade["exit_efficiency_days"] = None
        else:
            trade["max_possible_return"] = None
            trade["capture_ratio"] = None
            trade["peak_date"] = None
            trade["exit_efficiency_days"] = None

        # meta labels
        trade["meta_label"] = 1 if trade["final_return"] > 0.20 else 0
        capture = trade.get("capture_ratio")
        trade["capture_label"] = 1 if capture is not None and capture >= 0.6 else 0

        # final exit features for diagnostics
        if price_df is not None:
            try:
                last_row = price_df.loc[:final_day].iloc[-1]
                history_window = price_df.loc[:final_day].tail(10)
                trade["exit_features"] = compute_exit_features(last_row, trade, history_window)
            except Exception:
                trade["exit_features"] = {}
        else:
            trade["exit_features"] = {}

        # ensure meta_candidates exists & append the forced exit as a candidate
        trade.setdefault("meta_candidates", []).append({
            "exit_date": final_day,
            "exit_price": trade["exit_price"],
            "features": trade["exit_features"]
        })

        # label the candidates for that trade (optional offline diagnostics)
        label_meta_candidates_ranking(trade)

        all_results.append(trade)
        closed_keys.add(key)
        del open_positions[ticker]

    # (optional) final dedupe by (ticker, entry_date)
    deduped = {}
    for tr in all_results:
        k = (tr["ticker"], pd.Timestamp(tr["entry_date"]))
        deduped.setdefault(k, tr)  # keep first
    all_results = list(deduped.values())




    # ─── End-of-Month Label Deferral ──────────────────────────────────
    # raw_feats_this_month = []

    # for t in training_tickers:
    #     raw = dm.download_stock_data(t)
    #     feats = dm.compute_features(raw)

    #     feats["ema50_slope"] = feats["ema50"].diff(3)
    #     feats["vix_5d_slope"] = feats["vix_close"].diff(5)
    #     feats["fear_greed_slope"] = feats["fear_greed"].diff(5)
    #     feats["macro_trend_ok"] = (
    #         (feats["price_vs_ema200"] > 0.01) &
    #         (feats["rsi"] > 50) &
    #         (feats["vix_5d_slope"] < 0)
    #     )

    #     # Save raw features for deferral (we'll label them later)
    #     month_feats = feats.loc[month_start.strftime("%Y-%m-%d"):month_end.strftime("%Y-%m-%d")].copy()
    #     month_feats["regime"] = month_feats["vix_close"].apply(
    #         lambda x: "high" if x > 30 else "caution" if x > 20 else "low"
    #     )

    #     raw_feats_this_month.append(month_feats)

    # # Defer labeling this month until 180 days later
    # pending_label_queue.append((month_start, raw_feats_this_month))
    # print(f"[⏳] Deferring labeling for {month_start.strftime('%Y-%m')} until 180 days later.")

    # # ─── Process Deferred Months That Are Now Labelable ───────────────────
    # newly_labeled = []
    # for pending_month, raw_dfs in list(pending_label_queue):
    #     if current_month >= pending_month + datetime.timedelta(days=180):
    #         print(f"[📅] Labeling deferred month: {pending_month.strftime('%Y-%m')}")
    #         labeled_dfs = []
    #         for df in raw_dfs:
    #             if df.empty or "regime" not in df.columns:
    #                 continue
    #             labeled = labeler.label_by_regime(df, {
    #                 "high": labeling.label_rsi_reversal_triple_barrier,
    #                 "caution": labeling.label_stage2_confirmed_triple_barrier,
    #                 "low": labeling.label_stage2_breakout_with_exit_logic
    #             })
    #             if not labeled.empty and "entry_date" in labeled.columns:
    #                 labeled["entry_month"] = labeled["entry_date"].dt.to_period("M")
    #                 labeled_dfs.append(labeled)
    #         if labeled_dfs:
    #             combined_labeled = pd.concat(labeled_dfs)
    #             rolling_train_df.append(combined_labeled)
    #             print(f"[📬] Added labeled data from {pending_month.strftime('%Y-%m')} to training.")
    #         newly_labeled.append((pending_month, raw_dfs))

    # # Remove processed months from the pending queue
    # for item in newly_labeled:
    #     pending_label_queue.remove(item)

    # # ─── Train models on current rolling window ─────────────────────────────
    # train_df = pd.concat(list(rolling_train_df)).dropna(subset=feat_cols)

    # # 👀 Class balance before oversampling
    # if "target" in train_df.columns:
    #     pre_dist = train_df["target"].value_counts(normalize=True).to_dict()
    #     print(f"[📊] Label distribution (pre-oversampling): {pre_dist}")

    # main_model = ModelTrainer(feat_cols).train(train_df)
    # dip_model = ModelTrainer(feat_cols).train(train_df[train_df["label_type"] == "high"])
    # caution_model = ModelTrainer(feat_cols).train(train_df[train_df["label_type"] == "caution"])

    # current_models = {
    #     "low": main_model,
    #     "caution": caution_model,
    #     "high": dip_model
    # }

    # print(f"[✅] Models retrained on {len(train_df)} rows.")

    
    # ─── Save and Evaluate ───────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        df["final_return"] = (df["exit_price"] - df["entry_price"]) / df["entry_price"]
        df.to_csv("trade_log.csv", index=False)
        Evaluator().summarize_trades(df)




if __name__ == "__main__":
    main()
