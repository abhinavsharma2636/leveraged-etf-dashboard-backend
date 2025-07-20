# main.py
# Entry point for regime-aware ML trading pipeline

import argparse
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from sklearn.utils import resample
from Data_Manager.data import DataManager
from Labeler.labeling import Labeler
from Labeler.labeling_functions import LabelingFunctions
from Model.model import ModelTrainer
from Simulator.simulate import TradeSimulator

from Evaluate.evaluate import Evaluator
from collections import deque

from Simulator.simulate_helpers import check_exit_today

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
    "GE", "X", "FSLR", "F", "BBY", "WU", "TAP", "ET"
]



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

    dm = DataManager(data_start_date, data_end_date)
    dm.download_all_macro()

    labeler = Labeler(label_dir=LABEL_DIR)
    labeling = LabelingFunctions()
    rolling_train_df = deque(maxlen=180)


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

    # ─── Monthly Live Simulation ─────────────────────────────────────
    current_month = start_date
    all_results = []
    open_positions = {}
    core_trade_state = {t: None for t in args.test_tickers}

    while current_month <= end_month:
        print(f"\n=== SIMULATING {current_month.strftime('%Y-%m')} ===")

        month_start = current_month
        month_end = (current_month + relativedelta(months=1)) - datetime.timedelta(days=1)
        date_range = pd.date_range(month_start, month_end, freq="B")

        for current_day in date_range:
            daily_data = {}

            for t in args.test_tickers:
                raw = dm.download_stock_data(t)
                feats = dm.compute_features(raw)

                feats["ema50_slope"] = feats["ema50"].diff(3)
                feats["vix_5d_slope"] = feats["vix_close"].diff(5)
                feats["fear_greed_slope"] = feats["fear_greed"].diff(5)
                feats["macro_trend_ok"] = (
                    (feats["price_vs_ema200"] > 0.01) &
                    (feats["rsi"] > 50) &
                    (feats["vix_5d_slope"] < 0)
                )

                if current_day not in feats.index:
                    continue

                row = feats.loc[current_day]
                daily_data[t] = row

            # Run full trade simulation with entry + exit logic
            TradeSimulator.simulate_one_day(
                current_day,
                daily_data,
                feat_cols,
                current_models,
                open_positions=open_positions,     # ✅ persistent positions
                results=all_results,               # ✅ completed trades
                core_trade_state=core_trade_state
            )

        current_month += relativedelta(months=1)

    # ─── Finalize any open positions at the end of simulation ───
    final_day = end_month
    for ticker, trade in open_positions.items():
        trade["exit_date"] = final_day
        last_price = trade.get("last_seen_price", trade["entry_price"])
        trade["exit_price"] = last_price
        trade["exit_reason"] = "forced_timeout"
        all_results.append(trade)



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
