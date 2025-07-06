# main.py
# Entry point for regime-aware ML trading pipeline

import argparse
import datetime
import pandas as pd

from Data_Manager.data import DataManager
from Labeler.labeling import Labeler
from Labeler.labeling_functions import LabelingFunctions
from Model.model import ModelTrainer
from Simulator.simulate import TradeSimulator

from Evaluate.evaluate import Evaluator

# ─── USER-CONFIGURED TICKERS ─────────────────────────────────────────────────────
training_tickers = [
    "AAPL", "MSFT", "INTC", "CSCO", "IBM", "ORCL",
    "TXN", "ADI", "JPM", "BAC", "WFC", "GS", "AXP",
    "PG", "KO", "PEP", "WMT", "COST", "GE", "CAT",
    "MMM", "HON", "JNJ", "PFE", "MRK", "ABBV", "UNH",
    "SPY", "QQQ", "DIA", "IWM", "XOM", "CVX", "DUK",
    "NEE", "HD", "LOW", "TGT", "MCD", "NKE", "SBUX"
]

LABEL_DIR = "Labeler/data_labeling"
def parse_args():
    p = argparse.ArgumentParser(description="High-Confidence Entry Detection with Extended Exit Rules")
    p.add_argument("--test_year", type=int, required=True)
    p.add_argument("--test_tickers", nargs="+", default=training_tickers)
    return p.parse_args()

def main():
    args = parse_args()
    yr = args.test_year

    feat_cols = [
        "atr", "vix_close", "macd_diff", "rsi", "volume_surge",
        "price_vs_ema50", "stoch_d",
        "volatility_regime_low", "volatility_regime_neutral", "volatility_regime_high"
    ]

    # Date ranges
    train_start = (datetime.date(yr, 1, 1) - datetime.timedelta(days=365 * 15)).isoformat()
    train_end   = f"{yr-1}-12-31"
    test_start  = f"{yr}-01-01"
    test_end    = f"{yr}-12-31"

    dm = DataManager(train_start, test_end)
    dm.download_all_macro()

    labeler = Labeler(label_dir=LABEL_DIR)
    labeling = LabelingFunctions()

    all_train = []
    dip_labeled = []
    caution_labeled = []

    print("Building training set…")
    for t in training_tickers:
        print(f"▶ Processing {t} for {yr}")
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

        train_feats = feats.loc[:train_end].copy()
        train_feats["regime"] = train_feats["vix_close"].apply(
            lambda x: "high" if x > 30 else "caution" if x > 20 else "low"
        )

        labeled = labeler.load_or_label(train_feats, t, yr, {
            "high": labeling.label_rsi_reversal_enhanced,
            "caution": labeling.label_stage2_confirmed_with_exit_logic,
            "low": labeling.label_stage2_breakout_with_exit_logic
        })
        all_train.append(labeled)

        # Pre-train dip/caution models
        high_vix = train_feats[train_feats["vix_close"] > 30]
        dip = labeler.load_or_label_subset(high_vix, labeling.label_rsi_reversal_enhanced, t, yr, "dip")
        dip_labeled.append(dip)

        caution_vix = train_feats[(train_feats["vix_close"] > 20) & (train_feats["vix_close"] <= 30)]
        caution = labeler.load_or_label_subset(caution_vix, labeling.label_stage2_confirmed_with_exit_logic, t, yr, "confirmed")
        caution_labeled.append(caution)

    train_df = pd.concat(all_train).dropna(subset=feat_cols)
    dip_df = pd.concat(dip_labeled).dropna(subset=feat_cols)
    caution_df = pd.concat(caution_labeled).dropna(subset=feat_cols)

    # Upsample dip cases
    dip_cases = train_df[
        (train_df["rsi"] < 35) &
        (train_df["price_vs_ema50"] < -0.05) &
        (train_df["vix_close"] > 25)
    ]
    train_df = pd.concat([train_df, dip_cases, dip_cases])

    print(f"Training main model on {len(train_df)} rows…")
    main_model = ModelTrainer(feat_cols).train(train_df)

    print(f"Training dip model on {len(dip_df)} rows…")
    dip_model = ModelTrainer(feat_cols).train(dip_df)

    print(f"Training caution model on {len(caution_df)} rows…")
    caution_model = ModelTrainer(feat_cols).train(caution_df)

    models = {
        "low": main_model,
        "caution": caution_model,
        "high": dip_model
    }

    all_entries = []
    for t in args.test_tickers:
        print(f"\n=== TESTING {t} on {yr} ===")
        raw = dm.download_stock_data(t)
        feats = dm.compute_features(raw)
        test_feats = feats.loc[test_start:].copy()

        simulator = TradeSimulator(feat_cols=feat_cols, models=models)
        entries = simulator.simulate(test_feats)

        if not entries.empty:
            entries["ticker"] = t
            all_entries.append(entries)
        else:
            print(f"{t}: no trades made")

    if all_entries:
        results = pd.concat(all_entries).reset_index(drop=True)
        results.to_csv("trade_log.csv", index=False)
        Evaluator().summarize_trades(results)

if __name__ == "__main__":
    main()
