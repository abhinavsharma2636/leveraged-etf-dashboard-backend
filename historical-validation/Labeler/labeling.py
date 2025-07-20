
import os
import pandas as pd
from typing import Callable


class Labeler:
    def __init__(self, label_dir: str):
        self.label_dir = label_dir
        os.makedirs(label_dir, exist_ok=True)

    def load_or_label_monthly(self, month_feats: pd.DataFrame, ticker: str, year: int, month: int, label_funcs: dict) -> pd.DataFrame:
        year_dir = os.path.join(self.label_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
        path = os.path.join(year_dir, f"{ticker}_{year}_{month:02d}.parquet")

        # 🛑 Skip labeling if file already exists
        if os.path.exists(path):
            return pd.read_parquet(path, engine="pyarrow")

        # ─── Assign regime for this month ───
        vix_median = month_feats["vix_close"].median()
        if vix_median > 30:
            regime = "high"
        elif vix_median > 20:
            regime = "caution"
        else:
            regime = "low"

        # ─── Apply the appropriate labeler ───
        if regime in label_funcs:
            labeled = label_funcs[regime](month_feats.copy())
            if labeled is not None and not labeled.empty:
                labeled["label_type"] = regime
                labeled["entry_month"] = labeled["entry_date"].dt.to_period("M")

                # ✅ Only save if not empty
                labeled.to_parquet(path, engine="pyarrow", index=False)
                return labeled

        # 🟡 Nothing labeled — return empty DataFrame
        return pd.DataFrame()

    

    def load_or_label_subset(self, df: pd.DataFrame, label_func: Callable, ticker: str, year: int, tag: str) -> pd.DataFrame:
        year_dir = os.path.join(self.label_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
        path = os.path.join(year_dir, f"{ticker}_{year}_{tag}.parquet")

        if os.path.exists(path):
            print(f"[✔] Loaded cached labels for {ticker} {year} ({tag})")
            return pd.read_parquet(path)

        labeled = label_func(df)
        labeled.to_parquet(path, index=False)
        print(f"[💾] Saved labels for {ticker} {year} ({tag}) → {path}")
        return labeled
    
    def label_by_regime(self, df: pd.DataFrame, label_funcs: dict) -> pd.DataFrame:
        labeled_parts = []
        for regime, sub in df.groupby("regime"):
            if regime in label_funcs:
                labeled = label_funcs[regime](sub)
                if labeled.empty:
                    print(f"[⚠️] {regime} labeling returned 0 rows.")
                else:
                    labeled["label_type"] = regime
                    labeled_parts.append(labeled)

        if not labeled_parts:
            return pd.DataFrame()
        
        return pd.concat(labeled_parts)

