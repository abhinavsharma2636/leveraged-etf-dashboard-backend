
import os
import pandas as pd
from typing import Callable

class Labeler:
    def __init__(self, label_dir: str):
        self.label_dir = label_dir
        os.makedirs(label_dir, exist_ok=True)

    def load_or_label(self, train_feats: pd.DataFrame, ticker: str, year: int, label_funcs: dict) -> pd.DataFrame:
        year_dir = os.path.join(self.label_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
        path = os.path.join(year_dir, f"{ticker}_{year}_all.parquet")

        if os.path.exists(path):
            print(f"[✔] Loaded cached labels for {ticker} {year}")
            return pd.read_parquet(path)

        labeled_parts = []
        for regime, sub in train_feats.groupby("regime"):
            if regime in label_funcs:
                labeled = label_funcs[regime](sub)
                labeled["label_type"] = regime
                labeled_parts.append(labeled)

        lbl = pd.concat(labeled_parts)
        lbl.to_parquet(path, index=False)
        print(f"[💾] Saved labels for {ticker} {year} → {path}")
        return lbl

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
