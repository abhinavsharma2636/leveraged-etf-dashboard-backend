import pandas as pd
from xgboost import XGBClassifier
from typing import List
import lightgbm as lgb


class ModelTrainer:
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols

    def train(self, df: pd.DataFrame) -> XGBClassifier:
        X = df[self.feature_cols]
        y = df["target"]

        pos = (y == 1).sum()
        neg = (y == 0).sum()
        scale_pos_weight = neg / max(pos, 1)

        print("[📊] Training core model. Class balance:", y.value_counts(normalize=True).to_dict())

        model = XGBClassifier(
            objective="binary:logistic",
            base_score=0.5,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        model.fit(X, y)
        return model

    def train_meta(self, df: pd.DataFrame) -> XGBClassifier:
        X = df[self.feature_cols]
        y = df["meta_label"]

        pos = (y == 1).sum()
        neg = (y == 0).sum()
        scale_pos_weight = neg / max(pos, 1)

        print("[📊] Training meta model. Class balance:", y.value_counts(normalize=True).to_dict())

        model = XGBClassifier(
                objective="binary:logistic",
                base_score=0.5,
                n_estimators=200,
                max_depth=3,                # was 4
                learning_rate=0.1,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                reg_alpha=1.0,              # L1 regularization
                reg_lambda=1.0,             # L2 regularization
                random_state=42
            )

        model.fit(X, y)
        return model
    

    def train_exit_meta(self, df: pd.DataFrame) -> XGBClassifier:
        X = df[self.feature_cols].copy().astype(float)
        y = df["label"].copy()

        # === Drop rows with any NaNs in features
        nan_rows = X.isnull().any(axis=1)
        if nan_rows.sum() > 0:
            print(f"[⚠️] Dropping {nan_rows.sum()} rows with NaNs from X before training")
            X = X[~nan_rows]
            y = y.loc[X.index]

        # === Handle class imbalance
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        scale_pos_weight = neg / max(pos, 1)

        print("[📊] Training exit meta model. Class balance:", y.value_counts(normalize=True).to_dict())

        model = XGBClassifier(
            objective="binary:logistic",
            base_score=y.mean(),  # use actual class prior
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            use_label_encoder=False
        )

        model.fit(X, y)

        return model
    

    def train_exit_ranker(self, df: pd.DataFrame) -> lgb.LGBMRanker:
        """
        Train LightGBM lambdarank on graded labels.
        Preserves real feature names and persists the exact list for inference.
        """
        import numpy as np
        import lightgbm as lgb
        import json

        df = df.copy()

        # choose label: prefer 'rel' else 'target'
        if "rel" in df.columns:
            df["label"] = df["rel"].fillna(0).astype(int).clip(lower=0)
        elif "target" in df.columns:
            df["label"] = df["target"].fillna(0).astype(int).clip(0, 1)
        else:
            raise ValueError("Need 'rel' or 'target' in df")

        # drop groups with <2
        gsz = df.groupby("group_id").size()
        keep_groups = gsz[gsz >= 2].index
        df = df[df["group_id"].isin(keep_groups)].copy()

        # keep only available feature columns (and remember them)
        avail = [c for c in self.feature_cols if c in df.columns]
        if len(avail) < len(self.feature_cols):
            missing = [c for c in self.feature_cols if c not in df.columns]
            print(f"[warn] dropping {len(missing)} missing features: {missing[:10]}{'...' if len(missing)>10 else ''}")
        self.feature_cols = avail

        # sort by group to align group sizes with X order
        df = df.dropna(subset=self.feature_cols + ["label", "group_id"]) \
            .sort_values(["group_id"], kind="mergesort")

        X = df[self.feature_cols].astype(float)           # DATAFRAME (preserves names)
        y = df["label"].astype(int)
        groups = df.groupby("group_id").size().to_list()

        # simple block split (keep it, or use a time split if you have one)
        uniq = df["group_id"].unique()
        cut = int(0.8 * len(uniq))
        trn_ids = set(uniq[:cut])
        val_ids = set(uniq[cut:])

        trn = df[df["group_id"].isin(trn_ids)]
        val = df[df["group_id"].isin(val_ids)]

        X_trn, y_trn = trn[self.feature_cols].astype(float), trn["label"].astype(int)
        X_val, y_val = val[self.feature_cols].astype(float), val["label"].astype(int)

        grp_trn = trn.groupby("group_id").size().to_list()
        grp_val = val.groupby("group_id").size().to_list()

        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            min_data_in_leaf=50,
            min_child_samples=50,   # <-- add this to match; no warning even if both are set
            lambda_l2=1.0,
            random_state=42,
        )


        model.fit(
            X_trn, y_trn,
            group=grp_trn,
            eval_set=[(X_val, y_val)],
            eval_group=[grp_val],
            eval_at=[1, 3, 5],
            callbacks=[lgb.early_stopping(150, verbose=True)],
        )

        # prune zero-gain and refit (optional)
        booster = model.booster_
        gains = booster.feature_importance(importance_type="gain")
        zero_gain = [f for f, g in zip(self.feature_cols, gains) if g == 0]
        if zero_gain:
            print(f"[⚠️] Dropping {len(zero_gain)} zero-gain features:", zero_gain)
            self.feature_cols = [f for f in self.feature_cols if f not in zero_gain]

            X_trn = trn[self.feature_cols].astype(float)
            X_val = val[self.feature_cols].astype(float)

            model = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=3000,
                learning_rate=0.03,
                max_depth=6,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                min_data_in_leaf=50,
                min_child_samples=50,   # <-- add this to match; no warning even if both are set
                lambda_l2=1.0,
                random_state=42,
            )

            model.fit(
                X_trn, y_trn,
                group=grp_trn,
                eval_set=[(X_val, y_val)],
                eval_group=[grp_val],
                eval_at=[1, 3, 5],
                callbacks=[lgb.early_stopping(150, verbose=True)],
            )

        # persist exact feature list
        with open("exit_meta_features.json", "w") as f:
            json.dump(self.feature_cols, f)

        return model


