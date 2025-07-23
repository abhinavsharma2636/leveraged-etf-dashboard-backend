import pandas as pd
from xgboost import XGBClassifier
from typing import List

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
