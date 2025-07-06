import pandas as pd
from xgboost import XGBClassifier
from typing import List

class ModelTrainer:
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols

    def train(self, df: pd.DataFrame) -> XGBClassifier:
        X = df[self.feature_cols]
        y = df["target"]
        model = XGBClassifier(
            objective="binary:logistic",
            base_score=0.5,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X, y)
        return model
