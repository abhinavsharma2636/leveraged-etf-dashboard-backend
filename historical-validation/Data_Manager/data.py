import yfinance as yf
import pandas as pd
import numpy as np
import ta
import requests

class DataManager:
    def __init__(self, start: str, end: str):
        self.start = start
        self.end = end
        self.vix = None
        self.fear = None

    def download_all_macro(self):
        self.vix = self.get_vix()
        self.fear = self.get_fear_greed()

    def get_vix(self) -> pd.Series:
        v = yf.download("^VIX", start=self.start, end=self.end, progress=False, auto_adjust=False)
        if isinstance(v.columns, pd.MultiIndex):
            v.columns = v.columns.get_level_values(0)
        s = v["Close"].copy()
        s.name = "vix_close"
        return s


    def get_fear_greed(self):
        url = "https://api.alternative.me/fng/?limit=0"
        data = requests.get(url).json()["data"]
        rows = [{
            "date": pd.to_datetime(int(d["timestamp"]), unit="s"),
            "fear_greed": int(d["value"])
        } for d in data]
        fg = pd.DataFrame(rows)
        return fg.set_index("date").sort_index()

    def download_stock_data(self, ticker: str) -> pd.DataFrame:
        df = yf.download(ticker, start=self.start, end=self.end, interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.rename(columns=str.capitalize, inplace=True)
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        bb = ta.volatility.BollingerBands(data.Close, window=20, window_dev=2)
        data["bb_upper"] = bb.bollinger_hband()
        data["bb_lower"] = bb.bollinger_lband()
        data["bb_pct"] = (data.Close - data.bb_lower) / (data.bb_upper - data.bb_lower + 1e-9)

        for span in (20, 50, 200):
            data[f"ema{span}"] = data.Close.ewm(span=span, adjust=False).mean()

        data["price_vs_ema20"] = data.Close / data.ema20 - 1
        data["price_vs_ema50"] = data.Close / data.ema50 - 1
        data["price_vs_ema200"] = data.Close / data.ema200 - 1
        data["atr"] = ta.volatility.average_true_range(data.High, data.Low, data.Close, window=14)
        data["rsi"] = ta.momentum.rsi(data.Close, window=14)

        macd = ta.trend.MACD(data.Close)
        data["macd_diff"] = macd.macd_diff()

        stoch = ta.momentum.StochasticOscillator(data.High, data.Low, data.Close, window=14, smooth_window=3)
        data["stoch_k"], data["stoch_d"] = stoch.stoch(), stoch.stoch_signal()

        data["vol_avg5"] = data.Volume.rolling(5).mean()
        data["volume_surge"] = (data.Volume / data.vol_avg5 - 1).clip(lower=0)
        data["trend_strength"] = data["price_vs_ema50"] + data["price_vs_ema200"]

        # RSI2, VIX RSI, 1-year mean
        data["rsi2"] = ta.momentum.rsi(data["Close"], window=2)
        data["price_rolling_mean_252"] = data["Close"].rolling(window=252, min_periods=100).mean()

        # Candle patterns
        data = self.add_candle_patterns(data)

        # Join VIX/Fear & Greed
        data = data.join(self.vix, how="left").ffill()
        data = data.join(self.fear, how="left").ffill()

        data["vix_rsi2"] = ta.momentum.rsi(data["vix_close"], window=2)


        # Volatility Regime
        vix_p20 = np.percentile(data["vix_close"], 20)
        vix_p80 = np.percentile(data["vix_close"], 80)
        data["volatility_regime"] = pd.cut(
            data["vix_close"],
            bins=[-np.inf, vix_p20, vix_p80, np.inf],
            labels=["low", "neutral", "high"]
        )

        # One-hot encode regimes
        regime_dummies = pd.get_dummies(data["volatility_regime"], prefix="volatility_regime")
        data = pd.concat([data, regime_dummies], axis=1)

        # Drop rows with missing key features
        required_feats = [
            "Close", "vix_close", "atr", "rsi",
            "macd_diff", "stoch_d", "price_vs_ema50", "volume_surge"
        ]
        return data.dropna(subset=required_feats)

    def add_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        body = (df["Close"] - df["Open"]).abs()
        upper_wick = df["High"] - df[["Close", "Open"]].max(axis=1)
        lower_wick = df[["Close", "Open"]].min(axis=1) - df["Low"]
        df["is_hammer"] = ((body < (df["High"] - df["Low"]) * 0.3) & (lower_wick > body * 2) & (upper_wick < body)).astype(int)
        prev_open = df["Open"].shift(1)
        prev_close = df["Close"].shift(1)
        df["is_bullish_engulfing"] = ((prev_close < prev_open) & (df["Close"] > df["Open"]) & (df["Close"] > prev_open) & (df["Open"] < prev_close)).astype(int)
        return df
