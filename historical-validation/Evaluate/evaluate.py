import numpy as np
import pandas as pd
import yfinance as yf

class Evaluator:
    def __init__(self):
        pass

    def summarize_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        tickers = trades["ticker"].unique()
        all_summaries = []

        for t in tickers:
            sub = trades[trades["ticker"] == t].copy()
            entry_dates = sub["entry_date"]
            year = entry_dates.dt.year.mode()[0]

            start = f"{year}-01-01"
            end = f"{year}-12-31"
            hist = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)

            if "Close" in hist.columns and len(hist) >= 2:
                start_price = hist["Close"].iloc[0]
                end_price = hist["Close"].iloc[-1]
                ytd_return = (end_price - start_price) / start_price
            else:
                ytd_return = np.nan

            sub["win"] = sub["final_return"] > 0

            stats = {
                "ticker": t,
                "count": len(sub),
                "win_rate": sub["win"].mean(),
                "avg_return": sub["final_return"].mean(),
                "std_return": sub["final_return"].std(),
                "net_return": sub["final_return"].sum(),
                "ytd_return": ytd_return
            }

            all_summaries.append(stats)

        summary_df = pd.DataFrame(all_summaries).sort_values("win_rate", ascending=False)

        summary_df["ytd_return"] = summary_df["ytd_return"].apply(
            lambda x: float(x.iloc[0]) if isinstance(x, pd.Series) else float(x)
        )
        summary_df["net_return"] = summary_df["net_return"].astype(float)

        total_net = summary_df["net_return"].sum()
        total_ytd = summary_df["ytd_return"].sum()

        for col in ["avg_return", "net_return", "ytd_return"]:
            summary_df[col] = (summary_df[col] * 100).map("{:.2f}%".format)

        print("\n=== OVERALL SUMMARY ===")
        print(summary_df.to_string(index=False))
        print(f"\nTotal Net Return: {total_net * 100:.2f}%")
        print(f"Total YTD Return: {total_ytd * 100:.2f}%")

        return summary_df
