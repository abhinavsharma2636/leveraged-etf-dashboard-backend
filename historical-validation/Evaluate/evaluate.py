import numpy as np
import pandas as pd
import yfinance as yf

def get_buy_and_hold_return(ticker: str, start_date: str, end_date: str) -> float:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    if df.empty or "Close" not in df.columns or df["Close"].dropna().shape[0] < 2:
        raise ValueError(f"[❌] Not enough valid data for {ticker}")

    close_prices = df["Close"].dropna()
    start_price = close_prices.iloc[0].item()
    end_price = close_prices.iloc[-1].item()

    if start_price == 0.0:
        raise ValueError(f"[❌] Start price is 0 for {ticker}")

    return (end_price - start_price) / start_price


class Evaluator:
    def __init__(self):
        pass

    def summarize_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])

        tickers = trades["ticker"].unique()
        all_summaries = []

        for t in tickers:
            sub = trades[trades["ticker"] == t].copy()
            sub["win"] = sub["final_return"] > 0

            try:
                ytd_return = get_buy_and_hold_return(
                    t, "2020-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")
                )
            except Exception as e:
                print(f"[⚠️] Failed to fetch YTD return for {t}: {e}")
                ytd_return = np.nan

            avg_return = sub["final_return"].mean()
            std_return = sub["final_return"].std()

            # Calculate CAGR
            try:
                days = (sub["exit_date"].max() - sub["entry_date"].min()).days
                years = days / 365.25
                net_ret = (1 + sub["final_return"]).prod() - 1
                cagr = (1 + net_ret) ** (1 / years) - 1 if years > 0 else np.nan
            except Exception:
                cagr = np.nan

            # Calculate Sharpe Ratio
            sharpe = avg_return / std_return if std_return != 0 else np.nan

            stats = {
                "ticker": t,
                "count": len(sub),
                "win_rate": sub["win"].mean(),
                "avg_return": avg_return,
                "std_return": std_return,
                "net_return": sub["final_return"].sum(),
                "ytd_return": ytd_return,
                "cagr": cagr,
                "sharpe": sharpe
            }

            all_summaries.append(stats)

        summary_df = pd.DataFrame(all_summaries).sort_values("win_rate", ascending=False)

        # Preserve raw values for totals
        total_net = summary_df["net_return"].sum()
        total_ytd = summary_df["ytd_return"].sum()

        # Format percentages
        for col in ["avg_return", "net_return", "ytd_return", "cagr"]:
            summary_df[col] = (summary_df[col] * 100).map("{:.2f}%".format)

        # Format Sharpe ratio with 2 decimals
        summary_df["sharpe"] = summary_df["sharpe"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "nan")

        print("\n=== OVERALL SUMMARY ===")
        print(summary_df.to_string(index=False))
        print(f"\nTotal Net Return: {total_net * 100:.2f}%")
        print(f"Total YTD Return: {total_ytd * 100:.2f}%")
        overall_win_rate = trades["final_return"].gt(0).mean()
        print(f"Overall Win Rate: {overall_win_rate:.2%}")

        return summary_df
