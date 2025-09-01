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
        trades = trades.copy()
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])

        tickers = trades["ticker"].unique()
        all_summaries = []

        for t in tickers:
            sub = trades[trades["ticker"] == t].copy()
            sub["win"] = sub["final_return"] > 0

            # --- core stats on per-trade returns ---
            avg_return = sub["final_return"].mean()
            std_return = sub["final_return"].std(ddof=1)
            # Compound per ticker (not sum)
            gross_mult = float((1.0 + sub["final_return"]).prod())
            net_ret = gross_mult - 1.0

            # CAGR over span from first entry to last exit
            try:
                days = (sub["exit_date"].max() - sub["entry_date"].min()).days
                years = days / 365.25
                cagr = (gross_mult ** (1 / years) - 1.0) if years > 0 else np.nan
            except Exception:
                cagr = np.nan

            sharpe = (avg_return / std_return) if (std_return and std_return > 0) else np.nan

            # Sortino
            downside = sub.loc[sub["final_return"] < 0, "final_return"]
            d_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
            sortino = (avg_return / d_std) if d_std and d_std > 0 else np.nan

            # --- Max drawdown from the compounded equity curve (per ticker) ---
            if len(sub) > 0:
                equity = (1.0 + sub["final_return"]).cumprod()
                roll_max = equity.cummax()
                dd_series = (equity / roll_max) - 1.0              # <= 0
                max_dd_frac = float(dd_series.min()) if len(dd_series) else np.nan
                max_dd = abs(max_dd_frac) if pd.notnull(max_dd_frac) else np.nan  # store as positive magnitude
            else:
                max_dd = np.nan

            mar = (cagr / max_dd) if (pd.notnull(cagr) and pd.notnull(max_dd) and max_dd > 0) else np.nan

            # Expectancy & Payoff
            expectancy = avg_return
            wins = sub.loc[sub["win"], "final_return"]
            losses = sub.loc[~sub["win"], "final_return"]
            payoff_ratio = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and abs(losses.mean()) > 0 else np.nan

            # Capture / exit efficiency safeguards
            capture_ratio = sub["capture_ratio"].mean() if "capture_ratio" in sub.columns else np.nan
            avg_exit_delta = sub["exit_efficiency_days"].mean() if "exit_efficiency_days" in sub.columns else np.nan

            summary = {
                "ticker": t,
                "trades": len(sub),                 # <= NEW
                "net_return": net_ret,             # compounded per ticker
                "win_rate": sub["win"].mean(),
                "sharpe": sharpe,
                "sortino": sortino,
                "cagr": cagr,
                "max_drawdown": max_dd,            # positive magnitude (e.g., 0.23 = 23%)
                "mar": mar,
                "expectancy": expectancy,
                "payoff_ratio": payoff_ratio,
                "capture_ratio": capture_ratio,
                "avg_exit_delta": avg_exit_delta
            }
            all_summaries.append(summary)

        summary_df = pd.DataFrame(all_summaries).sort_values(["win_rate", "trades"], ascending=False)

        # === Overall / portfolio-style stats (trade-level) ===
        # NOTE: Without a daily equity curve we can’t compute true time-weighted portfolio P&L.
        # These trade-level stats are stable and comparable run-to-run:
        # - arithmetic averages (expectancy components)
        # - geometric per-trade return (compounds 'as-if' sequential)
        all_final = trades["final_return"].astype(float)
        wins_mask = all_final > 0
        losses_mask = ~wins_mask

        total_trades = int(len(trades))
        wins = int(wins_mask.sum())
        losses = total_trades - wins
        win_rate_overall = wins / total_trades if total_trades else float("nan")

        avg_win = float(all_final[wins_mask].mean()) if wins else float("nan")
        avg_loss = float(all_final[losses_mask].mean()) if losses else float("nan")
        expectancy_overall = (win_rate_overall * (avg_win if pd.notnull(avg_win) else 0.0)) + \
                             ((1 - win_rate_overall) * (avg_loss if pd.notnull(avg_loss) else 0.0))

        # Geometric per-trade return (compounds each trade’s return; overlap-agnostic summary)
        # Interpreted as “if you ran trades sequentially back-to-back with full capital”
        with np.errstate(invalid="ignore"):
            gross_mult_all = float((1.0 + all_final).prod()) if total_trades else float("nan")
        geo_per_trade = (gross_mult_all ** (1.0 / total_trades) - 1.0) if total_trades > 0 else float("nan")

        # Massive loss rate (tail risk snapshot)
        massive_thresh = -0.20
        massive_loss_rate = float((all_final <= massive_thresh).mean()) if total_trades else float("nan")

        # Keep your per-ticker compounded numbers table, BUT don’t market-sum as a portfolio return
        total_net_sum_of_tickers = float(summary_df["net_return"].sum())

        # --- formatting for the table (unchanged) ---
        percentage_cols = ["net_return", "win_rate", "cagr"]
        ratio_cols = ["sharpe", "sortino", "mar", "expectancy", "payoff_ratio", "capture_ratio"]
        integer_cols = ["trades", "avg_exit_delta"]

        for col in percentage_cols + ratio_cols + integer_cols:
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

        for col in percentage_cols:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "N/A")

        for col in ratio_cols:
            summary_df[col] = summary_df[col].apply(lambda x: ("—" if pd.isna(x) else f"{x:.2f}"))

        for col in integer_cols:
            summary_df[col] = summary_df[col].apply(
                lambda x: (f"{int(x)}" if pd.notnull(x) and col == "trades" else (f"{x:.1f}" if pd.notnull(x) else "N/A"))
            )

        # === Prints ===
                # === Prints ===
        print("\n=== OVERALL SUMMARY (per-ticker table; net_return is compounded per ticker) ===")
        print(summary_df.to_string(index=False))

        # =========================
        # Portfolio-level dashboard
        # =========================
        # Trade-level series
        all_final = trades["final_return"].astype(float)
        all_final = all_final[pd.notnull(all_final)]
        total_trades = int(len(all_final))
        wins_mask = all_final > 0
        losses_mask = ~wins_mask

        wins = int(wins_mask.sum())
        losses = total_trades - wins
        win_rate_micro = wins / total_trades if total_trades else float("nan")

        # Macro (context only): avg win rate across tickers (each ticker equal weight)
        per_ticker_wr = trades.groupby("ticker")["final_return"].apply(lambda s: (s > 0).mean())
        win_rate_macro = float(per_ticker_wr.mean()) if len(per_ticker_wr) else float("nan")

        avg_win = float(all_final[wins_mask].mean()) if wins else float("nan")
        avg_loss = float(all_final[losses_mask].mean()) if losses else float("nan")
        payoff_ratio = (avg_win / abs(avg_loss)) if (pd.notnull(avg_win) and pd.notnull(avg_loss) and abs(avg_loss) > 0) else float("nan")
        expectancy_overall = (win_rate_micro * (avg_win if pd.notnull(avg_win) else 0.0)) + \
                             ((1 - win_rate_micro) * (avg_loss if pd.notnull(avg_loss) else 0.0))

        # Geometric per-trade return (stable vs outliers)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_returns = np.log1p(all_final.clip(lower=-0.999999))
            geo_per_trade = float(np.exp(log_returns.mean()) - 1.0) if total_trades else float("nan")

        # Time span → trades per year → CAGR est from per-trade geo
        try:
            start_date = pd.to_datetime(trades["entry_date"]).min()
            end_date   = pd.to_datetime(trades["exit_date"]).max()
            days_span  = (end_date - start_date).days if pd.notnull(end_date) and pd.notnull(start_date) else None
            years_span = (days_span / 365.25) if days_span and days_span > 0 else None
            trades_per_year = (total_trades / years_span) if years_span and years_span > 0 else None
            cagr_est = ((1.0 + geo_per_trade) ** trades_per_year - 1.0) if (trades_per_year and trades_per_year > 0 and pd.notnull(geo_per_trade)) else float("nan")
        except Exception:
            trades_per_year, cagr_est = float("nan"), float("nan")

        # Equity curve & max drawdown (trade-level, sequential)
        equity = (1.0 + all_final).cumprod() if total_trades else pd.Series(dtype=float)
        if len(equity):
            roll_max = equity.cummax()
            dd_series = (equity / roll_max) - 1.0
            max_dd = float(dd_series.min())  # negative number
            max_drawdown_mag = abs(max_dd)
        else:
            max_drawdown_mag = float("nan")

        # Tail risk
        massive_thresh = -0.20
        massive_loss_rate = float((all_final <= massive_thresh).mean()) if total_trades else float("nan")

        # Loss variance (only losing trades)
        if losses:
            loss_var = float(all_final[losses_mask].var(ddof=1)) if losses > 1 else 0.0
        else:
            loss_var = float("nan")

        # Sortino (downside deviation on negative returns only)
        downside = all_final[losses_mask]
        d_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        sortino = (expectancy_overall / d_std) if (d_std and d_std > 0) else float("nan")

        # Efficiency (optional columns)
        capture_ratio = trades["capture_ratio"].astype(float).mean() if "capture_ratio" in trades.columns else float("nan")
        avg_exit_delta = trades["exit_efficiency_days"].astype(float).mean() if "exit_efficiency_days" in trades.columns else float("nan")

        # Activity
        trades_per_month = (total_trades / (days_span / 30.44)) if (days_span and days_span > 0) else float("nan")

        # Legacy info: sum of per-ticker compounded returns (not a portfolio return)
        total_net_sum_of_tickers = float(pd.to_numeric(summary_df["net_return"].str.rstrip("%"), errors="coerce").fillna(0).sum()) / 100.0 \
                                   if summary_df["net_return"].dtype == object else float(summary_df["net_return"].sum())

        # ===== Structured prints =====
        print("\n=== PORTFOLIO (EDGE) ===")
        print(f"Trades: {total_trades} | Wins: {wins} | Losses: {losses}")
        print(f"Win rate (micro): {win_rate_micro:.2%}   | Win rate (macro, avg per ticker): {win_rate_macro:.2%}")
        print(f"Avg win: {avg_win*100:.2f}%   | Avg loss: {avg_loss*100:.2f}%   | Payoff ratio: {payoff_ratio:.2f}")
        print(f"Expectancy per trade: {expectancy_overall*100:.2f}%   | Geometric per trade: {geo_per_trade*100:.2f}%")

        print("\n=== RISK ===")
        print(f"Massive loss rate (≤ -20%): {massive_loss_rate:.2%}   | Loss variance: {loss_var:.4f}")
        print(f"Sortino (downside-only): {('—' if pd.isna(sortino) else f'{sortino:.2f}')}")

        print("\n=== EFFICIENCY ===")
        print(f"Capture ratio (avg): {('N/A' if pd.isna(capture_ratio) else f'{capture_ratio:.2f}')}"
              f"   | Avg exit delta (days): {('N/A' if pd.isna(avg_exit_delta) else f'{avg_exit_delta:.1f}')}"
              f"   | Trades/month: {('N/A' if pd.isna(trades_per_month) else f'{trades_per_month:.1f}')}")

        print(f"\n[Info] Sum of per-ticker net returns (not a portfolio return): {total_net_sum_of_tickers*100:.2f}%")

        return summary_df

