import yfinance as yf
import pandas as pd

def get_buy_and_hold_return(ticker: str, start_date: str, end_date: str) -> float:
    # Download price data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    if df.empty or "Close" not in df.columns or df["Close"].dropna().shape[0] < 2:
        raise ValueError(f"[❌] Not enough valid data for {ticker}")

    # Clean and safely extract float values
    close_prices = df["Close"].dropna()

    start_price = close_prices.iloc[0].item()  # .item() extracts scalar from single-element Series
    end_price = close_prices.iloc[-1].item()

    if start_price == 0.0:
        raise ValueError(f"[❌] Start price is 0 for {ticker}")

    return (end_price - start_price) / start_price

# ✅ Example
if __name__ == "__main__":
    ret = get_buy_and_hold_return("LLY", "2020-01-01", "2025-07-21")
    print(f"AAPL Buy-and-Hold Return (2020–Today): {ret:.2%}")
