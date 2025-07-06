import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta

# Load trade log
df = pd.read_csv("trade_log.csv", parse_dates=["entry_date", "exit_date"])
ticker = "SPY"
sub = df[df["ticker"] == ticker].copy()


if ticker not in df["ticker"].unique():
    raise ValueError(f"Ticker '{ticker}' not found in trade_log.csv.")

# Download and clean price data with indicators
def get_price_data_with_indicators(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.rename(columns=str.capitalize, inplace=True)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)

    # Add RSI
    rsi = ta.momentum.RSIIndicator(close=df["Close"].astype(float), window=14)
    df["rsi"] = rsi.rsi().astype(float).squeeze()

    # Add Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband().squeeze()
    df["bb_lower"] = bb.bollinger_lband().squeeze()

    return df

def get_vix_data(start: str, end: str) -> pd.Series:
    vix = yf.download("^VIX", start=start, end=end, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix[["Close"]].rename(columns={"Close": "vix_close"})
    vix.index = pd.to_datetime(vix.index)
    return vix["vix_close"]

# Prepare date range
# Detect the year from the first entry in trade log
year = sub["entry_date"].dt.year.min()

# Always chart full year range
start = pd.Timestamp(f"{year}-01-01")
end   = pd.Timestamp(f"{year}-12-31")


# Get price + indicators and VIX
price_data = get_price_data_with_indicators(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
vix = get_vix_data(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
price_data = price_data.join(vix, how="left").ffill()

# ─── Plotting with shared X-axis ───────────────────────────
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

# Price with BB and trades
axs[0].plot(price_data["Close"], label=f"{ticker} Close", color="black")
axs[0].plot(price_data["bb_upper"], linestyle="--", color="gray", alpha=0.5, label="BB Upper")
axs[0].plot(price_data["bb_lower"], linestyle="--", color="gray", alpha=0.5, label="BB Lower")

for i, row in sub.iterrows():
    entry = price_data.index[price_data.index.get_indexer([row["entry_date"]], method="ffill")[0]]
    exit_  = price_data.index[price_data.index.get_indexer([row["exit_date"]], method="ffill")[0]]
    entry_px = price_data.loc[entry, "Close"]
    exit_px  = price_data.loc[exit_, "Close"]
        # Entry marker
    axs[0].scatter(entry, entry_px, color="green", marker="^", s=100, label="Entry" if i == 0 else "")
    # Exit marker
    axs[0].scatter(exit_, exit_px, color="red", marker="v", s=100, label="Exit" if i == 0 else "")


axs[0].set_title(f"{ticker} Price with Trades and Bollinger Bands")
axs[0].set_ylabel("Price")
axs[0].legend()
axs[0].grid(True)

# RSI plot
axs[1].plot(price_data["rsi"], label="RSI", color="purple")
axs[1].axhline(70, color="red", linestyle="--", alpha=0.4)
axs[1].axhline(30, color="green", linestyle="--", alpha=0.4)
axs[1].set_ylabel("RSI")
axs[1].set_title("RSI")
axs[1].grid(True)

# VIX plot
axs[2].plot(price_data["vix_close"], label="VIX", color="orange")
axs[2].set_ylabel("VIX")
axs[2].set_title("VIX")
axs[2].grid(True)

# Format and show
plt.xlabel("Date")
plt.tight_layout()
plt.show()
