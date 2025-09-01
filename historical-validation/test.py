import datetime as dt, pandas as pd, yfinance as yf

tickers = """
AAPL MSFT NVDA TSLA META AMZN GOOGL LLY UNH JNJ ABBV MRK PFE
XOM CVX COP SLB OXY JPM BAC WFC GS MS V MA AXP
TSM ASML AMD AVGO QCOM INTC AMAT LRCX KLAC ADI MU TXN
CRM ADBE ORCL NOW COST WMT TGT HD LOW NKE MCD SBUX PEP KO PG
DIS NFLX LMT NOC RTX BA GE HON CAT DE UPS FDX UNP CSX NSC
LIN APD FCX NUE AMT PLD O SPG NEE DUK SO
TMO DHR ABT MDT AMGN GILD REGN EOG PXD DVN ADP
BKNG UBER PYPL SHOP
""".split()

cutoff = pd.Timestamp("2020-01-01")

good, bad = [], []
for t in tickers:
    try:
        df = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=False)
        if df.empty:
            bad.append((t, "EMPTY"))
            continue
        idx = pd.to_datetime(df.index)
        try:
            first = idx.tz_localize(None).min()
        except TypeError:
            first = idx.min()
        if pd.isna(first) or first > cutoff:
            bad.append((t, str(first)))
        else:
            good.append(t)
    except Exception as e:
        bad.append((t, f"ERR: {e}"))

print("OK:", " ".join(good))
print("\nFiltered out:", bad)
