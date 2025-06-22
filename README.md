# leveraged-etf-dashboard-backend
USAGE FOR MULTIPLE STOCK VALIDATION:


python3 multiple-stock-validation.py --test_year 2022 \
--test_tickers AAPL MSFT GOOGL AMZN META NVDA BRK-B JPM V MA \
TSLA UNH HD PG LLY JNJ XOM CVX MRK PEP COST \
--profit_target 0.20 --stop_loss 0.08 \
--max_duration 60 --min_hold_days 20


Change tickers, profit target, stoploss duration and min hold days to desired values