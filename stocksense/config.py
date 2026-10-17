"""Business assumptions, kept in one place so they are easy to see and change.

The transaction data has no stock levels, purchase costs or supplier lead
times, so these are stated here rather than hidden in the code.
"""

# Forecasting
HORIZON_WEEKS = 4            # forecast the next four weeks
N_PRODUCTS = 300             # the products that make up most of the revenue
MIN_HISTORY_WEEKS = 52       # need a full year to see seasonality
BACKTEST_WEEKS = 24          # last 24 weeks are held out for testing

# Inventory policy (weekly review, order up to a target level)
LEAD_TIME_WEEKS = 2          # supplier lead time assumed for every product
SERVICE_LEVEL = {"A": 0.98, "B": 0.95, "C": 0.90}   # share of weeks without a stockout
COST_RATIO = 0.60            # purchase cost as a share of the selling price
RULE_OF_THUMB_WEEKS = 4      # the usual habit: keep about four weeks of recent average sales
