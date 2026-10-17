"""Load clean data into DuckDB and build the weekly demand table."""
import duckdb
import pandas as pd

from stocksense.clean import CleanResult


def connect(path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path))


def build(con: duckdb.DuckDBPyConnection, result: CleanResult) -> None:
    for name, frame in [("fct_sales", result.sales), ("fct_returns", result.returns),
                        ("fct_adjustments", result.adjustments), ("dq_report", result.report),
                        ("dq_descriptions", result.descriptions)]:
        con.register("_tmp", frame)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp")
        con.unregister("_tmp")

    con.execute("""
    CREATE OR REPLACE TABLE dim_product AS
    SELECT stock_code, any_value(description) AS description,
           min(invoice_date)::DATE AS first_sold, max(invoice_date)::DATE AS last_sold,
           sum(revenue) / sum(quantity) AS avg_price,
           mode(quantity) AS usual_pack,      -- most common line quantity: a good guess at the case size
           count(DISTINCT invoice) AS orders
    FROM fct_sales GROUP BY stock_code
    """)
    con.execute("""
    CREATE OR REPLACE TABLE dim_customer AS
    SELECT customer_id, any_value(country) AS country,
           min(invoice_date)::DATE AS first_order, max(invoice_date)::DATE AS last_order,
           count(DISTINCT invoice) AS orders, sum(revenue) AS revenue
    FROM fct_sales WHERE has_customer GROUP BY customer_id
    """)
    # Weeks start on Monday. The first and last weeks of the data are partial, so they are dropped.
    con.execute("""
    CREATE OR REPLACE TABLE weekly_demand AS
    WITH bounds AS (
        SELECT date_trunc('week', min(invoice_date))::DATE + 7 AS first_week,
               date_trunc('week', max(invoice_date))::DATE - 7 AS last_week
        FROM fct_sales
    ), weeks AS (
        SELECT unnest(generate_series(first_week, last_week, INTERVAL 7 DAY))::DATE AS week FROM bounds
    ), sold AS (
        SELECT stock_code, date_trunc('week', invoice_date)::DATE AS week,
               sum(quantity) FILTER (WHERE NOT is_bulk_outlier) AS units, sum(revenue) AS revenue
        FROM fct_sales GROUP BY ALL
    )
    SELECT p.stock_code, w.week, coalesce(s.units, 0) AS units, coalesce(s.revenue, 0) AS revenue
    FROM dim_product p
    JOIN weeks w ON w.week >= date_trunc('week', p.first_sold)::DATE
    LEFT JOIN sold s ON s.stock_code = p.stock_code AND s.week = w.week
    """)


def weekly(con: duckdb.DuckDBPyConnection, codes=None) -> pd.DataFrame:
    df = con.execute("SELECT stock_code, week, units, revenue FROM weekly_demand ORDER BY stock_code, week").df()
    if codes is not None:
        df = df[df["stock_code"].isin(codes)]
    df["week"] = pd.to_datetime(df["week"])
    return df.reset_index(drop=True)
