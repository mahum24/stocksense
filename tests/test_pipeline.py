import pandas as pd
from conftest import make_raw

from stocksense import pipeline
from stocksense.clean import clean
from stocksense.customers import segment


def test_pipeline_runs_end_to_end(tmp_path):
    s = pipeline.run(raw=make_raw(n_products=8, seed=1), use_nn=False, path=tmp_path / "w.duckdb")
    assert s["clean_lines"] < s["raw_lines"]
    assert 0 < s["fill_forecast"] <= 1
    assert 0 <= s["interval_coverage"] <= 1


def test_a_regular_customer_who_stops_buying_is_flagged_with_a_reason(raw):
    sales = clean(raw).sales
    cust = sales["customer_id"].value_counts().index[0]           # the busiest customer
    end = sales["invoice_date"].max()
    quiet = (sales["customer_id"] == cust) & (sales["invoice_date"] > end - pd.Timedelta(days=100))
    seg = segment(sales[~quiet]).set_index("customer_id")
    assert seg.loc[cust, "segment"] == "Slipping"
    assert "usual" in seg.loc[cust, "reason"]
