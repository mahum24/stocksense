from stocksense.clean import clean


def test_every_problem_is_caught_and_counted(raw):
    r = clean(raw)
    rep = r.report.set_index("rule")["rows"]
    assert rep["Exact duplicate lines"] == 5
    assert len(r.returns) == 3 and r.returns["invoice"].str.startswith("C").all()
    assert rep.filter(like="Non-product").iloc[0] == 2
    assert len(r.adjustments) == 2
    assert rep["Zero or negative price"] == 1
    assert r.sales["is_bulk_outlier"].sum() == 1
    assert (~r.sales["has_customer"]).sum() >= 10


def test_nothing_is_lost_silently(raw):
    r = clean(raw)
    rep = r.report.set_index("rule")["rows"]
    removed = rep[["Exact duplicate lines", "Zero or negative price"]].sum() + rep.filter(like="Non-product").iloc[0]
    assert len(raw) == removed + len(r.returns) + len(r.adjustments) + len(r.sales)


def test_one_description_per_product(raw):
    s = clean(raw).sales
    assert s.groupby("stock_code")["description"].nunique().max() == 1
    assert "PRODUCT 0 MISSPELT" not in set(s["description"])
