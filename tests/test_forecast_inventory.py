import numpy as np
import pandas as pd
import pytest

from stocksense import forecast, inventory, warehouse
from stocksense.clean import clean


@pytest.fixture(scope="module")
def setup(raw, tmp_path_factory):
    con = warehouse.connect(tmp_path_factory.mktemp("db") / "t.duckdb")
    warehouse.build(con, clean(raw))
    wk = warehouse.weekly(con)
    return con, wk, forecast.make_features(wk)


def test_features_never_use_the_future(setup):
    _, wk, feats = setup
    origin = wk["week"].sort_values().unique()[60]
    changed = wk.copy()
    changed.loc[changed["week"] > origin, "units"] = 10_000          # rewrite everything after the origin
    a = feats[feats["origin"] == origin].sort_values(["stock_code", "h"])
    b = forecast.make_features(changed)
    b = b[b["origin"] == origin].sort_values(["stock_code", "h"])
    cols = [c for c in forecast.FEATURES if c not in ("y_ly", "y_ly_rm4")]   # last year is in the past for h <= 4
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))
    assert (b["target"] == 10_000).all()                             # only the target sees the future


def test_backtest_only_scores_held_out_weeks(setup):
    _, _, feats = setup
    bt = forecast.backtest(feats, test_weeks=8, use_nn=False)
    assert (bt["target_week"] > bt["origin"]).all()
    assert bt["target_week"].nunique() == 8
    assert (bt["p10"] <= bt["hgb"]).all() and (bt["hgb"] <= bt["p90"]).all()
    s = forecast.score(bt)
    assert set(s["model"]) == {"moving_avg", "seasonal_naive", "gradient_boosting"}


def test_replay_respects_physics(setup):
    con, wk, feats = setup
    bt = forecast.backtest(feats, test_weeks=8, use_nn=False)
    codes = sorted(bt["stock_code"].unique())
    cls = inventory.classify(wk, codes)
    prod = con.execute("SELECT * FROM dim_product").df().set_index("stock_code")
    tgt = inventory.targets(bt, cls)
    sim, state = inventory.simulate(wk[wk["week"].isin(tgt["week"])], tgt, prod["usual_pack"], prod["avg_price"])
    assert (sim["served"] <= sim["demand"]).all() and (sim["on_hand"] >= 0).all()
    packs = sim["stock_code"].map(prod["usual_pack"].clip(lower=1))
    assert (sim["order"] % packs == 0).all()
    assert set(state["policy"]) == {"rule", "forecast"}


def test_budget_is_never_exceeded():
    plan = pd.DataFrame({"suggested_qty": [120, 48, 60], "pack": [12, 6, 10], "unit_cost": [1.0, 5.0, 2.0],
                         "avg_price": [2.0, 9.0, 4.0], "urgent": [True, False, True]})
    plan["order_cost"] = plan["suggested_qty"] * plan["unit_cost"]
    out = inventory.fit_budget(plan, budget=150)
    assert (out["approved_qty"] * out["unit_cost"]).sum() <= 150
    assert (out["approved_qty"] % out["pack"] == 0).all()
    assert (out["approved_qty"] <= out["suggested_qty"]).all()
    full = inventory.fit_budget(plan, budget=10_000)
    assert (full["approved_qty"] == full["suggested_qty"]).all()


def test_abc_classes_follow_revenue(setup):
    _, wk, _ = setup
    cls = inventory.classify(wk, wk["stock_code"].unique().tolist())
    assert cls["abc"].iloc[0] == "A" and list(cls["revenue"]) == sorted(cls["revenue"], reverse=True)
