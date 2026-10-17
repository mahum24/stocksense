"""Run everything end to end.

    python -m stocksense.pipeline              # uses the cached raw data, downloads it the first time
    python -m stocksense.pipeline --refresh    # download the data again
    python -m stocksense.pipeline --quick      # skip the neural network (faster)
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd

from stocksense import WAREHOUSE_PATH, config, customers, forecast, inventory, warehouse
from stocksense.clean import clean
from stocksense.ingest import load_raw

warnings.filterwarnings("ignore", category=FutureWarning)


def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(con, name, frame):
    con.register("_tmp", frame)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp")
    con.unregister("_tmp")


def stock_frontier(demand, base, products, rule_weeks=(3, 4, 5, 6, 7, 8, 10), z_values=(0, 0.5, 1, 1.5, 2, 2.5)):
    """Fill rate against stock held for many settings of each policy: a fair comparison at equal stock."""
    rows = []
    for k in rule_weeks:
        t = base.assign(target_rule=k * base["ma8"], target_forecast=k * base["ma8"])
        c = inventory.compare(inventory.simulate(demand, t, products["usual_pack"], products["avg_price"])[0]).loc["rule"]
        rows.append({"policy": "Rule of thumb", "setting": f"{k} weeks of average sales", **c.to_dict()})
    for z in z_values:
        t = base.assign(target_forecast=base["fc_cover"] + z * base["fc_sigma"])
        t["target_rule"] = t["target_forecast"]
        c = inventory.compare(inventory.simulate(demand, t, products["usual_pack"], products["avg_price"])[0]).loc["forecast"]
        rows.append({"policy": "Forecast-driven", "setting": f"safety factor {z}", **c.to_dict()})
    return pd.DataFrame(rows)


def run(refresh: bool = False, use_nn: bool = True, raw: pd.DataFrame | None = None, path=WAREHOUSE_PATH) -> dict:
    step("Loading raw invoice lines")
    raw = load_raw(refresh) if raw is None else raw
    step(f"Cleaning {len(raw):,} lines")
    result = clean(raw)
    con = warehouse.connect(path)
    warehouse.build(con, result)

    step("Building weekly demand and features")
    weekly = warehouse.weekly(con)
    codes = forecast.select_products(weekly)
    wk = weekly[weekly["stock_code"].isin(codes)]
    feats = forecast.make_features(wk)

    step(f"Backtesting forecasts for {len(codes)} products over the last {config.BACKTEST_WEEKS} weeks")
    bt = forecast.backtest(feats, use_nn=use_nn)
    metrics = forecast.score(bt)
    lt = bt[bt["h"] <= config.LEAD_TIME_WEEKS + 1].groupby(["stock_code", "origin"])
    lt = lt[[c for c in ["moving_avg", "seasonal_naive", "neural_net", "hgb"] if c in bt] + ["actual"]].sum()
    lead_wape = {m: float(np.abs(lt[m] - lt["actual"]).sum() / lt["actual"].sum()) for m in lt.columns if m != "actual"}

    step("Forecasting the next four weeks")
    future = forecast.forecast_future(feats)

    step("Replaying the test weeks for both ordering policies")
    classes = inventory.classify(weekly, codes)
    products = con.execute("SELECT * FROM dim_product").df().set_index("stock_code")
    tgt = inventory.targets(bt, classes)
    demand = wk[wk["week"].isin(tgt["week"].unique())]
    sim, state = inventory.simulate(demand, tgt, products["usual_pack"], products["avg_price"])
    comparison = inventory.compare(sim).reset_index()
    frontier = stock_frontier(demand, tgt, products)
    rule = frontier[frontier["policy"] == "Rule of thumb"].sort_values("fill_rate")
    fc = comparison.set_index("policy").loc["forecast"]
    rule_stock_same_fill = float(np.interp(fc["fill_rate"], rule["fill_rate"], rule["avg_stock_value"]))

    plan = inventory.reorder_list(future, state, classes, products.reset_index())
    step("Segmenting customers")
    cust = customers.segment(result.sales)

    summary = {
        "raw_lines": len(raw), "clean_lines": len(result.sales), "products_forecast": len(codes),
        "first_week": str(weekly["week"].min().date()), "last_week": str(weekly["week"].max().date()),
        "wape_gb": float(metrics.query("horizon == 'all' and model == 'gradient_boosting'")["wape"].iloc[0]),
        "wape_ma": float(metrics.query("horizon == 'all' and model == 'moving_avg'")["wape"].iloc[0]),
        "lead_wape_gb": lead_wape["hgb"], "lead_wape_ma": lead_wape["moving_avg"],
        "interval_coverage": forecast.interval_coverage(bt),
        "fill_forecast": float(fc["fill_rate"]), "stock_forecast": float(fc["avg_stock_value"]),
        "rule_stock_same_fill": rule_stock_same_fill,
        "stock_saving": 1 - float(fc["avg_stock_value"]) / rule_stock_same_fill,
        "slipping_customers": int((cust["segment"] == "Slipping").sum()),
    }
    step("Saving results")
    for name, frame in [("fc_backtest", bt), ("fc_metrics", metrics), ("fc_future", future), ("inv_classes", classes),
                        ("inv_targets", tgt), ("inv_sim", sim), ("inv_state", state), ("inv_comparison", comparison),
                        ("inv_frontier", frontier), ("inv_plan", plan), ("cust_segments", cust),
                        ("summary", pd.DataFrame([summary]))]:
        save(con, name, frame)
    con.close()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="download the raw data again")
    ap.add_argument("--quick", action="store_true", help="skip the neural network")
    args = ap.parse_args()
    s = run(refresh=args.refresh, use_nn=not args.quick)
    print(f"\nClean lines: {s['clean_lines']:,} of {s['raw_lines']:,}")
    print(f"Lead-time forecast error (WAPE): {s['lead_wape_gb']:.1%} vs {s['lead_wape_ma']:.1%} for the moving average")
    print(f"Forecast range coverage (aim 80%): {s['interval_coverage']:.1%}")
    print(f"Fill rate {s['fill_forecast']:.1%} with £{s['stock_forecast']:,.0f} of stock; the rule of thumb needs "
          f"£{s['rule_stock_same_fill']:,.0f} for the same fill rate ({s['stock_saving']:.0%} less stock)")


if __name__ == "__main__":
    main()
