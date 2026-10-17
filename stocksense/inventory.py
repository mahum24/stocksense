"""From forecasts to reorder decisions.

The data has no stock levels, so the two ordering policies are compared by
replaying the held-out weeks: each week the policy decides how much to order,
the order arrives after the lead time, and real demand is served from what is
on the shelf. This is the test that matters to a business: how often do we run
out, and how much stock do we hold to avoid it?

  - Rule of thumb: keep about four weeks of recent average sales.
  - Forecast-driven: cover the forecast demand over the lead time plus one week,
    plus safety stock sized from the forecast's own uncertainty range and the
    product's service target.
"""
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import norm

from stocksense import config

Z_WIDTH = 2 * norm.ppf(0.9)        # a 10th-90th percentile range is this many standard deviations wide


def classify(weekly: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """ABC by revenue (A = top 80%, B = next 15%, C = the rest); XYZ by how steady weekly demand is."""
    last = weekly["week"].max()
    yr = weekly[(weekly["week"] > last - pd.Timedelta(weeks=52)) & weekly["stock_code"].isin(codes)]
    s = yr.groupby("stock_code").agg(revenue=("revenue", "sum"), mean=("units", "mean"), sd=("units", "std"))
    s = s.sort_values("revenue", ascending=False)
    share = s["revenue"].cumsum() / s["revenue"].sum()
    s["abc"] = np.where(share <= 0.80, "A", np.where(share <= 0.95, "B", "C"))
    cv = s["sd"] / s["mean"].replace(0, np.nan)
    s["xyz"] = np.where(cv < 0.5, "X", np.where(cv < 1.0, "Y", "Z"))
    s["service_level"] = s["abc"].map(config.SERVICE_LEVEL)
    return s.reset_index()


def _round_up(qty: float, pack: int) -> int:
    return int(np.ceil(qty / pack) * pack) if qty > 0 else 0


def targets(bt: pd.DataFrame, classes: pd.DataFrame, lead: int = config.LEAD_TIME_WEEKS) -> pd.DataFrame:
    """Order-up-to level for each product and week under both policies."""
    cover = lead + 1
    near = bt[bt["h"] <= cover].copy()
    near["sigma"] = (near["p90"] - near["p10"]) / Z_WIDTH
    g = near.groupby(["stock_code", "origin"])
    t = pd.DataFrame({
        "fc_cover": g["hgb"].mean() * cover,              # mean of available horizons, so the last weeks still work
        "fc_sigma": np.sqrt(g["sigma"].apply(lambda s: (s ** 2).mean()) * cover),
        "ma8": g["moving_avg"].first(),
    }).reset_index()
    t["week"] = t["origin"] + pd.Timedelta(weeks=1)
    t = t.merge(classes[["stock_code", "service_level"]], on="stock_code")
    t["z"] = norm.ppf(t["service_level"])
    t["target_forecast"] = t["fc_cover"] + t["z"] * t["fc_sigma"]
    t["target_rule"] = config.RULE_OF_THUMB_WEEKS * t["ma8"]
    return t


def simulate(demand: pd.DataFrame, tgt: pd.DataFrame, packs: pd.Series, price: pd.Series,
             lead: int = config.LEAD_TIME_WEEKS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay the held-out weeks for both policies. Unmet demand is lost (customers buy elsewhere)."""
    weekly_rows, state = [], []
    d = demand.merge(tgt[["stock_code", "week", "target_forecast", "target_rule"]], on=["stock_code", "week"])
    for code, grp in d.sort_values("week").groupby("stock_code"):
        pack = max(int(packs.get(code, 1)), 1)
        unit_cost = float(price.get(code, 0)) * config.COST_RATIO
        for policy in ("rule", "forecast"):
            col = f"target_{policy}"
            on_hand = float(grp[col].iloc[0])
            pipeline = [0.0] * lead                      # orders arriving in 1..lead weeks
            for _, r in grp.iterrows():
                on_hand += pipeline.pop(0)
                position = on_hand + sum(pipeline)
                order = _round_up(r[col] - position, pack)
                pipeline.append(order)
                served = min(on_hand, r["units"])
                on_hand -= served
                weekly_rows.append({"stock_code": code, "policy": policy, "week": r["week"], "demand": r["units"],
                                    "served": served, "order": order, "on_hand": on_hand,
                                    "stock_value": on_hand * unit_cost, "stockout": r["units"] > served})
            state.append({"stock_code": code, "policy": policy, "on_hand": on_hand, "on_order": sum(pipeline)})
    return pd.DataFrame(weekly_rows), pd.DataFrame(state)


def compare(sim: pd.DataFrame) -> pd.DataFrame:
    g = sim.groupby("policy")
    out = pd.DataFrame({
        "fill_rate": g["served"].sum() / g["demand"].sum(),
        "stockout_weeks": g["stockout"].mean(),
        "avg_stock_value": sim.groupby(["policy", "week"])["stock_value"].sum().groupby("policy").mean(),
        "lost_units": g["demand"].sum() - g["served"].sum(),
    })
    return out.loc[["rule", "forecast"]]


def reorder_list(future: pd.DataFrame, state: pd.DataFrame, classes: pd.DataFrame, products: pd.DataFrame,
                 lead: int = config.LEAD_TIME_WEEKS) -> pd.DataFrame:
    """This week's suggested orders under the forecast-driven policy, with the reason in words."""
    cover = lead + 1
    f = future[future["h"] <= cover].copy()
    f["sigma"] = (f["p90"] - f["p10"]) / Z_WIDTH
    g = f.groupby("stock_code")
    plan = pd.DataFrame({"fc_cover": g["forecast"].sum(), "fc_low": g["p10"].sum(), "fc_high": g["p90"].sum(),
                         "fc_sigma": np.sqrt(g["sigma"].apply(lambda s: (s ** 2).sum())),
                         "next_week": g["forecast"].first()}).reset_index()
    st = state[state["policy"] == "forecast"][["stock_code", "on_hand", "on_order"]]
    plan = (plan.merge(st, on="stock_code").merge(classes[["stock_code", "abc", "xyz", "service_level"]], on="stock_code")
                .merge(products[["stock_code", "description", "avg_price", "usual_pack"]], on="stock_code"))
    plan["target"] = plan["fc_cover"] + norm.ppf(plan["service_level"]) * plan["fc_sigma"]
    plan["position"] = plan["on_hand"] + plan["on_order"]
    plan["pack"] = plan["usual_pack"].clip(lower=1).astype(int)
    plan["suggested_qty"] = [_round_up(t - p, k) for t, p, k in zip(plan["target"], plan["position"], plan["pack"])]
    plan["unit_cost"] = plan["avg_price"] * config.COST_RATIO
    plan["order_cost"] = plan["suggested_qty"] * plan["unit_cost"]
    plan["weeks_cover"] = plan["on_hand"] / (plan["fc_cover"] / cover).replace(0, np.nan)
    lead_demand = plan["fc_cover"] * lead / cover
    plan["urgent"] = plan["position"] < lead_demand             # will run out before a new order can arrive
    plan["reason"] = [
        f"Expect {fc:,.0f} units over the next {cover} weeks (likely {lo:,.0f} to {hi:,.0f}). "
        f"Stock covers {wc:.1f} weeks; class {abc} target {sl:.0%} of weeks without running out."
        + (" Order now: stock will run out before a new delivery arrives." if urg else "")
        for fc, lo, hi, wc, abc, sl, urg in zip(plan["fc_cover"], plan["fc_low"], plan["fc_high"],
                                                plan["weeks_cover"].fillna(0), plan["abc"], plan["service_level"],
                                                plan["urgent"])]
    return plan.sort_values(["urgent", "order_cost"], ascending=[False, False]).reset_index(drop=True)


def fit_budget(plan: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Choose how many packs of each product to buy when the budget will not cover every suggestion.

    Integer programme: maximise the margin protected (urgent products count fully,
    others at 60%), subject to total cost <= budget. Quantities stay in whole packs
    and never exceed the suggestion.
    """
    p = plan[plan["suggested_qty"] > 0].reset_index(drop=True)
    if p.empty or p["order_cost"].sum() <= budget:
        return p.assign(approved_qty=p["suggested_qty"])
    max_packs = (p["suggested_qty"] / p["pack"]).to_numpy()
    pack_cost = (p["pack"] * p["unit_cost"]).to_numpy()
    urgency = np.where(p["urgent"], 1.0, 0.6)
    value = p["pack"].to_numpy() * p["avg_price"].to_numpy() * (1 - config.COST_RATIO) * urgency
    res = milp(c=-value, constraints=LinearConstraint(pack_cost[None, :], 0, budget),
               integrality=np.ones(len(p)), bounds=Bounds(0, max_packs))
    packs = np.round(res.x).astype(int) if res.success else np.zeros(len(p), int)
    return p.assign(approved_qty=packs * p["pack"].to_numpy())
