"""Customer segments and early warning of customers who are buying less.

Rule-based on purpose: a sales team can check every segment by hand, which
builds trust before anything more complex is introduced.
"""
import numpy as np
import pandas as pd

SEGMENTS = ["Top customers", "Regular", "New", "Slipping", "Lapsed", "Occasional"]


def segment(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales[sales["has_customer"]]
    end = s["invoice_date"].max().normalize()
    year = s[s["invoice_date"] > end - pd.Timedelta(days=365)]
    c = year.groupby("customer_id").agg(last_order=("invoice_date", "max"), orders=("invoice", "nunique"),
                                        revenue=("revenue", "sum"), country=("country", "first"))
    first = s.groupby("customer_id")["invoice_date"].min()
    recent = year[year["invoice_date"] > end - pd.Timedelta(days=90)].groupby("customer_id")["revenue"].sum()
    before = year[year["invoice_date"] <= end - pd.Timedelta(days=90)].groupby("customer_id")["revenue"].sum()
    c = c.join(first.rename("first_order")).join(recent.rename("rev_last_90d")).join(before.rename("rev_prior_275d"))
    c[["rev_last_90d", "rev_prior_275d"]] = c[["rev_last_90d", "rev_prior_275d"]].fillna(0)
    c["days_since_order"] = (end - c["last_order"].dt.normalize()).dt.days
    # Compare like with like: the prior 275 days scaled to a 90-day rate.
    c["expected_90d"] = c["rev_prior_275d"] * 90 / 275
    c["change_vs_usual"] = np.where(c["expected_90d"] > 0, c["rev_last_90d"] / c["expected_90d"] - 1, np.nan)
    top_cut = c["revenue"].quantile(0.9)

    def label(r):
        if (end - r["first_order"]).days <= 90:
            return "New"
        if r["days_since_order"] > 180:
            return "Lapsed"
        if r["orders"] >= 4 and r["change_vs_usual"] <= -0.5:
            return "Slipping"
        if r["revenue"] >= top_cut:
            return "Top customers"
        if r["orders"] >= 4:
            return "Regular"
        return "Occasional"

    c["segment"] = c.apply(label, axis=1)
    c["reason"] = [
        f"Spent £{l90:,.0f} in the last 90 days against a usual £{exp:,.0f}; last order {d} days ago."
        if seg == "Slipping" else ""
        for seg, l90, exp, d in zip(c["segment"], c["rev_last_90d"], c["expected_90d"], c["days_since_order"])]
    return c.reset_index()
