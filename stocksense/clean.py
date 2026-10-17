"""Turn raw ERP invoice lines into clean sales, with a record of every fix.

Each rule says what it found, what was done and why, so anyone can check
the numbers behind a forecast. Nothing is silently dropped: removed rows are
counted in the quality report, and returns and stock adjustments are kept in
their own tables.
"""
from dataclasses import dataclass, field

import pandas as pd

# Real products have a five-digit code, sometimes with a letter suffix (e.g. 85123A).
# Anything else is postage, fees, manual adjustments, samples or gift vouchers.
PRODUCT_CODE = r"^\d{5}[A-Za-z]{0,3}$"

# One-off bulk deals distort weekly demand: flag lines that are both large and
# far above what the product normally sells in one line.
BULK_MIN_QTY = 1000
BULK_MULTIPLE = 10


@dataclass
class CleanResult:
    sales: pd.DataFrame
    returns: pd.DataFrame
    adjustments: pd.DataFrame
    report: pd.DataFrame
    descriptions: pd.DataFrame = field(default_factory=pd.DataFrame)


def clean(raw: pd.DataFrame) -> CleanResult:
    df = raw.copy()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["invoice"] = df["invoice"].astype(str).str.strip()
    total = len(df)
    rows = []

    def log(rule, n, action, why, share=True):
        rows.append({"rule": rule, "rows": int(n), "share": n / total if share else None, "action": action, "why": why})

    dup = df.duplicated()
    log("Exact duplicate lines", dup.sum(), "Removed",
        "The same invoice line recorded twice would double-count demand.")
    df = df[~dup]

    cancelled = df["invoice"].str.startswith("C")
    returns = df[cancelled].copy()
    log("Cancelled invoices (codes starting with C)", cancelled.sum(), "Moved to returns table",
        "Cancellations are not demand; they are kept separately for return-rate analysis.")
    df = df[~cancelled]

    non_product = ~df["stock_code"].str.match(PRODUCT_CODE)
    log("Non-product codes (postage, fees, manual adjustments, samples, vouchers)", non_product.sum(), "Removed",
        "These are charges and corrections, not stock that needs forecasting.")
    df = df[~non_product]

    adj = df["quantity"] <= 0
    adjustments = df[adj].copy()
    log("Zero or negative quantity on a sales invoice", adj.sum(), "Moved to adjustments table",
        "These are stock corrections (damaged, lost, found), not customer orders.")
    df = df[~adj]

    free = df["price"] <= 0
    log("Zero or negative price", free.sum(), "Removed",
        "Free or written-off lines carry no revenue and usually mirror an adjustment.")
    df = df[~free]

    no_cust = df["customer_id"].isna()
    log("Missing customer ID", no_cust.sum(), "Kept for demand, excluded from customer analysis",
        "The goods still left the warehouse, but the line cannot be linked to a customer.")
    df["has_customer"] = ~no_cust

    p95 = df.groupby("stock_code")["quantity"].transform(lambda q: q.quantile(0.95))
    bulk = (df["quantity"] >= BULK_MIN_QTY) & (df["quantity"] > BULK_MULTIPLE * p95)
    log("One-off bulk orders (1,000+ units and 10x the usual line size)", bulk.sum(),
        "Kept in sales, excluded from forecasting",
        "A single exceptional deal would teach the model to expect it again.")
    df["is_bulk_outlier"] = bulk

    # One description per product: the most common spelling wins.
    desc = (df.dropna(subset=["description"])
              .assign(description=lambda d: d["description"].str.strip().str.upper())
              .groupby(["stock_code", "description"]).size().rename("lines").reset_index()
              .sort_values(["stock_code", "lines"], ascending=[True, False]))
    variants = desc.groupby("stock_code")["description"].nunique()
    canonical = desc.drop_duplicates("stock_code").set_index("stock_code")["description"]
    n_variant_codes = int((variants > 1).sum())
    log("Products with more than one description (count of products)", n_variant_codes, "Standardised to the most common description",
        "The same product typed in different ways would look like several products.", share=False)
    df["description"] = df["stock_code"].map(canonical).fillna("UNKNOWN")

    df["revenue"] = df["quantity"] * df["price"]
    df["customer_id"] = df["customer_id"].astype("Int64")
    report = pd.DataFrame(rows)
    report.loc[len(report)] = {"rule": "Clean sales lines kept", "rows": len(df), "share": len(df) / total,
                               "action": "Used for analysis", "why": ""}
    return CleanResult(sales=df.reset_index(drop=True), returns=returns.reset_index(drop=True),
                       adjustments=adjustments.reset_index(drop=True), report=report,
                       descriptions=desc.rename(columns={"lines": "times_used"}))
