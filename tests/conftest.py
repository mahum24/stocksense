import numpy as np
import pandas as pd
import pytest


def make_raw(n_products=12, weeks=110, seed=0) -> pd.DataFrame:
    """Small invoice export with the same columns and problems as the real one."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2009-12-01")
    rows, inv = [], 500000
    for p in range(n_products):
        code, price, level = f"{20000 + p}", round(rng.uniform(0.5, 6), 2), rng.uniform(5, 40)
        for w in range(weeks):
            season = 1 + 0.6 * np.sin(2 * np.pi * (w % 52) / 52)
            for _ in range(rng.poisson(level * season / 6)):
                inv += 1
                rows.append([str(inv), code, f"PRODUCT {p}", int(rng.choice([1, 2, 6, 12])),
                             start + pd.Timedelta(weeks=w, hours=int(rng.integers(8, 17))), price,
                             float(rng.integers(12000, 12100)), "United Kingdom"])
    df = pd.DataFrame(rows, columns=["invoice", "stock_code", "description", "quantity", "invoice_date",
                                     "price", "customer_id", "country"])
    extra = df.iloc[:5].copy()                                        # 5 exact duplicates
    canc = df.iloc[5:8].copy(); canc["invoice"] = "C" + canc["invoice"]; canc["quantity"] *= -1
    post = df.iloc[8:10].copy(); post["stock_code"] = "POST"
    adj = df.iloc[10:12].copy(); adj["quantity"] = -3; adj["invoice"] = ["900001", "900002"]
    free = df.iloc[12:13].copy(); free["price"] = 0.0; free["invoice"] = "900003"
    bulk = df.iloc[13:14].copy(); bulk["quantity"] = 5000; bulk["invoice"] = "900004"
    df.loc[df.index[20:30], "customer_id"] = np.nan
    df.loc[df.index[40], "description"] = "product 0 misspelt"
    return pd.concat([df, extra, canc, post, adj, free, bulk], ignore_index=True)


@pytest.fixture(scope="session")
def raw():
    return make_raw()
