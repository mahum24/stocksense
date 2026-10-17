"""Download the UCI Online Retail II dataset and cache it as parquet.

Source: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5CG6D (CC BY 4.0). About a million
invoice lines from a UK-based wholesaler, December 2009 to December 2011.
"""
import io
import zipfile

import pandas as pd
import requests

from stocksense import RAW_PATH

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def download(url: str = URL) -> pd.DataFrame:
    print(f"Downloading {url}")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        with z.open(name) as f:
            print("Reading both sheets (takes a minute or two)")
            sheets = pd.read_excel(io.BytesIO(f.read()), sheet_name=None,
                                   dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price", "customer_id", "country"]
    return df


def load_raw(refresh: bool = False) -> pd.DataFrame:
    if refresh or not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        download().to_parquet(RAW_PATH, index=False)
    return pd.read_parquet(RAW_PATH)
