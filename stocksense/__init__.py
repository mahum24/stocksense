"""StockSense: demand forecasting and smart reordering for a wholesale distributor."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("STOCKSENSE_DATA_DIR", ROOT / "data"))
RAW_PATH = DATA_DIR / "raw" / "online_retail_ii.parquet"
WAREHOUSE_PATH = DATA_DIR / "stocksense.duckdb"
