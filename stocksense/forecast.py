"""Weekly demand forecasts for each product, one to four weeks ahead.

Three approaches are compared on the same held-out weeks:
  - moving average: the average of the last eight weeks (what most people do by hand)
  - seasonal naive: the same week last year
  - gradient boosting and a neural network: one model trained across all products

The models are tested the way they would be used. At each forecast date
("origin") they only see weeks up to that date, they are retrained every four
weeks, and they are scored on what actually sold afterwards. Gradient boosting
also gives a range (10th to 90th percentile), so users can see how sure the
forecast is.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from stocksense import config

LAG_COLS = ["y0", "y1", "y2", "y3", "rm4", "rm8", "rm13", "rm26", "rsd8", "nz13", "y_ly", "y_ly_rm4"]
FEATURES = LAG_COLS + ["h", "woy_sin", "woy_cos", "wk_to_xmas", "price"]
REFIT_EVERY = 4


def select_products(weekly: pd.DataFrame, n: int = config.N_PRODUCTS) -> list[str]:
    """The top products by revenue over the last year, with enough history to learn from."""
    last = weekly["week"].max()
    first_seen = weekly.groupby("stock_code")["week"].min()
    needed = config.MIN_HISTORY_WEEKS + config.BACKTEST_WEEKS
    old_enough = first_seen[first_seen <= last - pd.Timedelta(weeks=needed)].index
    recent = weekly[(weekly["week"] > last - pd.Timedelta(weeks=52)) & weekly["stock_code"].isin(old_enough)]
    return recent.groupby("stock_code")["revenue"].sum().nlargest(n).index.tolist()


def make_features(weekly: pd.DataFrame, horizon: int = config.HORIZON_WEEKS) -> pd.DataFrame:
    """One row per product, origin week and horizon. Every feature is known at the origin."""
    df = weekly.sort_values(["stock_code", "week"]).copy()
    g = df.groupby("stock_code")["units"]
    for k in range(4):
        df[f"y{k}"] = g.shift(k)
    for w in (4, 8, 13, 26):
        df[f"rm{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    df["rsd8"] = g.transform(lambda s: s.rolling(8, min_periods=2).std())
    df["nz13"] = g.transform(lambda s: (s > 0).rolling(13, min_periods=1).mean())
    rev = df.groupby("stock_code")["revenue"].transform(lambda s: s.rolling(13, min_periods=1).sum())
    units13 = g.transform(lambda s: s.rolling(13, min_periods=1).sum())
    df["price"] = (rev / units13.replace(0, np.nan)).groupby(df["stock_code"]).ffill()

    parts = []
    for h in range(1, horizon + 1):
        p = df.copy()
        p["h"] = h
        p["target_week"] = p["week"] + pd.Timedelta(weeks=h)
        p["target"] = g.shift(-h)
        p["y_ly"] = g.shift(52 - h)                                          # same week last year
        p["y_ly_rm4"] = g.transform(lambda s, h=h: s.shift(52 - h).rolling(4, min_periods=1, center=True).mean())
        parts.append(p)
    out = pd.concat(parts, ignore_index=True).rename(columns={"week": "origin"})
    woy = out["target_week"].dt.isocalendar().week.astype(float)
    out["woy_sin"], out["woy_cos"] = np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52)
    xmas = pd.to_datetime(out["target_week"].dt.year.astype(str) + "-12-25")
    out["wk_to_xmas"] = ((xmas - out["target_week"]).dt.days / 7).clip(lower=-2, upper=30)
    return out


# ---------- models ----------

def _hgb(loss: str, **kw) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss=loss, max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=0, **kw)


class GradientBoosting:
    """Point forecast (Poisson loss, suits counts) plus a 10th-90th percentile range."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.mid = _hgb("poisson").fit(X, y)
        self.lo = _hgb("quantile", quantile=0.1).fit(X, y)
        self.hi = _hgb("quantile", quantile=0.9).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        mid = self.mid.predict(X)
        lo = np.minimum(self.lo.predict(X), mid)
        hi = np.maximum(self.hi.predict(X), mid)
        return pd.DataFrame({"hgb": mid, "p10": lo.clip(min=0), "p90": hi})


class NeuralNet:
    """A small feed-forward network on log-scaled features, trained across all products."""

    def __init__(self, epochs: int = 25, seed: int = 0):
        self.epochs, self.seed = epochs, seed

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        Z = X.copy()
        Z[LAG_COLS] = np.log1p(Z[LAG_COLS].clip(lower=0))
        Z["price"] = np.log1p(Z["price"])
        return Z.fillna(Z.median(numeric_only=True)).fillna(0).to_numpy(np.float32)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        torch.manual_seed(self.seed)
        self.scaler = StandardScaler()
        Xs = torch.tensor(self.scaler.fit_transform(self._prep(X)), dtype=torch.float32)
        ys = torch.tensor(np.log1p(y.to_numpy(np.float32)), dtype=torch.float32).unsqueeze(1)
        self.net = torch.nn.Sequential(torch.nn.Linear(Xs.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                       torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)
        gen = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            for idx in torch.randperm(len(Xs), generator=gen).split(512):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(self.net(Xs[idx]), ys[idx])
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        with torch.no_grad():
            Xs = torch.tensor(self.scaler.transform(self._prep(X)), dtype=torch.float32)
            return np.expm1(self.net(Xs).squeeze(1).numpy()).clip(min=0)


def baselines(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"moving_avg": rows["rm8"].to_numpy(),
                         "seasonal_naive": rows["y_ly"].fillna(rows["rm8"]).to_numpy()})


# ---------- backtest and final forecast ----------

def backtest(feats: pd.DataFrame, test_weeks: int = config.BACKTEST_WEEKS, use_nn: bool = True) -> pd.DataFrame:
    """Forecast every week of the held-out period from the week before it, retraining every four weeks."""
    weeks = np.sort(feats["origin"].unique())
    last_week = weeks[-1]
    test_start = last_week - np.timedelta64(7 * (test_weeks - 1), "D")
    origins = [o for o in weeks if test_start - np.timedelta64(7, "D") <= o < last_week]
    labelled = feats.dropna(subset=["target"])
    out, models = [], None
    for i, origin in enumerate(origins):
        if i % REFIT_EVERY == 0:
            train = labelled[labelled["target_week"] <= origin]         # nothing after the origin
            models = {"gb": GradientBoosting().fit(train[FEATURES], train["target"])}
            if use_nn:
                models["nn"] = NeuralNet().fit(train[FEATURES], train["target"])
        rows = labelled[(labelled["origin"] == origin) & (labelled["target_week"] >= test_start)]
        if rows.empty:
            continue
        pred = pd.concat([rows[["stock_code", "origin", "h", "target_week", "target"]].reset_index(drop=True),
                          baselines(rows), models["gb"].predict(rows[FEATURES])], axis=1)
        if use_nn:
            pred["neural_net"] = models["nn"].predict(rows[FEATURES])
        out.append(pred)
    return pd.concat(out, ignore_index=True).rename(columns={"target": "actual"})


def score(bt: pd.DataFrame) -> pd.DataFrame:
    """WAPE: total absolute error as a share of total units sold. Bias: over (+) or under (-) forecasting."""
    models = [c for c in ["moving_avg", "seasonal_naive", "neural_net", "hgb"] if c in bt]
    rows = []
    for h, grp in [("all", bt)] + list(bt.groupby("h")):
        for m in models:
            err = grp[m] - grp["actual"]
            rows.append({"horizon": h, "model": m, "wape": err.abs().sum() / grp["actual"].sum(),
                         "bias": err.sum() / grp["actual"].sum()})
    res = pd.DataFrame(rows)
    res["model"] = res["model"].replace({"hgb": "gradient_boosting"})
    return res


def interval_coverage(bt: pd.DataFrame) -> float:
    return float(((bt["actual"] >= bt["p10"]) & (bt["actual"] <= bt["p90"])).mean())


def forecast_future(feats: pd.DataFrame) -> pd.DataFrame:
    labelled = feats.dropna(subset=["target"])
    model = GradientBoosting().fit(labelled[FEATURES], labelled["target"])
    last = feats["origin"].max()
    rows = feats[feats["origin"] == last].reset_index(drop=True)
    pred = model.predict(rows[FEATURES])
    return pd.concat([rows[["stock_code", "h", "target_week"]], pred], axis=1).rename(
        columns={"target_week": "week", "hgb": "forecast"})
