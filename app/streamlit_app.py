"""StockSense dashboard.

    streamlit run app/streamlit_app.py
"""
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from stocksense import WAREHOUSE_PATH, config, inventory

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LAYOUT = dict(font=dict(family="Inter, system-ui, sans-serif", size=13), margin=dict(l=8, r=8, t=48, b=8),
              plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb", legend=dict(orientation="h", y=1.12, x=0, title=None),
              xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#e1e0d9", zeroline=False), height=380)
MODEL_NAMES = {"moving_avg": "8-week moving average", "seasonal_naive": "Same week last year",
               "neural_net": "Neural network", "gradient_boosting": "Gradient boosting"}

st.set_page_config(page_title="StockSense", page_icon="📦", layout="wide")


@st.cache_resource
def con():
    return duckdb.connect(str(WAREHOUSE_PATH), read_only=True)


@st.cache_data
def q(sql: str) -> pd.DataFrame:
    return con().execute(sql).df()


def chart(fig):
    fig.update_layout(**LAYOUT)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


if not WAREHOUSE_PATH.exists():
    st.error("No data yet. Run `python -m stocksense.pipeline` first.")
    st.stop()

s = q("SELECT * FROM summary").iloc[0]
st.sidebar.title("StockSense")
st.sidebar.caption(f"UK wholesale distributor · weekly data {s.first_week} to {s.last_week}")
page = st.sidebar.radio("Page", ["Overview", "Reorder this week", "Product forecast", "Forecast accuracy",
                                 "Data quality", "Customers"])

# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Demand forecasting and smart reordering")
    st.write("Forecasts weekly demand for the products that make up most of the revenue, turns them into "
             "reorder suggestions, and shows how sure each forecast is.")
    c = st.columns(4)
    c[0].metric("Invoice lines kept after cleaning", f"{s.clean_lines / s.raw_lines:.1%}")
    c[1].metric("Products forecast", f"{s.products_forecast:,.0f}")
    c[2].metric("Error over the reorder window", f"{s.lead_wape_gb:.1%}",
                f"{s.lead_wape_gb - s.lead_wape_ma:+.1%} vs moving average", delta_color="inverse")
    c[3].metric("Stock needed for the same service", f"{s.stock_saving:.0%} less")
    st.caption(f"{s.clean_lines:,.0f} of {s.raw_lines:,.0f} raw invoice lines kept. Stock held for the same service: "
               f"£{s.stock_forecast:,.0f} forecast-driven against £{s.rule_stock_same_fill:,.0f} rule of thumb.")

    st.subheader("Service against stock held")
    st.caption(f"Replay of the last {config.BACKTEST_WEEKS} weeks with real demand, a {config.LEAD_TIME_WEEKS}-week "
               "lead time and orders rounded to the usual pack size. Higher and further left is better.")
    fr = q("SELECT * FROM inv_frontier")
    fig = px.line(fr, x="avg_stock_value", y="fill_rate", color="policy", markers=True, hover_name="setting",
                  color_discrete_sequence=SERIES, labels={"avg_stock_value": "Average stock held (£ at cost)",
                                                          "fill_rate": "Share of demand served"})
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickprefix="£")
    chart(fig)
    st.info(f"To serve {s.fill_forecast:.1%} of demand, the forecast-driven policy holds £{s.stock_forecast:,.0f} "
            f"of stock on average. Keeping weeks of average sales on the shelf needs about "
            f"£{s.rule_stock_same_fill:,.0f} for the same result, because it reacts late to seasonal peaks and "
            f"holds the same cover for steady and erratic products alike.")

# ---------------------------------------------------------------- Reorder
elif page == "Reorder this week":
    plan = q("SELECT * FROM inv_plan")
    todo = plan[plan["suggested_qty"] > 0]
    st.title("Reorder this week")
    st.caption("Stock positions come from the replay above, because the source data has no stock levels. "
               "Purchase cost is assumed to be "
               f"{config.COST_RATIO:.0%} of the selling price.")
    total = float(todo["order_cost"].sum())
    budget = st.slider("Purchasing budget this week (£)", 0, int(total * 1.2) + 1, int(total), step=100)
    approved = inventory.fit_budget(plan, budget)
    approved["approved_cost"] = approved["approved_qty"] * approved["unit_cost"]
    c = st.columns(3)
    c[0].metric("Products to order", f"{(approved['approved_qty'] > 0).sum()} of {len(todo)}")
    c[1].metric(f"Spend (of £{total:,.0f} suggested)", f"£{approved['approved_cost'].sum():,.0f}")
    c[2].metric("Urgent products covered", f"{(approved['urgent'] & (approved['approved_qty'] > 0)).sum()} "
                f"of {int(todo['urgent'].sum())}")
    if budget < total:
        st.caption("When the budget is short, an optimiser picks whole packs to protect the most margin, "
                   "favouring products that would run out before a delivery could arrive.")
    view = approved[["urgent", "description", "abc", "on_hand", "on_order", "suggested_qty", "approved_qty",
                     "pack", "approved_cost", "reason"]]
    st.dataframe(view, width="stretch", hide_index=True, column_config={
        "urgent": st.column_config.CheckboxColumn("Urgent"), "description": "Product", "abc": "Class",
        "on_hand": st.column_config.NumberColumn("On hand", format="%d"),
        "on_order": st.column_config.NumberColumn("On order", format="%d"),
        "suggested_qty": st.column_config.NumberColumn("Suggested", format="%d"),
        "approved_qty": st.column_config.NumberColumn("Within budget", format="%d"),
        "pack": st.column_config.NumberColumn("Pack", format="%d"),
        "approved_cost": st.column_config.NumberColumn("Cost", format="£%.0f"),
        "reason": st.column_config.TextColumn("Why", width="large")})
    st.download_button("Download order list (CSV)", view.to_csv(index=False), "order_list.csv", "text/csv")

# ---------------------------------------------------------------- Product
elif page == "Product forecast":
    products = q("""SELECT p.stock_code, p.description, c.abc, c.xyz FROM inv_classes c
                    JOIN dim_product p USING (stock_code) ORDER BY c.revenue DESC""")
    label = st.selectbox("Product", products["description"] + " (" + products["stock_code"] + ")")
    code = label.rsplit("(", 1)[1].rstrip(")")
    row = products.set_index("stock_code").loc[code]
    hist = q(f"SELECT week, units FROM weekly_demand WHERE stock_code = '{code}' ORDER BY week")
    bt = q(f"SELECT target_week, hgb, moving_avg FROM fc_backtest WHERE stock_code = '{code}' AND h = 1")
    fut = q(f"SELECT week, forecast, p10, p90 FROM fc_future WHERE stock_code = '{code}' ORDER BY week")
    fig = go.Figure()
    fig.add_bar(x=hist["week"], y=hist["units"], name="Units sold", marker_color="#c3c2b7")
    fig.add_scatter(x=bt["target_week"], y=bt["moving_avg"], name="Moving average (1 week ahead)",
                    line=dict(color=SERIES[1], width=2, dash="dot"))
    fig.add_scatter(x=bt["target_week"], y=bt["hgb"], name="Model (1 week ahead)", line=dict(color=SERIES[0], width=2))
    fig.add_scatter(x=pd.concat([fut["week"], fut["week"][::-1]]), y=pd.concat([fut["p90"], fut["p10"][::-1]]),
                    fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), mode="lines",
                    name="Likely range (80%)", hoverinfo="skip")
    fig.add_scatter(x=fut["week"], y=fut["forecast"], name="Next 4 weeks", line=dict(color=SERIES[0], width=3),
                    mode="lines+markers")
    fig.update_layout(hovermode="x unified")
    st.subheader(f"{row.description} · class {row.abc}{row.xyz}")
    chart(fig)
    plan = q(f"SELECT reason FROM inv_plan WHERE stock_code = '{code}'")
    if not plan.empty:
        st.write(plan["reason"].iloc[0])
    st.caption("Class A/B/C ranks products by revenue; X/Y/Z by how steady weekly demand is (X steady, Z erratic). "
               "Forecasts drop over Christmas because the business closes then, which the model learned from last year.")

# ---------------------------------------------------------------- Accuracy
elif page == "Forecast accuracy":
    st.title("How accurate are the forecasts?")
    st.write("Every model was tested on the last 24 weeks, forecasting from data available at the time and "
             "retrained every four weeks. Error is shown as WAPE: total units wrong as a share of units sold.")
    m = q("SELECT * FROM fc_metrics")
    m["model"] = m["model"].map(MODEL_NAMES)
    allh = m[m["horizon"] == "all"].sort_values("wape")
    c = st.columns(3)
    c[0].metric("Error over the 3-week reorder window", f"{s.lead_wape_gb:.1%}")
    c[1].metric("Range coverage (target 80%)", f"{s.interval_coverage:.1%}")
    c[2].metric("Same window, 8-week moving average", f"{s.lead_wape_ma:.1%}")
    st.dataframe(allh[["model", "wape", "bias"]], hide_index=True, width="stretch", column_config={
        "model": "Model", "wape": st.column_config.NumberColumn("Weekly error (WAPE)", format="percent"),
        "bias": st.column_config.NumberColumn("Bias (+ over, - under)", format="percent")})
    bt = q("SELECT date_trunc('month', target_week) AS month, sum(actual) a, sum(hgb) g, sum(moving_avg) ma "
           "FROM fc_backtest GROUP BY 1 ORDER BY 1")
    bt["Gradient boosting"] = bt["g"] / bt["a"] - 1
    bt["8-week moving average"] = bt["ma"] / bt["a"] - 1
    fig = px.bar(bt.melt(id_vars="month", value_vars=["Gradient boosting", "8-week moving average"]),
                 x="month", y="value", color="variable", barmode="group", color_discrete_sequence=SERIES,
                 labels={"value": "", "month": ""})
    fig.update_yaxes(tickformat="+.0%")
    st.subheader("Over- or under-forecasting by month")
    chart(fig)
    st.caption("Week-by-week demand for a single product at a wholesaler is lumpy, so no model gets close to "
               "perfect. The gains show up where decisions are made: over the reorder window, and in the "
               "autumn peak, when the moving average falls behind.")

# ---------------------------------------------------------------- Data quality
elif page == "Data quality":
    st.title("What the raw data needed")
    st.write("The source is an ERP-style invoice export that was never designed for analysis. Each fix is "
             "listed with what was found and why it matters.")
    dq = q("SELECT * FROM dq_report")
    dq["share"] = dq["share"].map(lambda v: "" if pd.isna(v) else f"{v:.2%}")
    st.dataframe(dq, hide_index=True, width="stretch", column_config={
        "rule": st.column_config.TextColumn("Check", width="medium"),
        "rows": st.column_config.NumberColumn("Count", format="localized"),
        "share": "Share of lines",
        "action": "Action", "why": st.column_config.TextColumn("Why", width="large")})
    st.subheader("Same product, different descriptions")
    ex = q("""SELECT stock_code, description, times_used FROM dq_descriptions WHERE stock_code IN (
                SELECT stock_code FROM dq_descriptions GROUP BY 1 ORDER BY count(*) DESC LIMIT 3)
              ORDER BY stock_code, times_used DESC""")
    st.dataframe(ex, hide_index=True, width="stretch", column_config={
        "stock_code": "Product code", "description": "Description as typed",
        "times_used": st.column_config.NumberColumn("Lines using it", format="localized")})

# ---------------------------------------------------------------- Customers
elif page == "Customers":
    st.title("Customers")
    cs = q("SELECT * FROM cust_segments")
    seg = cs.groupby("segment").agg(customers=("customer_id", "count"), revenue=("revenue", "sum")).reset_index()
    fig = px.bar(seg.sort_values("revenue"), x="revenue", y="segment", orientation="h", text="customers",
                 color_discrete_sequence=SERIES, labels={"revenue": "£", "segment": ""})
    fig.update_xaxes(tickprefix="£")
    st.subheader("Revenue in the last 12 months by segment")
    st.caption("Bar labels show the number of customers.")
    chart(fig)
    st.subheader("Slipping customers to contact")
    st.caption("Regular customers whose spending in the last 90 days is less than half their usual rate.")
    slip = cs[cs["segment"] == "Slipping"].sort_values("expected_90d", ascending=False)
    st.dataframe(slip[["customer_id", "country", "revenue", "rev_last_90d", "expected_90d", "days_since_order", "reason"]],
                 hide_index=True, width="stretch", column_config={
                     "customer_id": st.column_config.NumberColumn("Customer", format="%d"), "country": "Country",
                     "revenue": st.column_config.NumberColumn("12-month revenue", format="£%.0f"),
                     "rev_last_90d": st.column_config.NumberColumn("Last 90 days", format="£%.0f"),
                     "expected_90d": st.column_config.NumberColumn("Usual 90 days", format="£%.0f"),
                     "days_since_order": st.column_config.NumberColumn("Days since order", format="%d"),
                     "reason": st.column_config.TextColumn("Why", width="large")})
