"""StockSense dashboard.

    streamlit run app/streamlit_app.py
"""
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from stocksense import WAREHOUSE_PATH, config, inventory

BLUE, ORANGE, GREEN, AMBER, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#b8b6ad"
INK, INK2, LINE, CARD = "#0b0b0b", "#5f5e5a", "#e4e3dc", "#ffffff"
LAYOUT = dict(font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK),
              margin=dict(l=4, r=4, t=36, b=4), plot_bgcolor=CARD, paper_bgcolor=CARD,
              legend=dict(orientation="h", y=1.1, x=0, title=None, font=dict(color=INK2)),
              xaxis=dict(showgrid=False, linecolor=LINE, tickfont=dict(color=INK2), title_font=dict(color=INK2)),
              yaxis=dict(gridcolor="#efeee9", zeroline=False, tickfont=dict(color=INK2), title_font=dict(color=INK2)),
              hoverlabel=dict(font_family="Inter, system-ui, sans-serif"), height=360)
MODEL_NAMES = {"moving_avg": "8-week moving average", "seasonal_naive": "Same week last year",
               "neural_net": "Neural network", "gradient_boosting": "Gradient boosting (used)"}

st.set_page_config(page_title="StockSense", page_icon="📦", layout="wide", initial_sidebar_state="expanded")

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stMarkdown, button, input { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #f6f6f3; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #10213a; }
section[data-testid="stSidebar"] * { color: #dfe6f0 !important; }
[data-testid="stSidebarNav"] a { border-radius: 8px; padding: .35rem .6rem; }
[data-testid="stSidebarNav"] a:hover { background: rgba(255,255,255,.07); }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: rgba(127,178,240,.20); }
section[data-testid="stSidebar"] .foot { font-size: .75rem; color: #8193ab !important; line-height: 1.55; margin-top: 1.6rem; }
.pg-title { font-size: 1.75rem; font-weight: 700; color: #0b0b0b; letter-spacing: -0.02em; margin: 0; }
.pg-sub { color: #5f5e5a; font-size: .98rem; margin: .35rem 0 1.3rem; max-width: 920px; line-height: 1.5; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 16px; }
.kpi { background: #fff; border: 1px solid #e4e3dc; border-radius: 12px; padding: 16px 18px; }
.kpi .l { color: #5f5e5a; font-size: .74rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.kpi .v { font-size: 1.7rem; font-weight: 700; color: #0b0b0b; margin-top: 4px; letter-spacing: -0.02em; }
.kpi .n { font-size: .82rem; margin-top: 4px; color: #5f5e5a; min-height: 1em; }
.kpi .n.good { color: #0f8a5f; } .kpi .n.bad { color: #c2362f; }
.note { background: #eef4fc; border: 1px solid #d4e3f7; border-radius: 12px; padding: 14px 18px; color: #1d3f6e;
        font-size: .95rem; line-height: 1.55; margin: 4px 0 16px; }
.section { font-size: 1.02rem; font-weight: 600; color: #0b0b0b; margin: 2px 0 2px; }
.cap { color: #5f5e5a; font-size: .85rem; margin-bottom: 6px; line-height: 1.45; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-radius: 12px; }
.reason { background:#fff; border:1px solid #e4e3dc; border-left: 4px solid #2a78d6; border-radius: 10px;
          padding: 12px 16px; color:#0b0b0b; font-size:.95rem; line-height:1.5; }
</style>
""")


@st.cache_resource
def con():
    return duckdb.connect(str(WAREHOUSE_PATH), read_only=True)


@st.cache_data
def q(sql: str) -> pd.DataFrame:
    return con().execute(sql).df()


def header(title: str, sub: str = ""):
    st.html(f'<p class="pg-title">{title}</p>' + (f'<p class="pg-sub">{sub}</p>' if sub else ""))


def kpis(items):
    """items: (label, value, note, tone) with tone '', 'good' or 'bad'."""
    cards = "".join(f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div>'
                    f'<div class="n {t}">{n}</div></div>' for l, v, n, t in items)
    st.html(f'<div class="kpis">{cards}</div>')


def card(parent, title: str, caption: str = ""):
    box = parent.container(border=True)
    box.html(f'<div class="section">{title}</div>' + (f'<div class="cap">{caption}</div>' if caption else ""))
    return box


def plot(box, fig, height=360):
    fig.update_layout(**{**LAYOUT, "height": height})
    box.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


if not WAREHOUSE_PATH.exists():
    st.error("No data yet. Run `python -m stocksense.pipeline` first.")
    st.stop()

s = q("SELECT * FROM summary").iloc[0]
first, last = pd.Timestamp(s.first_week), pd.Timestamp(s.last_week)

st.logo(str(Path(__file__).with_name("logo.svg")), size="large")
with st.sidebar:
    st.html(f'<div class="foot">Demand forecasting and smart reordering for a UK wholesale distributor.<br><br>'
            f'Weekly data {first:%d %b %Y} to {last:%d %b %Y}<br>'
            f'{s.clean_lines:,.0f} invoice lines · {s.products_forecast:,.0f} products forecast<br>'
            'Source: UCI Online Retail II (CC BY 4.0)</div>')

# ---------------------------------------------------------------- Overview
def overview():
    header("Overview", "Forecasts weekly demand for the products that drive most of the revenue, turns them into "
                       "reorder suggestions with reasons, and tests the result against real demand.")
    kpis([
        ("Stock for the same service", f"{s.stock_saving:.0%} less",
         f"£{s.stock_forecast:,.0f} vs £{s.rule_stock_same_fill:,.0f}", "good"),
        ("Demand served", f"{s.fill_forecast:.1%}", f"replay of the last {config.BACKTEST_WEEKS} weeks", ""),
        ("Error over reorder window", f"{s.lead_wape_gb:.1%}",
         f"{s.lead_wape_gb - s.lead_wape_ma:+.1%} vs moving average", "good"),
        ("Data kept after cleaning", f"{s.clean_lines / s.raw_lines:.1%}",
         f"{s.clean_lines:,.0f} of {s.raw_lines:,.0f} lines", ""),
    ])
    left, right = st.columns([3, 2], gap="medium")
    fr = q("SELECT * FROM inv_frontier")
    box = card(left, "Service against stock held",
               f"Both policies replayed against real demand with a {config.LEAD_TIME_WEEKS}-week lead time and "
               "pack-size rounding. Higher and further left is better.")
    fig = px.line(fr, x="avg_stock_value", y="fill_rate", color="policy", markers=True, hover_name="setting",
                  color_discrete_map={"Rule of thumb": GREY, "Forecast-driven": BLUE},
                  labels={"avg_stock_value": "Average stock held (£ at cost)", "fill_rate": "Demand served"})
    fig.update_traces(line_width=3, marker_size=8)
    fig.add_scatter(x=[s.stock_forecast], y=[s.fill_forecast], mode="markers", name="Current setting",
                    marker=dict(size=15, color=BLUE, line=dict(color="white", width=3)), hoverinfo="skip")
    fig.add_annotation(x=s.stock_forecast, y=s.fill_forecast,
                       text=f"{s.stock_saving:.0%} less stock<br>for the same service", showarrow=True, arrowhead=0,
                       ax=-95, ay=-50, font=dict(color=INK, size=12), bgcolor="white", bordercolor=LINE, borderpad=6)
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickprefix="£")
    plot(box, fig, 380)

    wk = q("SELECT week, sum(revenue) AS revenue FROM weekly_demand GROUP BY 1 ORDER BY 1")
    box2 = card(right, "Weekly revenue", "All products. Demand builds from September to a November peak, then "
                                         "stops over Christmas.")
    fig2 = go.Figure(go.Scatter(x=wk["week"], y=wk["revenue"], mode="lines", line=dict(color=BLUE, width=2),
                                fill="tozeroy", fillcolor="rgba(42,120,214,0.10)",
                                hovertemplate="%{x|%d %b %Y}: £%{y:,.0f}<extra></extra>"))
    fig2.update_yaxes(tickprefix="£")
    plot(box2, fig2, 380)
    st.html(f'<div class="note"><b>What this means.</b> Keeping a fixed number of weeks of recent sales reacts late '
            f'to seasonal peaks and gives steady and erratic products the same cover. Sizing stock from forecasts and '
            f'their uncertainty serves {s.fill_forecast:.1%} of demand with about '
            f'£{s.rule_stock_same_fill - s.stock_forecast:,.0f} less stock tied up on average.</div>')

# ---------------------------------------------------------------- Reorder
def reorder():
    plan = q("SELECT * FROM inv_plan")
    todo = plan[plan["suggested_qty"] > 0]
    total = float(todo["order_cost"].sum())
    header("Reorder this week", "Suggested orders under the forecast-driven policy. Stock positions come from the "
                                "replay, because the source data has no stock levels; purchase cost is assumed to be "
                                f"{config.COST_RATIO:.0%} of the selling price.")
    c1, c2 = st.columns([3, 1], gap="large", vertical_alignment="bottom")
    full = int(-(-total // 100) * 100)                        # suggested spend rounded up to the slider step
    budget = c1.slider("Purchasing budget this week (£)", 0, int(full * 1.2), full, step=100)
    urgent_only = c2.toggle("Urgent products only", value=False)
    approved = inventory.fit_budget(plan, budget)
    approved["approved_cost"] = approved["approved_qty"] * approved["unit_cost"]
    spend = approved["approved_cost"].sum()
    kpis([
        ("Products to order", f"{(approved['approved_qty'] > 0).sum()}", f"of {len(todo)} suggested", ""),
        ("Spend", f"£{spend:,.0f}", f"of £{total:,.0f} suggested", ""),
        ("Urgent covered", f"{(approved['urgent'] & (approved['approved_qty'] > 0)).sum()} of {int(todo['urgent'].sum())}",
         "would run out before a delivery", ""),
        ("Budget left", f"£{max(budget - spend, 0):,.0f}", "", ""),
    ])
    if budget < total:
        st.html('<div class="note">The budget is below the suggested spend, so an optimiser has chosen whole packs to '
                'protect the most margin, giving priority to products that would run out before a delivery arrives.</div>')
    view = approved[approved["urgent"]] if urgent_only else approved
    view = view[["urgent", "description", "abc", "on_hand", "on_order", "suggested_qty", "approved_qty",
                 "pack", "approved_cost", "reason"]]
    box = st.container(border=True)
    box.dataframe(view, width="stretch", hide_index=True, height=500, column_config={
        "urgent": st.column_config.CheckboxColumn("Urgent", width="small"),
        "description": st.column_config.TextColumn("Product", width="medium"),
        "abc": st.column_config.TextColumn("Class", width="small"),
        "on_hand": st.column_config.NumberColumn("On hand", format="%d"),
        "on_order": st.column_config.NumberColumn("On order", format="%d"),
        "suggested_qty": st.column_config.NumberColumn("Suggested", format="%d"),
        "approved_qty": st.column_config.NumberColumn("Within budget", format="%d"),
        "pack": st.column_config.NumberColumn("Pack", format="%d", width="small"),
        "approved_cost": st.column_config.NumberColumn("Cost", format="£%.0f"),
        "reason": st.column_config.TextColumn("Why", width="large")})
    st.download_button("Download order list (CSV)", view.to_csv(index=False), "order_list.csv", "text/csv",
                       type="primary")

# ---------------------------------------------------------------- Product
def product():
    header("Product forecast", "Sales history, how the model and the moving average did on held-out weeks, and the "
                               "forecast for the next four weeks with its likely range.")
    products = q("""SELECT p.stock_code, p.description, c.abc, c.xyz, c.revenue FROM inv_classes c
                    JOIN dim_product p USING (stock_code) ORDER BY c.revenue DESC""")
    label = st.selectbox("Product", products["description"] + "  ·  " + products["stock_code"])
    code = label.rsplit("·", 1)[1].strip()
    row = products.set_index("stock_code").loc[code]
    plan = q(f"SELECT * FROM inv_plan WHERE stock_code = '{code}'")
    fut = q(f"SELECT week, forecast, p10, p90 FROM fc_future WHERE stock_code = '{code}' ORDER BY week")
    p = plan.iloc[0] if not plan.empty else None
    kpis([
        ("Class", f"{row.abc}{row.xyz}",
         {"A": "top 80% of revenue", "B": "next 15% of revenue", "C": "last 5% of revenue"}[row.abc] + " · "
         + {"X": "steady", "Y": "variable", "Z": "erratic"}[row.xyz], ""),
        ("Next week", f"{fut['forecast'].iloc[0]:,.0f} units",
         f"likely {fut['p10'].iloc[0]:,.0f} to {fut['p90'].iloc[0]:,.0f}", ""),
        ("Stock cover", f"{p.weeks_cover:.1f} weeks" if p is not None else "–",
         "order now" if p is not None and p.urgent else "", "bad" if p is not None and p.urgent else ""),
        ("Revenue, last 12 months", f"£{row.revenue:,.0f}", "", ""),
    ])
    hist = q(f"SELECT week, units FROM weekly_demand WHERE stock_code = '{code}' ORDER BY week")
    bt = q(f"SELECT target_week, hgb, moving_avg FROM fc_backtest WHERE stock_code = '{code}' AND h = 1 ORDER BY 1")
    box = card(st, row.description, "Weekly units. The forecast drops over Christmas because the business closes "
                                    "then, which the model learned from the year before.")
    fig = go.Figure()
    fig.add_vrect(x0=bt["target_week"].min(), x1=bt["target_week"].max(), fillcolor="#f1f0eb", opacity=1,
                  line_width=0, layer="below", annotation_text="Held-out test weeks", annotation_position="top left",
                  annotation_font=dict(color=INK2, size=11))
    fig.add_bar(x=hist["week"], y=hist["units"], name="Units sold", marker_color="#cfcdc4",
                hovertemplate="%{y:,.0f}<extra>Sold</extra>")
    fig.add_scatter(x=bt["target_week"], y=bt["moving_avg"], name="Moving average", mode="lines",
                    line=dict(color=ORANGE, width=2, dash="dot"))
    fig.add_scatter(x=bt["target_week"], y=bt["hgb"], name="Model", mode="lines", line=dict(color=BLUE, width=2))
    fig.add_scatter(x=pd.concat([fut["week"], fut["week"][::-1]]), y=pd.concat([fut["p90"], fut["p10"][::-1]]),
                    fill="toself", fillcolor="rgba(42,120,214,0.18)", line=dict(width=0), mode="lines",
                    name="Likely range (80%)", hoverinfo="skip")
    fig.add_scatter(x=fut["week"], y=fut["forecast"], name="Forecast", mode="lines+markers",
                    line=dict(color=BLUE, width=3), marker=dict(size=7))
    fig.update_layout(hovermode="x unified", bargap=0.15)
    plot(box, fig, 420)
    if p is not None:
        st.html(f'<div class="reason"><b>Reorder suggestion: {int(p.suggested_qty):,} units.</b> {p.reason}</div>')

# ---------------------------------------------------------------- Accuracy
def accuracy():
    header("Forecast accuracy", "Every model was tested on the last 24 weeks, forecasting only from data available "
                                "at the time and retrained every four weeks. Error is WAPE: units wrong as a share "
                                "of units sold.")
    m = q("SELECT * FROM fc_metrics")
    m["model"] = m["model"].map(MODEL_NAMES)
    kpis([
        ("Error over reorder window", f"{s.lead_wape_gb:.1%}", f"moving average {s.lead_wape_ma:.1%}", "good"),
        ("Weekly error", f"{s.wape_gb:.1%}", f"moving average {s.wape_ma:.1%}", ""),
        ("Range coverage", f"{s.interval_coverage:.1%}", "target 80%", ""),
        ("Products tested", f"{s.products_forecast:,.0f}", f"{config.BACKTEST_WEEKS} weeks each", ""),
    ])
    left, right = st.columns([2, 3], gap="medium")
    allh = m[m["horizon"] == "all"].sort_values("wape")
    box = card(left, "Model comparison", "Gradient boosting is used: the neural network has lower weekly error but "
                                         "under-forecasts, which would cause stockouts.")
    box.dataframe(allh[["model", "wape", "bias"]], hide_index=True, width="stretch", column_config={
        "model": "Model", "wape": st.column_config.NumberColumn("Weekly error", format="percent"),
        "bias": st.column_config.NumberColumn("Bias", format="percent",
                                              help="Positive means over-forecasting, negative under-forecasting")})
    bt = q("SELECT date_trunc('month', target_week) AS month, sum(actual) a, sum(hgb) g, sum(moving_avg) ma "
           "FROM fc_backtest GROUP BY 1 ORDER BY 1")
    bt["Gradient boosting"] = bt["g"] / bt["a"] - 1
    bt["8-week moving average"] = bt["ma"] / bt["a"] - 1
    box2 = card(right, "Over- or under-forecasting by month", "The moving average falls behind as demand builds in "
                                                              "the autumn; the model stays close to zero.")
    fig = px.bar(bt.melt(id_vars="month", value_vars=["Gradient boosting", "8-week moving average"]),
                 x="month", y="value", color="variable", barmode="group", color_discrete_sequence=[BLUE, GREY],
                 labels={"value": "", "month": ""})
    fig.update_yaxes(tickformat="+.0%")
    fig.update_xaxes(tickformat="%b %Y", dtick="M1")
    plot(box2, fig, 300)

# ---------------------------------------------------------------- Data quality
def data_quality():
    header("Data quality", "The source is an invoice export that was never designed for analysis. Each check records "
                           "what it found, what was done and why, so every number can be traced.")
    dq = q("SELECT * FROM dq_report")
    kept = dq[dq["rule"] == "Clean sales lines kept"].iloc[0]
    kpis([
        ("Raw lines", f"{s.raw_lines:,.0f}", "", ""),
        ("Kept for analysis", f"{kept['rows']:,.0f}", f"{kept['share']:.1%} of raw", "good"),
        ("Missing customer ID", f"{dq[dq['rule'] == 'Missing customer ID']['rows'].iloc[0]:,.0f}",
         "kept for demand only", ""),
        ("Checks applied", f"{len(dq) - 1}", "every removal counted", ""),
    ])
    dq["share"] = dq["share"].map(lambda v: "" if pd.isna(v) else f"{v:.2%}")
    box = card(st, "Checks and fixes")
    box.dataframe(dq[dq["rule"] != "Clean sales lines kept"], hide_index=True, width="stretch", column_config={
        "rule": st.column_config.TextColumn("Check", width="medium"),
        "rows": st.column_config.NumberColumn("Count", format="localized"),
        "share": "Share of lines", "action": st.column_config.TextColumn("Action", width="medium"),
        "why": st.column_config.TextColumn("Why it matters", width="large")})
    ex = q("""SELECT stock_code, description, times_used FROM dq_descriptions WHERE stock_code IN (
                SELECT stock_code FROM dq_descriptions GROUP BY 1 ORDER BY count(*) DESC LIMIT 3)
              ORDER BY stock_code, times_used DESC""")
    box2 = card(st, "Same product, different descriptions", "The most common spelling is used everywhere.")
    box2.dataframe(ex, hide_index=True, width="stretch", column_config={
        "stock_code": "Product code", "description": "Description as typed",
        "times_used": st.column_config.NumberColumn("Lines using it", format="localized")})

# ---------------------------------------------------------------- Customers
def customers():
    header("Customers", "Rule-based segments over the last 12 months, and regular customers who are buying less than "
                        "usual, so the sales team can follow up early.")
    cs = q("SELECT * FROM cust_segments")
    seg = cs.groupby("segment").agg(customers=("customer_id", "count"), revenue=("revenue", "sum"))
    slip = cs[cs["segment"] == "Slipping"].sort_values("expected_90d", ascending=False)
    kpis([
        ("Active customers", f"{len(cs):,}", "ordered in the last 12 months", ""),
        ("Top customers' share", f"{seg.loc['Top customers', 'revenue'] / seg['revenue'].sum():.0%}",
         f"{int(seg.loc['Top customers', 'customers'])} customers", ""),
        ("Slipping customers", f"{len(slip)}",
         f"£{(slip['expected_90d'] - slip['rev_last_90d']).sum():,.0f} below usual in 90 days", "bad"),
        ("New customers", f"{int(seg.loc['New', 'customers'])}", "first order in the last 90 days", ""),
    ])
    left, right = st.columns([2, 3], gap="medium")
    box = card(left, "Revenue by segment", "Last 12 months; labels show the number of customers.")
    seg = seg.reset_index().sort_values("revenue")
    fig = px.bar(seg, x="revenue", y="segment", orientation="h", text="customers", color="segment",
                 color_discrete_map={"Top customers": BLUE, "Regular": "#6ea6e6", "New": GREEN,
                                     "Occasional": GREY, "Slipping": ORANGE, "Lapsed": RED},
                 labels={"revenue": "", "segment": ""})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickprefix="£", range=[0, seg["revenue"].max() * 1.15])
    plot(box, fig, 360)
    box2 = card(right, "Slipping customers to contact", "Regular customers whose spending in the last 90 days is less "
                                                        "than half their usual rate.")
    box2.dataframe(slip[["customer_id", "country", "rev_last_90d", "expected_90d", "days_since_order", "reason"]],
                   hide_index=True, width="stretch", height=330, column_config={
                       "customer_id": st.column_config.NumberColumn("Customer", format="%d"), "country": "Country",
                       "rev_last_90d": st.column_config.NumberColumn("Last 90 days", format="£%.0f"),
                       "expected_90d": st.column_config.NumberColumn("Usual", format="£%.0f"),
                       "days_since_order": st.column_config.NumberColumn("Days since order", format="%d"),
                       "reason": st.column_config.TextColumn("Why", width="large")})


st.navigation([
    st.Page(overview, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page(reorder, title="Reorder this week", icon=":material/shopping_cart:", url_path="reorder"),
    st.Page(product, title="Product forecast", icon=":material/insights:", url_path="product"),
    st.Page(accuracy, title="Forecast accuracy", icon=":material/target:", url_path="accuracy"),
    st.Page(data_quality, title="Data quality", icon=":material/fact_check:", url_path="data_quality"),
    st.Page(customers, title="Customers", icon=":material/groups:", url_path="customers"),
]).run()
