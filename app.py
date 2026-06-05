import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="🛒",
    layout="wide",
)

# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    train = pd.read_csv("train.csv", parse_dates=["date"])
    test  = pd.read_csv("test.csv",  parse_dates=["date"])
    sub   = pd.read_csv("submission.csv")
    # submission ids are float in the file — align types
    test["id"] = test["id"].astype(int)
    sub["id"]  = sub["id"].astype(int)
    forecast   = test.merge(sub, on="id")
    return train, forecast

train, forecast = load_data()

STORES = sorted(train["store"].unique())
ITEMS  = sorted(train["item"].unique())

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.title("🔎 Filters")
selected_store = st.sidebar.selectbox("Store", STORES, format_func=lambda x: f"Store {x}")
selected_item  = st.sidebar.selectbox("Item",  ITEMS,  format_func=lambda x: f"Item {x}")

st.sidebar.markdown("---")
st.sidebar.markdown("**History window**")
show_years = st.sidebar.multiselect(
    "Years to show",
    options=sorted(train["year"].unique() if "year" in train.columns else train["date"].dt.year.unique()),
    default=sorted((train["date"].dt.year).unique())[-2:],   # last 2 years by default
)

# ── Filter data ───────────────────────────────────────────────────────────────
mask_train = (train["store"] == selected_store) & (train["item"] == selected_item)
hist = train[mask_train].copy()
hist["year"] = hist["date"].dt.year

if show_years:
    hist_view = hist[hist["year"].isin(show_years)]
else:
    hist_view = hist

mask_fc = (forecast["store"] == selected_store) & (forecast["item"] == selected_item)
fc_view = forecast[mask_fc].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛒 Retail Sales Forecasting Dashboard")
st.markdown(
    f"Showing **Store {selected_store}** · **Item {selected_item}** — "
    f"historical data (2013–2017) + 2018 Q1 forecast"
)

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

avg_hist   = hist["sales"].mean()
avg_fc     = fc_view["sales"].mean()
pct_change = ((avg_fc - avg_hist) / avg_hist) * 100
total_fc   = fc_view["sales"].sum()

col1.metric("Avg Daily Sales (historical)", f"{avg_hist:.1f}")
col2.metric("Avg Daily Sales (forecast)",   f"{avg_fc:.1f}",
            delta=f"{pct_change:+.1f}% vs history")
col3.metric("Total Forecast Sales (Q1 2018)", f"{int(total_fc):,}")
col4.metric("Forecast Period", f"{fc_view['date'].min().date()} → {fc_view['date'].max().date()}")

st.markdown("---")

# ── Main chart: History + Forecast ───────────────────────────────────────────
st.subheader("📈 Sales History & Forecast")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=hist_view["date"], y=hist_view["sales"],
    mode="lines",
    name="Historical Sales",
    line=dict(color="#4C9BE8", width=1.5),
    opacity=0.85,
))

fig.add_trace(go.Scatter(
    x=fc_view["date"], y=fc_view["sales"],
    mode="lines+markers",
    name="Forecast (2018 Q1)",
    line=dict(color="#F97316", width=2.5, dash="dot"),
    marker=dict(size=4),
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Sales",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="#fafafa"),
    xaxis=dict(gridcolor="#333"),
    yaxis=dict(gridcolor="#333"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Monthly aggregation ───────────────────────────────────────────────────────
st.subheader("📊 Monthly Average Sales")

hist_monthly = (
    hist.assign(month=hist["date"].dt.to_period("M").astype(str))
        .groupby("month")["sales"].mean()
        .reset_index()
)
hist_monthly["type"] = "Historical"

fc_monthly = (
    fc_view.assign(month=fc_view["date"].dt.to_period("M").astype(str))
           .groupby("month")["sales"].mean()
           .reset_index()
)
fc_monthly["type"] = "Forecast"

monthly = pd.concat([hist_monthly, fc_monthly], ignore_index=True)

fig2 = px.bar(
    monthly, x="month", y="sales", color="type",
    color_discrete_map={"Historical": "#4C9BE8", "Forecast": "#F97316"},
    labels={"sales": "Avg Daily Sales", "month": "Month", "type": ""},
    height=360,
)
fig2.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="#fafafa"),
    xaxis=dict(gridcolor="#333", tickangle=-45),
    yaxis=dict(gridcolor="#333"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig2, use_container_width=True)

# ── Day-of-week pattern ───────────────────────────────────────────────────────
st.subheader("📅 Sales by Day of Week")

dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

hist_dow = (
    hist.assign(dow=hist["date"].dt.dayofweek)
        .groupby("dow")["sales"].mean()
        .reindex(range(7))
        .reset_index()
)
hist_dow["day"] = hist_dow["dow"].map(lambda d: dow_labels[d])

fc_dow = (
    fc_view.assign(dow=fc_view["date"].dt.dayofweek)
           .groupby("dow")["sales"].mean()
           .reindex(range(7))
           .reset_index()
)
fc_dow["day"] = fc_dow["dow"].map(lambda d: dow_labels[d])

fig3 = go.Figure()
fig3.add_trace(go.Bar(x=hist_dow["day"], y=hist_dow["sales"],
                      name="Historical", marker_color="#4C9BE8"))
fig3.add_trace(go.Bar(x=fc_dow["day"],  y=fc_dow["sales"],
                      name="Forecast",   marker_color="#F97316"))
fig3.update_layout(
    barmode="group",
    xaxis_title="Day of Week", yaxis_title="Avg Sales",
    height=340,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="#fafafa"),
    xaxis=dict(gridcolor="#333"),
    yaxis=dict(gridcolor="#333"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig3, use_container_width=True)

# ── Cross-store / cross-item heatmap ─────────────────────────────────────────
st.markdown("---")
st.subheader("🗺️ Forecast Heatmap — All Stores × Items (Q1 2018 avg)")

pivot = (
    forecast.groupby(["store", "item"])["sales"]
            .mean()
            .reset_index()
            .pivot(index="store", columns="item", values="sales")
)

fig4 = px.imshow(
    pivot,
    labels=dict(x="Item", y="Store", color="Avg Daily Sales"),
    color_continuous_scale="Blues",
    aspect="auto",
    height=420,
)
fig4.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="#0e1117",
    font=dict(color="#fafafa"),
)
st.plotly_chart(fig4, use_container_width=True)

# ── Raw forecast table ────────────────────────────────────────────────────────
with st.expander("📋 View Raw Forecast Data"):
    st.dataframe(
        fc_view[["date", "store", "item", "sales"]]
          .rename(columns={"date": "Date", "store": "Store",
                            "item": "Item", "sales": "Forecasted Sales"})
          .reset_index(drop=True),
        use_container_width=True,
        height=300,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: LightGBM · Features: calendar + 365-day lag · Validation RMSE: 8.20")
