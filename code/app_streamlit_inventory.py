# app_streamlit_inventory.py
# Streamlit What-if Optimization Dashboard
# Expiry-aware EOQ & ROP inventory simulation (multi-SKU)
# Run: streamlit run app_streamlit_inventory.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import sqrt

st.set_page_config(layout="wide", page_title="What-if Inventory Dashboard", initial_sidebar_state="expanded")

# -------------------------
# Utility / Simulation Code
# -------------------------
@st.cache_data
def load_forecast(path: str):
    df = pd.read_csv(path, parse_dates=["date"], dayfirst=True)
    # ensure types
    df["store"] = df["store"].astype(int)
    df["item"] = df["item"].astype(int)
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return df

def default_expiry_map(unique_items):
    # Example default mapping; you should tune/replace with domain info or CSV
    # shorter life for small item ids as example, else 90 days default
    mapping = {}
    for it in unique_items:
        if it % 7 == 0:
            mapping[it] = 30
        elif it % 5 == 0:
            mapping[it] = 45
        elif it % 3 == 0:
            mapping[it] = 60
        else:
            mapping[it] = 90
    return mapping

def compute_EOQ(D_annual, ordering_cost, holding_cost, cap_days, avg_daily):
    if D_annual <= 0 or holding_cost <= 0:
        return max(1, avg_daily)  # fallback
    EOQ = sqrt((2 * D_annual * ordering_cost) / holding_cost)
    EOQ = min(EOQ, avg_daily * cap_days)
    return float(max(1.0, EOQ))

def simulate_expiry_eoq(subset_df,
                        expiry_days:int,
                        ordering_cost:float,
                        holding_cost:float,
                        shortage_cost_per_day:float,
                        waste_cost_per_unit:float,
                        lead_time:int,
                        z:float,
                        cap_eoq_days:int=90,
                        allow_multiple_outstanding:bool=False):
    """
    Expiry-aware, FIFO consumption, EOQ+ROP simulation for a single SKU subset_df (sorted by date).
    Returns: dict with timeseries + summary metrics.
    """
    subset = subset_df.sort_values("date").reset_index(drop=True).copy()
    n_days = len(subset)
    avg_daily = subset["sales"].mean()
    std_daily = subset["sales"].std(ddof=1) if n_days > 1 else 0.0
    D_annual = avg_daily * 365.0

    # EOQ, safety stock, ROP
    EOQ = compute_EOQ(D_annual, ordering_cost, holding_cost, cap_eoq_days, avg_daily)
    safety_stock = z * std_daily * np.sqrt(max(1, lead_time))
    ROP = avg_daily * lead_time + safety_stock

    # Start stock near steady state
    initial_stock = int(max(ROP + EOQ/2, 2*ROP))

    # simulation state
    inventory_batches = [[initial_stock, expiry_days]]  # [qty, expiry_index]
    inventory_levels = []
    reorder_flags = []
    orders_placed = []  # (order_day_idx, qty, arrival_idx)
    waste_units = 0.0
    stockout_days = 0
    unmet_units = 0.0

    order_outstanding = False
    on_order = []  # allow queue of outstanding orders: (arrival_idx, qty)

    for i, row in subset.iterrows():
        # Receive arrivals (handle multiple)
        arrivals = [o for o in on_order if o[0] == i]
        if arrivals:
            qty_arrived = sum(q for (_, q) in arrivals)
            inventory_batches.append([qty_arrived, i + expiry_days])
            on_order = [o for o in on_order if o[0] != i]

        # Remove expired batches
        expired = [b for b in inventory_batches if b[1] <= i]
        if expired:
            waste_units += sum(b[0] for b in expired)
            inventory_batches = [b for b in inventory_batches if b[1] > i]

        # Demand (FIFO)
        demand = float(row["sales"])
        while demand > 0 and inventory_batches:
            batch_qty, expiry_idx = inventory_batches[0]
            if batch_qty <= demand + 1e-9:
                demand -= batch_qty
                inventory_batches.pop(0)
            else:
                inventory_batches[0][0] -= demand
                demand = 0.0
        if demand > 0:
            # unmet
            stockout_days += 1
            unmet_units += demand
            demand = 0.0

        current_stock = sum(b[0] for b in inventory_batches)
        inventory_levels.append(current_stock)

        # Place order if below ROP
        outstanding_qty = sum(q for (_, q) in on_order)
        if (current_stock <= ROP) and (not on_order or allow_multiple_outstanding):
            arrival_idx = i + lead_time
            on_order.append((arrival_idx, EOQ))
            orders_placed.append((i, EOQ, arrival_idx))
            reorder_flags.append(1)
        else:
            reorder_flags.append(0)

    # metrics
    total_orders = len(orders_placed)
    avg_inventory = np.mean(inventory_levels) if len(inventory_levels)>0 else 0.0
    horizon_days = len(inventory_levels)
    service_level_by_volume = 1.0 - (unmet_units / (subset["sales"].sum() + 1e-9))
    waste_pct = (waste_units / (subset["sales"].sum() + waste_units + 1e-9)) * 100.0

    # cost components
    ordering_cost_total = total_orders * ordering_cost
    holding_cost_total = avg_inventory * (holding_cost / 365.0) * horizon_days
    shortage_cost_total = stockout_days * shortage_cost_per_day
    waste_cost_total = waste_units * waste_cost_per_unit
    total_cost = ordering_cost_total + holding_cost_total + shortage_cost_total + waste_cost_total

    # prepare timeseries df
    timeseries = subset.copy()
    timeseries["inventory_level"] = inventory_levels
    timeseries["reorder_flag"] = reorder_flags

    summary = {
        "EOQ": round(EOQ,2),
        "ROP": round(ROP,2),
        "total_orders": int(total_orders),
        "avg_inventory": round(avg_inventory,2),
        "stockout_days": int(stockout_days),
        "unmet_units": round(unmet_units,2),
        "service_level": round(service_level_by_volume,4),
        "waste_units": round(waste_units,2),
        "waste_percent": round(waste_pct,3),
        "ordering_cost": round(ordering_cost_total,2),
        "holding_cost": round(holding_cost_total,2),
        "shortage_cost": round(shortage_cost_total,2),
        "waste_cost": round(waste_cost_total,2),
        "total_cost": round(total_cost,2),
        "n_days": horizon_days
    }

    return timeseries, summary, orders_placed

# -------------------------
# Sidebar: load & parameters
# -------------------------
st.sidebar.title("What-if Inventory Dashboard")
st.sidebar.markdown("Upload `forecast.csv` or use the project's file path.")

default_path = "../data/forecast.csv"  
uploaded_file = st.sidebar.file_uploader("Upload forecast.csv (optional)", type=["csv"])

if uploaded_file is None:
    try:
        df = load_forecast(default_path)
    except Exception as e:
        st.sidebar.error(f"Could not load default path: {default_path}. Please upload forecast.csv. Err: {e}")
        st.stop()
else:
    df = pd.read_csv(uploaded_file, parse_dates=["date"], dayfirst=True)
    df["store"] = df["store"].astype(int)
    df["item"] = df["item"].astype(int)
    df = df.sort_values(["store","item","date"]).reset_index(drop=True)

# expiry map (default)
unique_items = sorted(df["item"].unique().tolist())
expiry_map = default_expiry_map(unique_items)

# Option to upload expiry mapping file
st.sidebar.markdown("### Expiry mapping (optional)")
expiry_upload = st.sidebar.file_uploader("Upload CSV with columns: item, expiry_days (optional)", type=["csv"], key="expiry")
if expiry_upload is not None:
    expiry_df = pd.read_csv(expiry_upload)
    if {"item","expiry_days"}.issubset(expiry_df.columns):
        for _, r in expiry_df.iterrows():
            expiry_map[int(r["item"])] = int(r["expiry_days"])
        st.sidebar.success("Expiry mapping loaded")
    else:
        st.sidebar.error("Expiry CSV must have columns: item, expiry_days")

# sidebar controls
st.sidebar.markdown("---")
st.sidebar.markdown("### Simulation Parameters")
ordering_cost = st.sidebar.number_input("Ordering cost per order (S)", value=20.0, min_value=0.0, step=1.0)
holding_cost = st.sidebar.number_input("Holding cost per unit per year (H)", value=50.0, min_value=0.0, step=1.0)
shortage_cost = st.sidebar.number_input("Shortage cost per stockout day", value=5.0, min_value=0.0, step=1.0)
waste_cost = st.sidebar.number_input("Waste cost per expired unit", value=1.0, min_value=0.0, step=0.1)
lead_time = st.sidebar.slider("Lead time (days)", min_value=0, max_value=30, value=7, step=1)
z = st.sidebar.slider("Safety factor z (service level)", min_value=0.0, max_value=3.0, value=1.65, step=0.05)
eoq_cap_days = st.sidebar.slider("Max days to cap EOQ", min_value=30, max_value=365, value=90, step=10)
allow_multi = st.sidebar.checkbox("Allow multiple outstanding orders", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### What-if sweep (safety factor grid)")
sweep_min = st.sidebar.number_input("z min", value=1.0, min_value=0.0, step=0.05)
sweep_max = st.sidebar.number_input("z max", value=2.5, min_value=0.0, step=0.05)
sweep_steps = st.sidebar.number_input("z steps", value=16, min_value=2, step=1)

# -------------------------
# Main layout
# -------------------------
st.title("🔮 What-if Inventory Optimization Dashboard (Expiry-aware)")
st.markdown(
    """Use the controls on the left to test reorder policies (EOQ + ROP) interactively.
    Select store and item, try different lead times, safety factors and costs, and observe
    service level, waste, and total cost. The dashboard recommends a z (safety factor)
    based on a sweep that balances cost vs service level."""
)

# SKU selector
stores = sorted(df["store"].unique().tolist())
selected_store = st.selectbox("Select Store", stores, index=0)
items_for_store = sorted(df[df["store"]==selected_store]["item"].unique().tolist())
selected_item = st.selectbox("Select Item (SKU)", items_for_store)

# subset of forecast
subset = df[(df["store"]==selected_store) & (df["item"]==selected_item)].sort_values("date").reset_index(drop=True).copy()
expiry_days_for_item = expiry_map.get(selected_item, 90)
subset["expiry_days"] = expiry_days_for_item

# run simulation button
if st.button("Run Simulation"):
    with st.spinner("Running expiry-aware simulation..."):
        ts, summary, orders = simulate_expiry_eoq(
            subset,
            expiry_days=expiry_days_for_item,
            ordering_cost=ordering_cost,
            holding_cost=holding_cost,
            shortage_cost_per_day=shortage_cost,
            waste_cost_per_unit=waste_cost,
            lead_time=lead_time,
            z=z,
            cap_eoq_days=eoq_cap_days,
            allow_multiple_outstanding=allow_multi
        )

    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("EOQ", f"{summary['EOQ']}")
    col2.metric("ROP", f"{summary['ROP']}")
    col3.metric("Service Level", f"{summary['service_level']*100:.2f}%")
    col4.metric("Total Cost (₹)", f"{summary['total_cost']:,}")

    # small summary text
    st.markdown(f"**SKU:** Store {selected_store} — Item {selected_item}  •  Expiry = {expiry_days_for_item} days  •  Horizon = {summary['n_days']} days")

    # time series plot: demand vs inventory
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=ts["date"], y=ts["sales"], mode="lines+markers", name="Forecast Demand", line=dict(color="royalblue")))
    fig_ts.add_trace(go.Scatter(x=ts["date"], y=ts["inventory_level"], mode="lines+markers", name="Inventory Level", line=dict(color="orange")))
    # mark reorder days
    reorder_days = ts[ts["reorder_flag"]==1]["date"].tolist()
    for d in reorder_days:
        fig_ts.add_vline(x=d, line=dict(color="purple", dash="dash"), opacity=0.5)
    fig_ts.update_layout(title="Forecast Demand & Inventory Level", xaxis_title="Date", yaxis_title="Units", legend=dict(orientation="h"))
    st.plotly_chart(fig_ts, use_container_width=True)

    # Cost breakdown pie
    cost_values = [summary['ordering_cost'], summary['holding_cost'], summary['shortage_cost'], summary['waste_cost']]
    cost_labels = ["Ordering", "Holding", "Shortage", "Waste"]
    fig_pie = px.pie(values=cost_values, names=cost_labels, title="Cost Composition (₹)", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Detailed metrics table
    detail_df = pd.DataFrame([summary])
    st.dataframe(detail_df.T.rename(columns={0:"Value"}), use_container_width=True)

    # What-if sweep: grid for z
    st.markdown("### 🔁 What-if Sweep: Safety Factor (z) vs Total Cost & Service Level")
    zs = np.linspace(sweep_min, sweep_max, int(sweep_steps))
    sweep_results = []
    for z_val in zs:
        _, ssum, _ = simulate_expiry_eoq(
            subset,
            expiry_days=expiry_days_for_item,
            ordering_cost=ordering_cost,
            holding_cost=holding_cost,
            shortage_cost_per_day=shortage_cost,
            waste_cost_per_unit=waste_cost,
            lead_time=lead_time,
            z=z_val,
            cap_eoq_days=eoq_cap_days,
            allow_multiple_outstanding=allow_multi
        )
        sweep_results.append({"z": z_val, "total_cost": ssum["total_cost"], "service_level": ssum["service_level"]})

    sweep_df = pd.DataFrame(sweep_results)
    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(x=sweep_df["z"], y=sweep_df["total_cost"], mode="lines+markers", name="Total Cost (₹)", yaxis="y1"))
    fig_sweep.add_trace(go.Scatter(x=sweep_df["z"], y=sweep_df["service_level"]*100, mode="lines+markers", name="Service Level (%)", yaxis="y2"))
    fig_sweep.update_layout(
        title="Safety Factor sweep",
        xaxis=dict(title="z (safety factor)"),
        yaxis=dict(title="Total Cost (₹)", side="left"),
        yaxis2=dict(title="Service Level (%)", overlaying="y", side="right", range=[0,100])
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    # Recommendation: choose z with low cost while service_level >= target (e.g., 0.95)
    target_service = 0.95
    feasible = sweep_df[sweep_df["service_level"] >= target_service]
    if not feasible.empty:
        best = feasible.loc[feasible["total_cost"].idxmin()]
        st.success(f"Recommendation: choose z = {best['z']:.2f} → Total Cost ₹{best['total_cost']:.2f} with Service Level {best['service_level']*100:.2f}%")
    else:
        best = sweep_df.loc[sweep_df["total_cost"].idxmin()]
        st.warning(f"No z gives service >= {target_service*100:.0f}%. Choose z = {best['z']:.2f} to minimize cost (Service {best['service_level']*100:.2f}%).")

    # Export results for this SKU
    st.download_button(
        label="Download timeseries (CSV) for this SKU",
        data=ts.to_csv(index=False).encode('utf-8'),
        file_name=f"timeseries_store{selected_store}_item{selected_item}.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download summary (CSV)",
        data=detail_df.to_csv(index=False).encode('utf-8'),
        file_name=f"summary_store{selected_store}_item{selected_item}.csv",
        mime="text/csv"
    )

else:
    st.info("Select a Store & Item and click 'Run Simulation' to start. Try different sliders on the left to test what-if scenarios.")
    st.markdown("**Tips:** Increase `z` to reduce stockouts at the cost of higher holding; increase `lead time` to model slower suppliers; use the sweep to find a recommended `z`.")

# -------------------------
# Footer / novelty notes
# -------------------------
st.markdown("---")
st.markdown("### Novelty & Features included")
st.markdown("""
- Expiry-aware FIFO inventory simulation (FEFO/expiry tracking per batch).
- Integrated forecast-driven EOQ + ROP computations.
- Interactive what-if sliders for lead time, safety factor, ordering/holding/waste costs.
- Safety-factor sweep recommending a cost-aware `z` while meeting service level target.
- Multi-SKU selection and downloadable time series & summary.
""")

st.markdown("Developed for capstone project: Store Item Demand Forecasting & Inventory Assistance — brings forecasting into operational ordering decisions.")
