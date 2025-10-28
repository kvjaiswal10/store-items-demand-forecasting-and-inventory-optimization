# Inventory simulation - robust version (single SKU)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- load forecast.csv (your file) ---
forecast_df = pd.read_csv(
    r'C:\Users\kvjai\ML PROJECTS\Store items demand forecasting\data\forecast.csv',
    parse_dates=['date'],
    dayfirst=True
)

# --- select SKU (store, item) ---
store_id, item_id = 1, 1
subset = forecast_df[(forecast_df['store'] == store_id) & (forecast_df['item'] == item_id)].copy()
subset = subset.sort_values('date').reset_index(drop=True)

# --- demand stats ---
avg_daily_demand = subset['sales'].mean()
std_daily_demand = subset['sales'].std(ddof=1)
annual_demand = avg_daily_demand * 365

# --- inventory economics (tune these) ---
ordering_cost = 20     # S (currency per order)
holding_cost = 10      # H (currency per unit per year) -- lower H => larger EOQ
lead_time = 7          # L (days)
z = 1.65               # service-level z-score (1.65 ~ 95%)
max_days_to_order = 90 # cap EOQ to at most X days of demand

# --- compute EOQ, safety stock, ROP ---
EOQ = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
EOQ = float(min(EOQ, avg_daily_demand * max_days_to_order))  # cap if necessary
safety_stock = z * std_daily_demand * np.sqrt(lead_time)
ROP = avg_daily_demand * lead_time + safety_stock

# --- initial stock (start near steady state) ---
initial_stock = int(max(ROP + EOQ/2, 2*ROP))  # safer start: at least 2*ROP

print(f"Avg daily demand: {avg_daily_demand:.2f}")
print(f"Std daily demand: {std_daily_demand:.2f}")
print(f"EOQ: {EOQ:.0f}, ROP: {ROP:.0f}, initial_stock: {initial_stock}")

# --- simulation containers ---
inventory_levels = []
reorder_flags = []
orders_placed = []    # list of dicts: {order_day_idx, order_date, qty, arrival_idx}
on_order = []         # queue of (arrival_idx, qty)
stock = float(initial_stock)
unmet_demand = 0.0    # total units not satisfied
stockout_days = 0     # count days when demand > available stock
allow_negative_stock = False  # set True if you permit backorders; False => no negative stock

# --- day-by-day simulation ---
for i, row in subset.iterrows():
    # 1) receive any arriving orders
    arrivals = [tq for tq in on_order if tq[0] == i]
    if arrivals:
        qty_arrived = sum(q for (_, q) in arrivals)
        stock += qty_arrived
        # remove arrivals from queue
        on_order = [tq for tq in on_order if tq[0] != i]
    # 2) demand arrives
    demand = float(row['sales'])
    if stock >= demand:
        stock -= demand
    else:
        # not enough stock
        unmet = demand - stock
        unmet_demand += unmet
        stockout_days += 1
        if allow_negative_stock:
            stock -= demand  # allows negative -> backorders
        else:
            stock = 0.0      # cannot go below zero; unmet demand recorded

    # 3) check reorder condition and place order(s)
    # Option A: if no outstanding order allow placing one (simple policy)
    # Option B: allow multiple outstanding orders (comment block A and uncomment B)
    outstanding_qty = sum(q for (_, q) in on_order)
    if (stock <= ROP) and (len(on_order) == 0):
        arrival_idx = i + lead_time
        on_order.append((arrival_idx, EOQ))
        orders_placed.append({
            "order_idx": i,
            "order_date": row['date'],
            "qty": EOQ,
            "arrival_idx": arrival_idx,
            "arrival_date": subset['date'].iloc[arrival_idx] if arrival_idx < len(subset) else None
        })
        reorder_flags.append(1)
    else:
        reorder_flags.append(0)

    inventory_levels.append(stock)

# --- attach results safely using .loc to avoid warnings ---
subset.loc[:, 'inventory_level'] = inventory_levels
subset.loc[:, 'reorder'] = reorder_flags

# --- metrics ---
total_orders = len(orders_placed)
avg_inventory = float(np.mean(subset['inventory_level']))
h_days = len(subset)
service_level = 1 - (unmet_demand / subset['sales'].sum()) if subset['sales'].sum() > 0 else np.nan

print("Total orders placed in horizon:", total_orders)
print("Average inventory level:", round(avg_inventory, 1))
print("Total unmet demand (units):", round(unmet_demand, 2))
print("Stockout days:", stockout_days, "/", h_days)
print("Approx service level (by volume):", round(service_level, 4))

# --- plotting (limit window to first N days for clarity) ---
plot_n = min(120, len(subset))
plot_df = subset.iloc[:plot_n].copy()

plt.figure(figsize=(12,5))
plt.plot(plot_df['date'], plot_df['sales'], label='Forecasted Demand', alpha=0.7)
plt.plot(plot_df['date'], plot_df['inventory_level'], label='Inventory Level', color='orange', linewidth=2)
plt.axhline(ROP, color='red', linestyle='--', label=f'ROP ≈ {ROP:.0f}')
# mark order placement dates
for od in orders_placed:
    if od['order_idx'] < plot_n:
        plt.axvline(od['order_date'], color='purple', linestyle=':', alpha=0.6)
    if od['arrival_idx'] < plot_n:
        arrival_date = od['arrival_date']
        if arrival_date is not None:
            plt.scatter(arrival_date, max(plt.ylim())*0.05, color='green', s=25)  # arrival marker
plt.title(f'Inventory Simulation (Store {store_id}, Item {item_id}) - first {plot_n} days')
plt.xlabel('Date'); plt.ylabel('Units')
plt.legend(); plt.xticks(rotation=45); plt.grid(True); plt.tight_layout()
plt.show()

# --- preview subset head/tail ---
subset.head(8)
subset.tail(8)
