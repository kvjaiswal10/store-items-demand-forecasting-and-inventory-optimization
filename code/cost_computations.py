import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load forecast data ---
forecast_df = pd.read_csv(r'C:\Users\kvjai\ML PROJECTS\Store items demand forecasting\data\forecast.csv', 
                          parse_dates=['date'], dayfirst=True)

# --- Select one SKU ---
store_id, item_id = 1, 1
subset = forecast_df[(forecast_df['store']==store_id) & (forecast_df['item']==item_id)].copy()
subset = subset.sort_values('date').reset_index(drop=True)

# --- Demand stats ---
avg_daily_demand = subset['sales'].mean()
std_daily_demand = subset['sales'].std(ddof=1)
annual_demand = avg_daily_demand * 365

# --- Inventory parameters ---
ordering_cost = 20         # cost per order
holding_cost = 50          # cost per unit per year
shortage_cost_per_day = 5  # penalty per stockout day
lead_time = 7
z = 1.65                   # 95% service level

# --- EOQ & ROP ---
EOQ = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
safety_stock = z * std_daily_demand * np.sqrt(lead_time)
ROP = avg_daily_demand * lead_time + safety_stock

EOQ = min(EOQ, avg_daily_demand * 90)   # cap to 90 days demand
initial_stock = int(ROP + EOQ/2)

# --- Simulation ---
inventory_levels = []
reorder_flags = []
orders_placed = []
stock = initial_stock
order_outstanding = False
arrival_index = None
stockout_days = 0

for i, row in subset.iterrows():
    # Receive order
    if order_outstanding and arrival_index == i:
        stock += EOQ
        order_outstanding = False
        arrival_index = None

    # Demand reduces stock
    stock -= row['sales']

    # Stockout tracking
    if stock < 0:
        stockout_days += 1
        stock = 0  # no negative inventory

    # Check reorder
    if (stock <= ROP) and (not order_outstanding):
        arrival_index = i + lead_time
        orders_placed.append((i, row['date'], int(EOQ), arrival_index))
        order_outstanding = True
        reorder_flags.append(1)
    else:
        reorder_flags.append(0)

    inventory_levels.append(stock)

# Attach results
subset['inventory_level'] = inventory_levels
subset['reorder'] = reorder_flags

# --- Metrics ---
total_orders = len(orders_placed)
avg_inventory = np.mean(subset['inventory_level'])
horizon_days = len(subset)

# --- Cost Calculations ---
ordering_cost_total = total_orders * ordering_cost
holding_cost_total = avg_inventory * (holding_cost / 365) * horizon_days
shortage_cost_total = stockout_days * shortage_cost_per_day

total_cost = ordering_cost_total + holding_cost_total + shortage_cost_total

# --- Service level ---
service_level = 1 - (stockout_days / horizon_days)

# --- Print Summary ---
print(f"Total Orders Placed: {total_orders}")
print(f"Stockout Days: {stockout_days} / {horizon_days}")
print(f"Average Inventory: {avg_inventory:.2f}")
print(f"Service Level: {service_level:.3f}")

print("\n--- Cost Summary ---")
print(f"Ordering Cost Total: ₹{ordering_cost_total:,.2f}")
print(f"Holding Cost Total: ₹{holding_cost_total:,.2f}")
print(f"Shortage Cost Total: ₹{shortage_cost_total:,.2f}")
print(f"✅ Total Estimated Cost: ₹{total_cost:,.2f}")
