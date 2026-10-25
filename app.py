"""
Dark Store Location Optimizer — Chennai
AS 5570 Assignment 1

Profit-maximizing facility location model:
  - choose which candidate dark stores to open (<= K of them)
  - optionally serve each customer from an open store within D_max km
  - maximize revenue - transport cost - fixed cost

See README.md for the full writeup of assumptions, formulation, and the
feasibility bug found (and fixed) in the earlier draft of this model.
"""

import pandas as pd
import streamlit as st
import altair as alt
import folium
from streamlit_folium import st_folium

from core import (
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    generate_data as _generate_data,
    compute_distance_matrix as _compute_distance_matrix,
    solve_model as _solve_model,
    run_sensitivity as _run_sensitivity,
)

st.set_page_config(page_title="Dark Store Location Optimizer", layout="wide", page_icon="🏪")

# Thin, cached wrappers around core.py so the optimization logic itself stays
# framework-free and unit-testable (see test_solver.py).
generate_data = st.cache_data(_generate_data)
compute_distance_matrix = st.cache_data(_compute_distance_matrix)
solve_model = _solve_model
run_sensitivity = st.cache_data(show_spinner=False)(_run_sensitivity)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🏪 Dark Store Location Optimizer — Chennai")
st.caption(
    "Profit-maximizing facility location with a maximum service-distance constraint. "
    "Customers are served only if an open store is within the allowed distance and doing so is "
    "included in the optimal assignment."
)

with st.sidebar:
    st.header("Input Parameters")
    n_customers = st.number_input("Number of customers", min_value=10, max_value=10000, value=3000, step=50)
    n_stores = st.number_input("Number of candidate dark stores", min_value=2, max_value=50, value=30, step=1)
    max_dist = st.slider("Maximum distance D_max (km)", min_value=0.5, max_value=10.0, value=4.0, step=0.5)
    max_open = st.slider("Max dark stores to open (K)", min_value=1, max_value=int(n_stores),
                          value=min(10, int(n_stores)))

    st.subheader("Cost / Revenue Assumptions (monthly)")
    fixed_cost = st.number_input("Fixed cost per store / month (Rs.)", min_value=10000, value=150000, step=10000)
    revenue = st.number_input("Revenue per served customer / month (Rs.)", min_value=10, value=1800, step=50)
    transp_cost = st.number_input("Transport cost per km per customer / month (Rs.)", min_value=1, value=60, step=5)
    st.caption(
        "Defaults are calibrated as MONTHLY figures (recurring customer, aggregated over repeat "
        "deliveries), not a single order: see README.md 'Calibration note' for why a single-order "
        "margin (e.g. Rs.120 revenue, Rs.8/km transport) makes every store unprofitable at this "
        "customer density and D_max, which the corrected model correctly reports as 'open nothing'."
    )

    st.caption(f"Geography: Chennai bounding box, lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]. "
               "Distance metric: haversine (great-circle), a road-network proxy.")

customers, stores = generate_data(int(n_customers), int(n_stores))
D = compute_distance_matrix(customers, stores)

run = st.button("🚀 Optimize", type="primary")

if run:
    with st.spinner("Solving..."):
        result = solve_model(D, max_dist, int(max_open), fixed_cost, revenue, transp_cost)

    if result["status"] != "Optimal":
        st.error(f"Solver returned status: {result['status']}. No solution to display.")
    else:
        st.subheader("📊 Key Performance Indicators")
        kpi_df = pd.DataFrame({
            "Metric": [
                "Profit (Rs.)", "Revenue (Rs.)", "Transport Cost (Rs.)", "Fixed Cost (Rs.)",
                "Dark Stores Opened", "Customers Served", "Customers Unserved",
                "Service Rate",
            ],
            "Value": [
                f"{result['profit']:,.0f}",
                f"{result['revenue']:,.0f}",
                f"{result['transport_cost']:,.0f}",
                f"{result['fixed_cost']:,.0f}",
                f"{result['num_open']} / {int(n_stores)}",
                f"{result['num_served']} / {int(n_customers)}",
                f"{int(n_customers) - result['num_served']}",
                f"{result['num_served'] / int(n_customers):.1%}",
            ],
        })
        st.table(kpi_df)

        if result["num_served"] < int(n_customers):
            st.info(
                f"{int(n_customers) - result['num_served']} customer(s) are not served in the optimal "
                "solution -- either no open store is within D_max, or serving them would not be "
                "profitable / was not selected under the store-count limit K. This is expected "
                "behavior for a profit-maximizing (not coverage-maximizing) model."
            )

        st.subheader("🗺️ Solution Map")
        map_center = [customers["lat"].mean(), customers["lon"].mean()]
        m = folium.Map(location=map_center, zoom_start=11, tiles="OpenStreetMap")

        # Cap markers plotted for responsiveness on large instances.
        MAX_MARKERS = 1500
        if len(customers) > MAX_MARKERS:
            plot_idx = customers.sample(MAX_MARKERS, random_state=0).index
            st.caption(f"Showing a random sample of {MAX_MARKERS:,} of {len(customers):,} customers on the map "
                       "for rendering performance; KPIs above reflect all customers.")
        else:
            plot_idx = customers.index

        served_idx = set(result["assignments"].keys())
        for idx in plot_idx:
            row = customers.loc[idx]
            is_served = idx in served_idx
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=2,
                color="#2a7de1" if is_served else "#999999",
                fill=True, fill_opacity=0.6, weight=0,
            ).add_to(m)

        for j, row in stores.iterrows():
            opened = j in result["open_stores"]
            folium.Marker(
                location=[row["lat"], row["lon"]],
                icon=folium.Icon(color="green" if opened else "gray", icon="home"),
                popup=f"Store {j} {'(OPEN)' if opened else '(candidate, not opened)'}",
            ).add_to(m)

        line_idx = [i for i in result["assignments"] if i in set(plot_idx)]
        for i in line_idx:
            j = result["assignments"][i]
            folium.PolyLine(
                locations=[
                    [customers.iloc[i]["lat"], customers.iloc[i]["lon"]],
                    [stores.iloc[j]["lat"], stores.iloc[j]["lon"]],
                ],
                color="green", weight=0.6, opacity=0.35,
            ).add_to(m)

        st_folium(m, width=None, height=520, use_container_width=True, key="solution_map")
        st.caption("Green homes = opened stores · gray homes = unopened candidates · "
                   "blue dots = served customers · gray dots = unserved customers.")
else:
    st.info("Set parameters in the sidebar and click **Optimize** to run the model.")

# ----------------------------------------------------------------------------
# Required sensitivity analysis: 10,000 customers, 50 candidates, K in {2,5,10,20,25}
# ----------------------------------------------------------------------------
st.divider()
st.subheader("📈 Profit vs. Max Open Stores — 10,000 customers, 50 candidate stores")
st.caption("Fixed benchmark instance (independent of the sidebar controls above): "
           "D_max = 4 km, fixed cost Rs. 1,50,000/store/month, revenue Rs. 1,800/customer/month, "
           "transport Rs. 60/km/customer/month -- see README.md for the calibration reasoning.")

BENCH_FC, BENCH_REV, BENCH_TC = 150000, 1800, 60

with st.spinner("Running sensitivity sweep for K = 2, 5, 10, 20, 25 (10,000 x 50 instance)..."):
    sens_df = run_sensitivity(
        n_customers=10000, n_stores=50, max_dist=4.0,
        fixed_cost=BENCH_FC, revenue=BENCH_REV, transp_cost=BENCH_TC,
        k_values=(2, 5, 10, 20, 25),
    )

chart = alt.Chart(sens_df).mark_line(point=True, strokeWidth=2).encode(
    x=alt.X("Max Open (K):O", title="Max dark stores allowed (K)"),
    y=alt.Y("Profit (Rs.):Q", title="Profit (Rs.)", axis=alt.Axis(format="~s")),
    tooltip=["Max Open (K)", alt.Tooltip("Profit (Rs.):Q", format=",.0f"),
             "Stores Opened", "Customers Served", "Status"],
).properties(height=380)
st.altair_chart(chart, width="stretch")
st.dataframe(sens_df, width="stretch", hide_index=True)

with st.expander("How to read this curve"):
    st.markdown(
        "- Profit rises steadily across K = 2 -> 25 because, at this store density (50 candidates "
        "over the whole Chennai bounding box with D_max = 4 km), coverage is still the binding "
        "constraint, not store profitability: each additional open store reaches new customers "
        "that were previously out of range of any open store, and every one of those customers "
        "is individually profitable to serve.\n"
        "- It does **not** plateau within the required K in {2,5,10,20,25} range -- running the same "
        "model further (not required by the assignment, included here for completeness) shows it "
        "actually plateaus around K ~ 30-34, where only 34 of the 50 candidate locations are ever "
        "worth opening; beyond that, remaining candidates overlap existing coverage or sit in sparse "
        "areas and add no profitable customers.\n"
        "- With D_max = 4 km and only 50 randomly placed candidates over this bounding box, about "
        "22% of customers have no candidate store within range at all, at any K -- so even full "
        "build-out doesn't reach 100% of customers.\n"
        "- Customers who cannot be reached, or whom it would not be profitable to serve, are left "
        "unserved rather than forcing an infeasible solution -- this is the feasibility fix described "
        "in README.md."
    )
