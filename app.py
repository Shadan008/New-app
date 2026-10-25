
"""
Dark Store Location Optimizer — Chennai

Profit-maximizing facility location model:
- Choose which candidate dark stores to open (at most K).
- Serve customers only from an open store within D_max km.
- Maximize revenue - transport cost - fixed cost.

Requires core.py to contain the optimization functions.
"""

import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt

from core import (
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    generate_data as _generate_data,
    compute_distance_matrix as _compute_distance_matrix,
    solve_model as _solve_model,
    run_sensitivity as _run_sensitivity,
)

st.set_page_config(
    page_title="Dark Store Location Optimizer",
    layout="wide",
    page_icon="🏪",
)

# Cache data generation and sensitivity calculations.
generate_data = st.cache_data(_generate_data)
compute_distance_matrix = st.cache_data(_compute_distance_matrix)
solve_model = _solve_model
run_sensitivity = st.cache_data(show_spinner=False)(_run_sensitivity)


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.title("🏪 Dark Store Location Optimizer — Chennai")

st.caption(
    "Profit-maximizing facility location with a maximum service-distance "
    "constraint. Customers can be served only when an open store is within "
    "the allowed distance and the assignment is included in the optimum."
)

with st.sidebar:
    st.header("Input Parameters")

    n_customers = st.number_input(
        "Number of customers",
        min_value=10,
        max_value=10000,
        value=3000,
        step=50,
    )

    n_stores = st.number_input(
        "Number of candidate dark stores",
        min_value=2,
        max_value=50,
        value=30,
        step=1,
    )

    max_dist = st.slider(
        "Maximum distance D_max (km)",
        min_value=0.5,
        max_value=10.0,
        value=4.0,
        step=0.5,
    )

    max_open = st.slider(
        "Max dark stores to open (K)",
        min_value=1,
        max_value=int(n_stores),
        value=min(10, int(n_stores)),
    )

    st.subheader("Cost / Revenue Assumptions (monthly)")

    fixed_cost = st.number_input(
        "Fixed cost per store / month (Rs.)",
        min_value=10000,
        value=150000,
        step=10000,
    )

    revenue = st.number_input(
        "Revenue per served customer / month (Rs.)",
        min_value=10,
        value=1800,
        step=50,
    )

    transp_cost = st.number_input(
        "Transport cost per km per customer / month (Rs.)",
        min_value=1,
        value=60,
        step=5,
    )

    st.caption(
        "The financial assumptions are monthly figures. See README.md "
        "for the model's calibration assumptions."
    )

    st.caption(
        f"Geography: Chennai bounding box, latitude "
        f"[{LAT_MIN}, {LAT_MAX}], longitude [{LON_MIN}, {LON_MAX}]. "
        "Distance metric: haversine (great-circle distance)."
    )


# ============================================================
# GENERATE CUSTOMER AND STORE DATA
# ============================================================

customers, stores = generate_data(
    int(n_customers),
    int(n_stores),
)

D = compute_distance_matrix(customers, stores)

run = st.button("🚀 Optimize", type="primary")


# ============================================================
# OPTIMIZATION RESULTS AND MAP
# ============================================================

if run:
    with st.spinner("Solving the optimization model..."):
        result = solve_model(
            D,
            max_dist,
            int(max_open),
            fixed_cost,
            revenue,
            transp_cost,
        )

    if result["status"] != "Optimal":
        st.error(
            f"Solver returned status: {result['status']}. "
            "No solution to display."
        )

    else:
        st.subheader("📊 Key Performance Indicators")

        kpi_df = pd.DataFrame(
            {
                "Metric": [
                    "Profit (Rs.)",
                    "Revenue (Rs.)",
                    "Transport Cost (Rs.)",
                    "Fixed Cost (Rs.)",
                    "Dark Stores Opened",
                    "Customers Served",
                    "Customers Unserved",
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
            }
        )

        st.dataframe(
            kpi_df,
            use_container_width=True,
            hide_index=True,
        )

        if result["num_served"] < int(n_customers):
            unserved = int(n_customers) - result["num_served"]

            st.info(
                f"{unserved:,} customers are unserved. They may be outside "
                "the maximum service distance or not profitable to serve "
                "under the current store-count constraint."
            )

        # ----------------------------------------------------
        # SOLUTION MAP
        # ----------------------------------------------------

        st.subheader("🗺️ Solution Map")

        MAX_MARKERS = 1500

        if len(customers) > MAX_MARKERS:
            plot_idx = customers.sample(
                MAX_MARKERS,
                random_state=0,
            ).index

            st.caption(
                f"Showing a sample of {MAX_MARKERS:,} customers. "
                "KPIs are calculated using all customers."
            )
        else:
            plot_idx = customers.index

        served_idx = set(result["assignments"].keys())

        customer_map = customers.loc[
            plot_idx, ["lat", "lon"]
        ].copy()

        customer_map["status_color"] = [
            [42, 125, 225, 170]
            if idx in served_idx
            else [145, 145, 145, 150]
            for idx in plot_idx
        ]

        customer_map["radius"] = 35

        store_map = stores[["lat", "lon"]].copy()

        store_map["store_label"] = [
            f"Store {j} — OPEN"
            if j in result["open_stores"]
            else f"Store {j} — candidate"
            for j in stores.index
        ]

        store_map["store_color"] = [
            [20, 150, 70, 230]
            if j in result["open_stores"]
            else [120, 120, 120, 190]
            for j in stores.index
        ]

        store_map["radius"] = [
            150 if j in result["open_stores"] else 95
            for j in stores.index
        ]

        plot_idx_set = set(plot_idx)
        path_rows = []

        for i, j in result["assignments"].items():
            if i in plot_idx_set:
                path_rows.append(
                    {
                        "path": [
                            [
                                float(customers.iloc[i]["lon"]),
                                float(customers.iloc[i]["lat"]),
                            ],
                            [
                                float(stores.iloc[j]["lon"]),
                                float(stores.iloc[j]["lat"]),
                            ],
                        ]
                    }
                )

        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=customer_map,
                get_position="[lon, lat]",
                get_fill_color="status_color",
                get_radius="radius",
                radius_min_pixels=2,
                radius_max_pixels=5,
                pickable=True,
                id="customers",
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=store_map,
                get_position="[lon, lat]",
                get_fill_color="store_color",
                get_radius="radius",
                radius_min_pixels=5,
                radius_max_pixels=13,
                pickable=True,
                id="stores",
            ),
        ]

        if path_rows:
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=path_rows,
                    get_path="path",
                    get_color=[30, 150, 70, 75],
                    width_min_pixels=1,
                    get_width=1,
                    pickable=False,
                    id="assignments",
                )
            )

        view_state = pdk.ViewState(
            latitude=float(customers["lat"].mean()),
            longitude=float(customers["lon"].mean()),
            zoom=10,
            pitch=0,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                tooltip={"text": "{store_label}"},
                map_style=(
                    "https://basemaps.cartocdn.com/gl/"
                    "positron-gl-style/style.json"
                ),
            ),
            use_container_width=True,
        )

        st.caption(
            "Green markers = opened stores; gray markers = unopened "
            "candidates; blue dots = served customers; gray dots = "
            "unserved customers; green lines = customer-store assignments."
        )

else:
    st.info(
        "Set the sidebar parameters and click Optimize to run the model."
    )


# ============================================================
# REQUIRED ASSIGNMENT BENCHMARK
# 10,000 customers, 50 candidate stores
# K = 2, 5, 10, 20, 25
# ============================================================

st.divider()

st.subheader(
    "📈 Profit vs. Maximum Open Stores — "
    "10,000 Customers, 50 Candidate Stores"
)

st.caption(
    "This benchmark uses fixed parameters independent of the sidebar: "
    "D_max = 4 km, fixed cost Rs. 150,000/store/month, "
    "revenue Rs. 1,800/customer/month, and transport cost "
    "Rs. 60/km/customer/month."
)

BENCH_FC = 150000
BENCH_REV = 1800
BENCH_TC = 60

REQUIRED_K = [2, 5, 10, 20, 25]

with st.spinner(
    "Running the benchmark optimization for K = 2, 5, 10, 20, 25..."
):
    sens_df = run_sensitivity(
        n_customers=10000,
        n_stores=50,
        max_dist=4.0,
        fixed_cost=BENCH_FC,
        revenue=BENCH_REV,
        transp_cost=BENCH_TC,
        k_values=tuple(REQUIRED_K),
    )


# Prepare benchmark results.
chart_df = sens_df.copy()

chart_df["Max Open (K)"] = pd.to_numeric(
    chart_df["Max Open (K)"],
    errors="coerce",
)

chart_df["Profit (Rs.)"] = pd.to_numeric(
    chart_df["Profit (Rs.)"],
    errors="coerce",
)

chart_df = chart_df[
    chart_df["Max Open (K)"].isin(REQUIRED_K)
].dropna(
    subset=["Max Open (K)", "Profit (Rs.)"]
)

chart_df = (
    chart_df.drop_duplicates(subset=["Max Open (K)"])
    .set_index("Max Open (K)")
    .reindex(REQUIRED_K)
    .reset_index()
)


# ============================================================
# BENCHMARK PROFIT GRAPH
# ============================================================

st.subheader("Profit Comparison")

if chart_df["Profit (Rs.)"].notna().all():

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        chart_df["Max Open (K)"],
        chart_df["Profit (Rs.)"],
        marker="o",
        linewidth=2,
    )

    ax.set_xlabel("Maximum Dark Stores Allowed (K)")
    ax.set_ylabel("Monthly Profit (Rs.)")
    ax.set_xticks(REQUIRED_K)

    ax.ticklabel_format(
        axis="y",
        style="plain",
        useOffset=False,
    )

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    st.pyplot(fig, clear_figure=True)

    plt.close(fig)

else:
    st.error(
        "Valid profit results were not returned for all required "
        "K values: 2, 5, 10, 20, 25."
    )


# ============================================================
# BENCHMARK TABLE — RENDERED ONLY ONCE
# ============================================================

st.subheader("Benchmark Results Table")

st.dataframe(
    chart_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# INTERACTIVE PROFIT CURVE
# Uses current sidebar settings
# ============================================================

with st.expander(
    "Interactive Profit Curve Using Current Sidebar Inputs"
):

    st.caption(
        "This additional analysis responds to the current sidebar inputs. "
        "It is separate from the fixed assignment benchmark above."
    )

    interactive_k = tuple(
        k for k in REQUIRED_K if k <= int(n_stores)
    )

    if not interactive_k:

        st.warning(
            "Increase the number of candidate stores to at least 2."
        )

    else:

        with st.spinner(
            "Calculating interactive sensitivity results..."
        ):

            interactive_df = run_sensitivity(
                n_customers=int(n_customers),
                n_stores=int(n_stores),
                max_dist=float(max_dist),
                fixed_cost=float(fixed_cost),
                revenue=float(revenue),
                transp_cost=float(transp_cost),
                k_values=interactive_k,
            )

        interactive_df = interactive_df.copy()

        interactive_df["Max Open (K)"] = pd.to_numeric(
            interactive_df["Max Open (K)"],
            errors="coerce",
        )

        interactive_df["Profit (Rs.)"] = pd.to_numeric(
            interactive_df["Profit (Rs.)"],
            errors="coerce",
        )

        interactive_df = interactive_df.dropna(
            subset=["Max Open (K)", "Profit (Rs.)"]
        ).sort_values("Max Open (K)")

        if not interactive_df.empty:

            fig_interactive, ax_interactive = plt.subplots(
                figsize=(10, 4.5)
            )

            ax_interactive.plot(
                interactive_df["Max Open (K)"],
                interactive_df["Profit (Rs.)"],
                marker="o",
                linewidth=2,
            )

            ax_interactive.set_xlabel(
                "Maximum Dark Stores Allowed (K)"
            )

            ax_interactive.set_ylabel("Monthly Profit (Rs.)")

            ax_interactive.set_xticks(
                interactive_df["Max Open (K)"].astype(int)
            )

            ax_interactive.ticklabel_format(
                axis="y",
                style="plain",
                useOffset=False,
            )

            ax_interactive.grid(True, alpha=0.3)

            fig_interactive.tight_layout()

            st.pyplot(
                fig_interactive,
                clear_figure=True,
            )

            plt.close(fig_interactive)

            st.subheader("Interactive Results Table")

            st.dataframe(
                interactive_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.warning(
                "No valid interactive sensitivity results were returned."
            )


# ============================================================
# INTERPRETATION
# ============================================================

with st.expander("How to Read the Benchmark Profit Curve"):

    st.markdown(
        """
        - The horizontal axis represents the maximum number of dark
          stores allowed to open, K.

        - The vertical axis represents monthly profit in rupees.

        - The benchmark uses 10,000 customers, 50 candidate stores,
          a 4 km maximum service distance, and the fixed financial
          assumptions stated above.

        - The optimization is profit-maximizing, not simply
          coverage-maximizing. Customers may remain unserved if no
          open store is within the maximum distance or if serving
          them is not selected by the optimal solution.

        - The maximum-distance constraint restricts customer-store
          assignments. A customer cannot be assigned to a store
          outside D_max.

        - Increasing K permits additional stores to open, but does
          not require every permitted store to open. The optimizer
          chooses the configuration that maximizes profit.

        - The graph is generated from the optimization results;
          it is not a manually specified or hard-coded profit curve.
        """
    )
