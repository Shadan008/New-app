"""
Core optimization logic for the Dark Store Location Optimizer.
Kept separate from app.py (UI) so it can be unit-tested without Streamlit/folium.
"""

import numpy as np
import pandas as pd
from pulp import (
    LpProblem, LpMaximize, LpVariable, LpStatus, lpSum, value, PULP_CBC_CMD,
)

# Chennai bounding box
LAT_MIN, LAT_MAX = 12.80, 13.30
LON_MIN, LON_MAX = 80.00, 80.30


def generate_data(n_customers: int, n_stores: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    customers = pd.DataFrame({
        "id": range(n_customers),
        "lat": rng.uniform(LAT_MIN, LAT_MAX, n_customers),
        "lon": rng.uniform(LON_MIN, LON_MAX, n_customers),
    })
    stores = pd.DataFrame({
        "id": range(n_stores),
        "lat": rng.uniform(LAT_MIN, LAT_MAX, n_stores),
        "lon": rng.uniform(LON_MIN, LON_MAX, n_stores),
    })
    return customers, stores


def compute_distance_matrix(customers: pd.DataFrame, stores: pd.DataFrame) -> np.ndarray:
    """Vectorized haversine distance matrix (km), shape (n_customers, n_stores)."""
    R = 6371.0
    lat1 = np.radians(customers["lat"].to_numpy())[:, None]
    lon1 = np.radians(customers["lon"].to_numpy())[:, None]
    lat2 = np.radians(stores["lat"].to_numpy())[None, :]
    lon2 = np.radians(stores["lon"].to_numpy())[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def solve_model(D: np.ndarray, max_dist: float, max_open: int,
                 fixed_cost: float, revenue: float, transp_cost: float,
                 msg: bool = False):
    """
    Profit-maximizing facility location with an optional-service assignment.

    Fix vs. a naive "serve everyone" formulation: customers are served AT
    MOST once (sum_j x_ij <= 1), not EXACTLY once. Forcing full coverage
    makes the model infeasible whenever a customer has no candidate store
    within max_dist, or whenever K is too small to reach every customer --
    exactly the situation the K in {2, 5} / 10,000-customer sensitivity run
    produces. Here, a customer who can't be reached/profitably served is
    simply left unserved, which is correct for a profit-maximizing retailer.

    Performance note: for FIXED y, the per-customer assignment problem
    decomposes (each customer independently picks its best open store,
    since the only coupling constraint is sum_j x_ij <= 1 per customer).
    So the LP relaxation of x is already integral at the optimum, and x is
    declared continuous in [0,1] rather than binary. Only y (one binary
    variable per candidate store) is integer. This cuts the integer-variable
    count from n_customers*n_stores to n_stores, which is what makes the
    10,000-customer x 50-store sweep solve in seconds instead of being an
    intractable MILP.
    """
    n_c, n_s = D.shape
    prob = LpProblem("DarkStoreLocation", LpMaximize)

    y = LpVariable.dicts("open", range(n_s), cat="Binary")

    pairs = [(i, j) for i in range(n_c) for j in range(n_s) if D[i, j] <= max_dist]
    x = {(i, j): LpVariable(f"assign_{i}_{j}", lowBound=0, upBound=1, cat="Continuous")
         for (i, j) in pairs}

    by_customer = {}
    for (i, j) in pairs:
        by_customer.setdefault(i, []).append(j)

    prob += (
        lpSum((revenue - transp_cost * D[i, j]) * x[i, j] for (i, j) in pairs)
        - lpSum(fixed_cost * y[j] for j in range(n_s))
    )

    for i, js in by_customer.items():
        prob += lpSum(x[i, j] for j in js) <= 1

    for (i, j) in pairs:
        prob += x[i, j] <= y[j]

    prob += lpSum(y[j] for j in range(n_s)) <= max_open

    solver = PULP_CBC_CMD(msg=msg)
    prob.solve(solver)
    status = LpStatus[prob.status]

    open_stores = [j for j in range(n_s) if value(y[j]) is not None and value(y[j]) > 0.5]
    assignments = {}
    for i, js in by_customer.items():
        best_j, best_val = None, 0.5
        for j in js:
            v = value(x[i, j])
            if v is not None and v > best_val:
                best_val, best_j = v, j
        if best_j is not None:
            assignments[i] = best_j

    total_revenue = revenue * len(assignments)
    total_transport = sum(transp_cost * D[i, j] for i, j in assignments.items())
    total_fixed = fixed_cost * len(open_stores)
    profit = total_revenue - total_transport - total_fixed

    return {
        "status": status,
        "open_stores": open_stores,
        "assignments": assignments,
        "profit": profit,
        "revenue": total_revenue,
        "transport_cost": total_transport,
        "fixed_cost": total_fixed,
        "num_open": len(open_stores),
        "num_served": len(assignments),
        "num_customers": n_c,
    }


def run_sensitivity(n_customers: int, n_stores: int, max_dist: float,
                     fixed_cost: float, revenue: float, transp_cost: float,
                     k_values: tuple, seed: int = 123):
    cust, store = generate_data(n_customers, n_stores, seed=seed)
    D = compute_distance_matrix(cust, store)
    rows = []
    for k in k_values:
        res = solve_model(D, max_dist, k, fixed_cost, revenue, transp_cost)
        rows.append({
            "Max Open (K)": k,
            "Profit (Rs.)": res["profit"],
            "Stores Opened": res["num_open"],
            "Customers Served": res["num_served"],
            "Status": res["status"],
        })
    return pd.DataFrame(rows)
