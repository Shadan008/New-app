"""Standalone smoke tests for the optimization core (no Streamlit server needed)."""
import time
import numpy as np
from core import generate_data, compute_distance_matrix, solve_model

print("=== Test 1a: tiny instance, ORIGINAL single-order economics (Rs.120 rev, Rs.8/km, Rs.5,00,000 fixed) ===")
cust, store = generate_data(20, 3, seed=1)
D = compute_distance_matrix(cust, store)
res = solve_model(D, max_dist=5.0, max_open=2, fixed_cost=500000, revenue=120, transp_cost=8)
print(res["status"], "| served:", res["num_served"], "/20 | opened:", res["num_open"], "| profit:", round(res["profit"]))
assert res["status"] == "Optimal"
assert res["num_open"] == 0, "at these assumptions no store should ever be worth opening"
print("  -> correctly opens 0 stores: Rs.120/order can never recover a Rs.5,00,000 fixed cost "
      "(this is WHY the calibrated monthly-economics defaults are used elsewhere in the app)")

print("\n=== Test 1b: same instance, CALIBRATED monthly economics (Rs.1800 rev, Rs.60/km, Rs.1,50,000 fixed) ===")
res1b = solve_model(D, max_dist=5.0, max_open=2, fixed_cost=150000, revenue=1800, transp_cost=60)
print(res1b["status"], "| served:", res1b["num_served"], "/20 | opened:", res1b["num_open"], "| profit:", round(res1b["profit"]))
assert res1b["status"] == "Optimal"

print("\n=== Test 2: the ORIGINAL bug reproduced -- an unreachable customer ===")
D2 = np.array([[100.0, 100.0], [1.0, 2.0], [1.5, 1.0]])  # customer 0 unreachable, D_max=5
res2 = solve_model(D2, max_dist=5.0, max_open=2, fixed_cost=100, revenue=120, transp_cost=8)
print(res2["status"], "| served:", res2["num_served"], "/3 | assignments:", res2["assignments"])
assert res2["status"] == "Optimal"
assert 0 not in res2["assignments"], "customer 0 should be unreachable/unserved"
assert res2["num_served"] == 2

print("\n=== Test 3: mid-size instance, calibrated economics ===")
cust3, store3 = generate_data(60, 6, seed=7)
D3 = compute_distance_matrix(cust3, store3)
res3 = solve_model(D3, max_dist=4.0, max_open=3, fixed_cost=150000, revenue=1800, transp_cost=60)
print(res3["status"], "served:", res3["num_served"], "opened:", res3["num_open"], "profit:", round(res3["profit"]))
assert res3["status"] == "Optimal"

print("\n=== Test 4: required sensitivity sweep, 10,000 customers x 50 stores, K in {2,5,10,20,25} ===")
t0 = time.time()
cust4, store4 = generate_data(10000, 50, seed=123)
D4 = compute_distance_matrix(cust4, store4)
print(f"distance matrix computed in {time.time()-t0:.2f}s, shape={D4.shape}")

for k in [2, 5, 10, 20, 25]:
    t1 = time.time()
    r = solve_model(D4, max_dist=4.0, max_open=k, fixed_cost=150000, revenue=1800, transp_cost=60)
    dt = time.time() - t1
    print(f"K={k:>2} | status={r['status']:>10} | opened={r['num_open']:>2} | "
          f"served={r['num_served']:>5}/10000 | profit=Rs.{r['profit']:>14,.0f} | solve_time={dt:.1f}s")
    assert r["status"] == "Optimal", f"K={k} did not solve to optimality: {r['status']}"

print(f"\nTotal sweep time: {time.time()-t0:.1f}s")
print("\nALL TESTS PASSED")
