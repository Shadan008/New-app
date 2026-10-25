# Dark Store Location Optimizer — Chennai

AS 5570, Assignment 1: profit-maximizing dark-store facility location with a
maximum service-distance constraint.

## 1. Verification of the uploaded solution

Two real bugs were found in the draft that was uploaded, both of which affect
whether the system actually produces a feasible, usable solution.

### Bug 1 (critical): forced full coverage makes the model infeasible

The uploaded model constrained every customer to be assigned to exactly one
store:

```python
prob += lpSum([x[i][j] for j in range(n_s) if D[i][j] <= max_dist]) == 1
```

This is infeasible in two situations that the assignment itself creates:

- **Any customer with zero candidate stores within `D_max`.** The sum is
  empty, so the constraint becomes `0 == 1` — infeasible by construction.
- **Too few open stores to reach everyone.** The required sensitivity run
  (10,000 customers, 50 candidates, `D_max = 4 km`, `K` as low as 2) cannot
  possibly put every customer within 4 km of one of only 2–5 open stores
  scattered over the whole Chennai bounding box.

The uploaded `app.py` also never checks `LpStatus[prob.status]`. When the
solver can't find a feasible solution, `value(x[i][j])` and `value(y[j])`
return `None`; the result-extraction code silently treats that as "0 stores
opened, 0 customers served, profit = 0" instead of surfacing that the solve
failed. **The sensitivity chart the assignment explicitly asks for would
have been built on silently-wrong infeasible results.**

**Fix:** make service optional — `sum_j x_ij <= 1` instead of `== 1` — so a
customer who can't be reached (or isn't worth reaching) is simply left
unserved. For a profit-maximizing retailer this is also the economically
correct behavior, not just a technical workaround. The fixed model also
checks and reports `status` explicitly.

### Bug 2: the stated cost/revenue assumptions are never profitable

Running the corrected model with the uploaded defaults (₹120 revenue per
customer, ₹8/km transport, ₹5,00,000 fixed cost per store) against the
required 10,000-customer / 50-store instance opens **zero** stores at every
value of `K`. This isn't a bug in the solver — it's correct given those
numbers:

- Over the Chennai bounding box (~1,815 km²), a 4 km service radius covers
  about 2.8% of the area, so a randomly placed store reaches roughly
  250–300 customers.
- At ₹120 revenue and up to ₹32 transport cost per customer (8 × 4 km), the
  maximum margin per customer is ~₹88.
- Breaking even on a ₹5,00,000 fixed cost needs ~5,700 served customers per
  store — more than 10x the store's addressable base.

So the original numbers describe a single delivery's margin against a
facility's full fixed cost, which can never pencil out at this density. The
earlier draft never surfaced this because its infeasibility bug (Bug 1) was
silently returning `profit = 0` regardless of the real cause.

**Fix (an explicit, documented assumption):** treat `revenue` / `transport
cost` as **monthly, aggregated-over-repeat-orders** figures for a recurring
customer, not a single delivery, which is the standard way quick-commerce
unit economics are modeled. Calibrated defaults used in this app:

| Parameter | Value | Rationale |
|---|---|---|
| Fixed cost per store / month | ₹1,50,000 | Rent, staffing, utilities |
| Revenue per served customer / month | ₹1,800 | ~₹9,000 monthly basket × 20% margin |
| Transport cost per km / customer / month | ₹60 | Aggregated across ~8 deliveries/month |

These are adjustable in the sidebar; the original single-order figures are
still available by typing them in, and the app will correctly report "0
stores opened" if you do, which is itself a useful sanity check to show in
a submission.

### Other issues fixed along the way

- **Scalability.** The uploaded model declares `x_ij` as `Binary` for every
  customer–store pair — 500,000 binary variables for the 10,000×50 case,
  on top of CBC being invoked fresh for every point on the sensitivity
  curve. In this formulation, for any **fixed** `y`, the per-customer
  assignment problem decomposes (the only coupling constraint is "serve at
  most once"), so the LP relaxation of `x` is already integral at the
  optimum. `x` is declared continuous here; only `y` (one binary per
  candidate store — at most 50) stays integer. The full 10,000×50 sweep
  over all five required `K` values now solves in **~6 seconds** total
  (tested; see `test_solver.py`), versus a MILP with 500,050 integer
  variables in the original.
- **Distance matrix.** Replaced the nested Python double loop with a
  vectorized NumPy haversine computation.
- **Map rendering.** Plotting 10,000 individual Folium markers freezes the
  browser; the map now samples up to 1,500 customers for display while
  KPIs are computed over the full population.
- **Missing import** introduced during my own refactor (`pd` used but not
  imported) — caught by the automated test harness below before shipping.

## 2. Model

**Sets:** customers `I`, candidate stores `J`. **Decision variables:**
`y_j ∈ {0,1}` (open store `j`), `x_ij ∈ [0,1]` (serve customer `i` from
store `j`; continuous but integral at the optimum — see above).

```
maximize   Σ_i Σ_j (R - C·d_ij) x_ij  -  Σ_j F·y_j

s.t.       Σ_j x_ij ≤ 1                         for all i   (serve at most once)
           x_ij ≤ y_j                           for all i,j (only open stores)
           x_ij = 0                             whenever d_ij > D_max
           Σ_j y_j ≤ K
           y_j ∈ {0,1},  0 ≤ x_ij ≤ 1
```

Distance `d_ij` is haversine (great-circle) over randomly generated
customer/candidate points inside the Chennai bounding box
(lat 12.80–13.30, lon 80.00–80.30), as a proxy for road-network distance.

## 3. Files

- `app.py` — Streamlit UI (sidebar inputs, KPI table, Folium map, required
  sensitivity chart).
- `core.py` — optimization logic only, framework-free, so it can be unit
  tested without a browser or a running Streamlit server.
- `test_solver.py` — smoke tests: a tiny feasible case, the exact
  infeasibility scenario from Bug 1 (confirms it's now handled), a
  mid-size case, and the full required 10,000×50 / `K∈{2,5,10,20,25}`
  sweep with timing and an `Optimal`-status assertion on every run.
- `requirements.txt` — pinned, verified in a clean virtualenv
  (`pulp==2.8.0` specifically — PuLP 4.x restructures its public API and
  no longer exposes `PULP_CBC_CMD`/`LpStatus` the same way).

## 4. Running locally

```bash
pip install -r requirements.txt
python test_solver.py      # optional: verify the model before launching the UI
streamlit run app.py
```

## 5. Deployment compatibility fix (Python / Altair)

The repository includes both `.python-version` (`3.12.10`) and `runtime.txt`
(`python-3.12`) so the deployment does not silently select Python 3.14. The
`requirements.txt` file also pins a compatible Streamlit / Altair / NumPy /
pandas stack. This addresses the reported `TypedDict(closed=True)` import
failure that occurred with Altair 5.5.0 on the Python 3.14.9 runtime.

If Streamlit Cloud keeps using the old environment, open **Manage app →
Settings → Reboot app** after pushing these files. If the environment remains
stuck on Python 3.14, delete and redeploy the app after confirming the runtime
files are present at the repository root.

## 6. Deploying to Streamlit Community Cloud

Deployment needs your own GitHub account and Streamlit Cloud sign-in, which
I don't have access to, so this part has to be done by you — it's a five
minute, five-click process:

1. Create a new **public** GitHub repository and push these four files to
   it (`app.py`, `core.py`, `requirements.txt`, and the `.streamlit/`
   folder).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, pick the repository/branch, set the main file path to
   `app.py`.
4. Click **Deploy**. First build takes a couple of minutes (installing
   PuLP's CBC solver binary etc.).
5. Copy the resulting `https://<app-name>.streamlit.app` URL — that's your
   submission link.

## 7. Interpreting the required sensitivity chart

At the calibrated monthly economics above, profit rises steadily across
`K = 2 → 25`: at this candidate density (50 locations over the whole
Chennai bounding box with `D_max = 4 km`), **coverage**, not per-store
profitability, is still the binding constraint through `K = 25` — every
additional store reaches new customers who were previously out of range,
and each one served is individually profitable. Running the same model
further (not required by the assignment, but checked here for
completeness) shows it actually plateaus around `K ≈ 30–34`: beyond that,
the remaining candidate locations only overlap existing coverage or sit in
sparse areas, so opening them adds no profitable customers. With `D_max =
4 km` and only 50 random candidates, about 22% of all customers have no
candidate store within range at any `K`, so even full build-out never
reaches 100%.
