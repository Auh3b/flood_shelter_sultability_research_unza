# lscp_mclp_pulp.py
from collections import defaultdict

import pandas as pd
import pulp

# ---------- User params ----------
OD_CSV = 'od_matrix.csv'       # input OD matrix exported from QGIS
S = 60                         # coverage threshold in minutes
# epsilon-constraint targets (fractions)
COVERAGE_TARGETS = [0.80, 0.90, 0.95, 1.00]
# max p to sweep for MCLP (choose <= number of candidate sites)
PMAX = 8
# ---------------------------------

# Read OD matrix
df = pd.read_csv(OD_CSV)

# Basic checks
required_cols = {'demand_id', 'site_id', 'travel_time', 'pop'}
if not required_cols.issubset(df.columns):
    raise SystemExit(f"Input CSV must contain columns: {required_cols}")

# Build sets and params
demands = sorted(df['demand_id'].unique().tolist())
sites = sorted(df['site_id'].unique().tolist())

# Map demand -> population (assume consistent)
pop = df[['demand_id', 'pop']].drop_duplicates().set_index('demand_id')[
    'pop'].to_dict()
total_pop = sum(pop.values())

# Build d_ij dict and coverage boolean
d_ij = {(row.demand_id, row.site_id)
         : row.travel_time for row in df.itertuples()}
coverable = {(i, j): (d_ij[(i, j)] <= S) for i in demands for j in sites}

# Helper: pretty join


def sites_to_str(sol_sites):
    return ';'.join(sorted(sol_sites))


# ---------- LSCP (full coverage) ----------
prob = pulp.LpProblem("LSCP_full_coverage", pulp.LpMinimize)
y = pulp.LpVariable.dicts('y', sites, lowBound=0, upBound=1, cat='Binary')
# Objective: minimize number of sites

prob += pulp.lpSum(y[j] for j in sites)
# Constraints: each demand must be covered by at least one chosen site that is within S
for i in demands:
    # sites that can cover i
    Sj = [j for j in sites if coverable[(i, j)]]
    if not Sj:
        # This demand cannot be covered by any site within S -> LSCP infeasible
        prob = None
        break
    prob += pulp.lpSum(y[j] for j in Sj) >= 1, f"cover_{i}"

if prob is None:
    print(
        f"LSCP infeasible: some demand points cannot be covered within S={S} minutes by any candidate site.")
    lscp_result = None
else:
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = [j for j in sites if pulp.value(y[j]) > 0.5]
    lscp_result = {'num_sites': len(selected), 'selected_sites': selected}
    print("LSCP solved. Num sites:", lscp_result['num_sites'])

# Save LSCP results
if lscp_result:
    pd.DataFrame([{'selected_sites': sites_to_str(lscp_result['selected_sites']),
                 'num_sites': lscp_result['num_sites']}]).to_csv('lscp_solution.csv', index=False)

# ---------- Epsilon-constraint: minimize sites subject to coverage >= target ----------
epsilon_rows = []
for target in COVERAGE_TARGETS:
    prob = pulp.LpProblem(f"LSCP_epsilon_{int(target*100)}", pulp.LpMinimize)
    y = pulp.LpVariable.dicts('y', sites, lowBound=0, upBound=1, cat='Binary')
    # x_i = 1 if demand i covered (by chosen site)
    x = pulp.LpVariable.dicts(
        'x', demands, lowBound=0, upBound=1, cat='Binary')
    # Objective
    prob += pulp.lpSum(y[j] for j in sites)
    # Link x and y
    for i in demands:
        Sj = [j for j in sites if coverable[(i, j)]]
        if Sj:
            prob += x[i] <= pulp.lpSum(y[j] for j in Sj)
        else:
            # Can't be covered at all, force x[i]=0
            prob += x[i] == 0
    # Coverage constraint
    prob += pulp.lpSum(pop[i] * x[i] for i in demands) >= target * total_pop
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = [j for j in sites if pulp.value(y[j]) > 0.5]
    covered_demands = [i for i in demands if pulp.value(x[i]) > 0.5]
    covered_pop = sum(pop[i] for i in covered_demands)
    epsilon_rows.append({
        'coverage_target': target,
        'num_sites': len(selected),
        'selected_sites': sites_to_str(selected),
        'covered_pop': covered_pop,
        'coverage_pct': covered_pop/total_pop
    })
    print(f"Target {target:.2f}: num_sites={len(selected)}, coverage={covered_pop}/{total_pop} ({100*covered_pop/total_pop:.1f}%)")

pd.DataFrame(epsilon_rows).to_csv('epsilon_solutions.csv', index=False)

# ---------- MCLP sweep (for p = 1..PMAX) ----------
mclp_rows = []
Pmax = min(PMAX, len(sites))
for p_val in range(1, Pmax+1):
    prob = pulp.LpProblem(f"MCLP_p_{p_val}", pulp.LpMaximize)
    y = pulp.LpVariable.dicts('y', sites, lowBound=0, upBound=1, cat='Binary')
    # 1 if demand covered by chosen sites
    x = pulp.LpVariable.dicts(
        'x', demands, lowBound=0, upBound=1, cat='Binary')
    # Objective: maximize covered population
    prob += pulp.lpSum(pop[i] * x[i] for i in demands)
    # Link x and y
    for i in demands:
        Sj = [j for j in sites if coverable[(i, j)]]
        if Sj:
            prob += x[i] <= pulp.lpSum(y[j] for j in Sj)
        else:
            prob += x[i] == 0
    # Exactly p sites chosen
    prob += pulp.lpSum(y[j] for j in sites) == p_val
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = [j for j in sites if pulp.value(y[j]) > 0.5]
    covered_demands = [i for i in demands if pulp.value(x[i]) > 0.5]
    covered_pop = sum(pop[i] for i in covered_demands)
    mclp_rows.append({
        'p': p_val,
        'num_selected': len(selected),
        'selected_sites': sites_to_str(selected),
        'covered_pop': covered_pop,
        'coverage_pct': covered_pop/total_pop
    })
    print(f"p={p_val}: coverage {covered_pop}/{total_pop} ({100*covered_pop/total_pop:.1f}%) with sites: {selected}")

pd.DataFrame(mclp_rows).to_csv('mclp_sweep.csv', index=False)

print("Done. Outputs: lscp_solution.csv (if any), epsilon_solutions.csv, mclp_sweep.csv")
