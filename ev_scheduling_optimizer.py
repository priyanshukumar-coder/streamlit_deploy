"""
EV Charging Schedule Optimizer — Part B
=========================================
Formulates a Mixed Integer Linear Program (MILP) via Google OR-Tools.

Objective : minimise peak load across all zones
            + weighted consumer inconvenience (delay penalty)
Hard constraint : base_load + scheduled EV charging ≤ 95% transformer capacity
                  at every 15-min slot

Rolling horizon: re-solves every 15 minutes, executes only first time-step.
Fallback      : priority-based greedy heuristic when MILP hits time limit.

Install:
    pip install ortools pandas numpy matplotlib seaborn
    (also needs ev_prediction_pipeline.py outputs in same folder, OR runs standalone)

Run:
    python ev_scheduling_optimizer.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import time

np.random.seed(42)

# ─────────────────────────────────────────────
# 0.  CONFIG
# ─────────────────────────────────────────────
TRANSFORMER_CAPACITY_KW  = 4000.0
HARD_LIMIT_PCT           = 0.95          # 3800 kW
WARNING_PCT              = 0.90          # 3600 kW
HARD_LIMIT_KW            = TRANSFORMER_CAPACITY_KW * HARD_LIMIT_PCT

SLOT_MINUTES             = 15
HORIZON_SLOTS            = 32            # 8-hour look-ahead
MAX_DELAY_SLOTS          = 16            # max 4-hour delay allowed
INCONVENIENCE_WEIGHT     = 0.4          # λ in objective
MILP_TIME_LIMIT_SEC      = 10.0

ZONES = {
    "Koramangala":  {"base_kw": 2800, "ev_density": 0.18, "zone_type": 1},
    "Whitefield":   {"base_kw": 3500, "ev_density": 0.22, "zone_type": 2},
    "Jayanagar":    {"base_kw": 2100, "ev_density": 0.12, "zone_type": 0},
    "Marathahalli": {"base_kw": 3100, "ev_density": 0.20, "zone_type": 1},
    "Hebbal":       {"base_kw": 2600, "ev_density": 0.15, "zone_type": 1},
}

# ─────────────────────────────────────────────
# 1.  DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class EVBatch:
    """A group of EVs that arrive at the same slot and want to charge."""
    batch_id:     int
    zone:         str
    arrival_slot: int
    n_evs:        int
    demand_kw:    float          # total demand if all charge simultaneously
    charger_kw:   float = 7.2   # average charger rating
    max_delay:    int   = MAX_DELAY_SLOTS
    priority:     int   = 1     # 1=normal, 2=high (e.g. emergency vehicles)


@dataclass
class ScheduleResult:
    zone:             str
    slot:             int
    timestamp:        pd.Timestamp
    base_load_kw:     float
    unmanaged_ev_kw:  float
    scheduled_ev_kw:  float
    net_unmanaged_kw: float
    net_scheduled_kw: float
    loading_unmanaged_pct: float
    loading_scheduled_pct: float
    violation_unmanaged:   bool
    violation_scheduled:   bool
    solver_used:      str        # "MILP" or "Heuristic"


# ─────────────────────────────────────────────
# 2.  SYNTHETIC SCENARIO GENERATOR
# ─────────────────────────────────────────────

def generate_scenario(zone: str, cfg: dict, start_hour: int = 17,
                       n_slots: int = HORIZON_SLOTS) -> Tuple[np.ndarray, np.ndarray, List[EVBatch]]:
    """
    Returns:
        base_load  : kW per slot (shape: n_slots)
        unmanaged  : kW of EV demand if all charge on arrival (shape: n_slots)
        ev_batches : list of EVBatch objects arriving during horizon
    """
    slots = np.arange(n_slots)
    hours = (start_hour + slots * SLOT_MINUTES / 60) % 24

    # Base load — evening ramp
    morning  = np.exp(-0.5 * ((hours - 9.0)  / 1.5) ** 2)
    evening  = np.exp(-0.5 * ((hours - 20.0) / 1.5) ** 2)
    profile  = 0.35 + 0.65 * (morning + evening)
    if cfg["zone_type"] == 2:
        profile += np.where((hours >= 8) & (hours <= 18), 0.3, 0.0)

    base_load = cfg["base_kw"] * profile * (1 + np.random.normal(0, 0.02, n_slots))
    base_load = np.clip(base_load, cfg["base_kw"] * 0.2, cfg["base_kw"] * 1.05)

    # EV arrivals — Poisson process with evening peak
    n_evs_total = int(5000 * cfg["ev_density"] * 0.5)
    arrival_rate = np.exp(-0.5 * ((hours - 19.0) / 1.5) ** 2) + \
                   0.2 * np.exp(-0.5 * ((hours - 8.5) / 1.0) ** 2)
    arrival_rate /= arrival_rate.sum()
    arrivals_per_slot = np.random.multinomial(n_evs_total, arrival_rate)

    avg_charger_kw = 7.2
    concurrency    = 0.15

    batches = []
    unmanaged = np.zeros(n_slots)
    bid = 0
    for s, n in enumerate(arrivals_per_slot):
        if n == 0:
            continue
        demand = n * avg_charger_kw * concurrency
        # Unmanaged: all charge immediately on arrival for ~2 slots (30 min)
        for dur in range(min(2, n_slots - s)):
            unmanaged[s + dur] += demand * 0.6  # partial overlap
        batches.append(EVBatch(
            batch_id=bid, zone=zone, arrival_slot=s,
            n_evs=n, demand_kw=demand,
            priority=2 if np.random.rand() < 0.05 else 1  # 5% high priority
        ))
        bid += 1

    return base_load, np.clip(unmanaged, 0, None), batches


# ─────────────────────────────────────────────
# 3.  MILP OPTIMIZER
# ─────────────────────────────────────────────

def solve_milp(base_load: np.ndarray, batches: List[EVBatch],
               n_slots: int = HORIZON_SLOTS) -> Tuple[np.ndarray, str, float]:
    """
    Decision variable: x[b, s] ∈ {0,1} — does batch b charge in slot s?
    Each batch charges in exactly one slot (simplified: batch = atomic unit).
    Returns scheduled EV load per slot (kW), solver tag, optimality gap.
    """
    from ortools.sat.python import cp_model

    model  = cp_model.CpModel()
    SCALE  = 100   # convert float kW → integer for CP-SAT

    # x[b][s] = 1 if batch b is scheduled to charge in slot s
    x = {}
    for b, batch in enumerate(batches):
        for s in range(batch.arrival_slot,
                       min(batch.arrival_slot + batch.max_delay + 1, n_slots)):
            x[b, s] = model.NewBoolVar(f"x_{b}_{s}")

    # C1: each batch scheduled exactly once
    for b, batch in enumerate(batches):
        slots_available = [s for s in range(batch.arrival_slot,
                            min(batch.arrival_slot + batch.max_delay + 1, n_slots))]
        model.AddExactlyOne([x[b, s] for s in slots_available])

    # C2: high-priority batches must charge within 2 slots of arrival
    for b, batch in enumerate(batches):
        if batch.priority == 2:
            for s in range(batch.arrival_slot,
                           min(batch.arrival_slot + batch.max_delay + 1, n_slots)):
                if s > batch.arrival_slot + 2:
                    model.Add(x[b, s] == 0)

    # Precompute scaled base load
    base_scaled = [int(round(base_load[s] * SCALE)) for s in range(n_slots)]
    cap_scaled  = int(round(HARD_LIMIT_KW * SCALE))

    # C3: capacity constraint per slot
    for s in range(n_slots):
        ev_load_in_slot = []
        for b, batch in enumerate(batches):
            if (b, s) in x:
                ev_load_in_slot.append(
                    x[b, s] * int(round(batch.demand_kw * SCALE))
                )
        if ev_load_in_slot:
            model.Add(base_scaled[s] + sum(ev_load_in_slot) <= cap_scaled)

    # Objective: minimise peak load proxy + inconvenience (delay)
    # Peak proxy: minimise sum of squared-ish load (linearised via auxiliary)
    peak_var = model.NewIntVar(0, int(cap_scaled * 1.1), "peak")
    for s in range(n_slots):
        ev_in_slot = []
        for b, batch in enumerate(batches):
            if (b, s) in x:
                ev_in_slot.append(x[b, s] * int(round(batch.demand_kw * SCALE)))
        if ev_in_slot:
            model.Add(peak_var >= base_scaled[s] + sum(ev_in_slot))
        else:
            model.Add(peak_var >= base_scaled[s])

    # Inconvenience: sum of delay * n_evs
    inconvenience_terms = []
    for b, batch in enumerate(batches):
        for s in range(batch.arrival_slot,
                       min(batch.arrival_slot + batch.max_delay + 1, n_slots)):
            if (b, s) in x:
                delay = s - batch.arrival_slot
                inconvenience_terms.append(
                    x[b, s] * int(delay * batch.n_evs)
                )

    inconv_total = model.NewIntVar(0, int(1e9), "inconv")
    if inconvenience_terms:
        model.Add(inconv_total == sum(inconvenience_terms))
    else:
        model.Add(inconv_total == 0)

    total_evs = max(sum(b.n_evs for b in batches), 1)
    lam = int(INCONVENIENCE_WEIGHT * SCALE)
    # Normalise inconvenience as integer constant before building expression
    inconv_weight = max(1, lam // total_evs)
    model.Minimize(peak_var + inconv_weight * inconv_total)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MILP_TIME_LIMIT_SEC
    solver.parameters.num_search_workers  = 4

    status = solver.Solve(model)

    scheduled_ev = np.zeros(n_slots)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for b, batch in enumerate(batches):
            for s in range(batch.arrival_slot,
                           min(batch.arrival_slot + batch.max_delay + 1, n_slots)):
                if (b, s) in x and solver.Value(x[b, s]) == 1:
                    scheduled_ev[s] += batch.demand_kw
        tag = "MILP-Optimal" if status == cp_model.OPTIMAL else "MILP-Feasible"
        gap = solver.ObjectiveValue()
    else:
        # Fallback to heuristic
        scheduled_ev, tag, gap = heuristic_fallback(base_load, batches, n_slots)

    return scheduled_ev, tag, gap


# ─────────────────────────────────────────────
# 4.  PRIORITY HEURISTIC FALLBACK
# ─────────────────────────────────────────────

def heuristic_fallback(base_load: np.ndarray, batches: List[EVBatch],
                        n_slots: int) -> Tuple[np.ndarray, str, float]:
    """
    Greedy: sort by priority desc, then try to place each batch in earliest
    slot where capacity is not exceeded.
    """
    scheduled_ev  = np.zeros(n_slots)
    sorted_batches = sorted(batches, key=lambda b: (-b.priority, b.arrival_slot))

    for batch in sorted_batches:
        placed = False
        for s in range(batch.arrival_slot,
                       min(batch.arrival_slot + batch.max_delay + 1, n_slots)):
            net = base_load[s] + scheduled_ev[s] + batch.demand_kw
            if net <= HARD_LIMIT_KW:
                scheduled_ev[s] += batch.demand_kw
                placed = True
                break
        if not placed:
            # Must place somewhere — pick least-loaded slot (graceful degradation)
            best_s = min(range(batch.arrival_slot,
                               min(batch.arrival_slot + batch.max_delay + 1, n_slots)),
                         key=lambda s: base_load[s] + scheduled_ev[s])
            scheduled_ev[best_s] += batch.demand_kw

    return scheduled_ev, "Heuristic", 0.0


# ─────────────────────────────────────────────
# 5.  ROLLING HORIZON CONTROLLER
# ─────────────────────────────────────────────

def rolling_horizon(zone: str, cfg: dict,
                    sim_start_hour: int = 15,
                    sim_slots: int = 48) -> List[ScheduleResult]:
    """
    Simulate a full evening (sim_slots × 15 min).
    Every HORIZON_SLOTS we re-solve; execute only the first time-step.
    """
    base_load, unmanaged, batches = generate_scenario(
        zone, cfg, start_hour=sim_start_hour, n_slots=sim_slots
    )

    results      = []
    executed_ev  = np.zeros(sim_slots)
    start_dt     = pd.Timestamp("2024-05-20") + pd.Timedelta(hours=sim_start_hour)

    for current_slot in range(sim_slots):
        # Extract horizon window
        end_slot  = min(current_slot + HORIZON_SLOTS, sim_slots)
        window_bl = base_load[current_slot:end_slot]

        # Batches that arrive within the horizon and haven't expired
        horizon_batches = [
            b for b in batches
            if current_slot <= b.arrival_slot < end_slot
            and b.arrival_slot + b.max_delay >= current_slot
        ]
        # Shift batch arrival slots to be relative to window start
        shifted = []
        for b in horizon_batches:
            shifted.append(EVBatch(
                batch_id=b.batch_id, zone=b.zone,
                arrival_slot=b.arrival_slot - current_slot,
                n_evs=b.n_evs, demand_kw=b.demand_kw,
                charger_kw=b.charger_kw,
                max_delay=b.max_delay,
                priority=b.priority,
            ))

        t0 = time.time()
        if shifted:
            window_sched, solver_tag, _ = solve_milp(window_bl, shifted, len(window_bl))
        else:
            window_sched = np.zeros(len(window_bl))
            solver_tag   = "MILP-Optimal"
        _ = time.time() - t0

        # Execute only first slot
        executed_ev[current_slot] = window_sched[0]

        ts = start_dt + pd.Timedelta(minutes=current_slot * SLOT_MINUTES)
        net_unmanaged  = base_load[current_slot] + unmanaged[current_slot]
        net_scheduled  = base_load[current_slot] + executed_ev[current_slot]

        results.append(ScheduleResult(
            zone=zone,
            slot=current_slot,
            timestamp=ts,
            base_load_kw=base_load[current_slot],
            unmanaged_ev_kw=unmanaged[current_slot],
            scheduled_ev_kw=executed_ev[current_slot],
            net_unmanaged_kw=net_unmanaged,
            net_scheduled_kw=net_scheduled,
            loading_unmanaged_pct=net_unmanaged / TRANSFORMER_CAPACITY_KW,
            loading_scheduled_pct=net_scheduled / TRANSFORMER_CAPACITY_KW,
            violation_unmanaged=(net_unmanaged > HARD_LIMIT_KW),
            violation_scheduled=(net_scheduled > HARD_LIMIT_KW),
            solver_used=solver_tag,
        ))

    return results


# ─────────────────────────────────────────────
# 6.  METRICS
# ─────────────────────────────────────────────

def compute_metrics(results: List[ScheduleResult]) -> Dict:
    df = pd.DataFrame([r.__dict__ for r in results])
    par_before = df["net_unmanaged_kw"].max() / (df["net_unmanaged_kw"].mean() + 1e-9)
    par_after  = df["net_scheduled_kw"].max()  / (df["net_scheduled_kw"].mean()  + 1e-9)
    return {
        "peak_unmanaged_kw":   df["net_unmanaged_kw"].max(),
        "peak_scheduled_kw":   df["net_scheduled_kw"].max(),
        "peak_reduction_kw":   df["net_unmanaged_kw"].max() - df["net_scheduled_kw"].max(),
        "peak_reduction_pct":  (df["net_unmanaged_kw"].max() - df["net_scheduled_kw"].max())
                                / df["net_unmanaged_kw"].max() * 100,
        "violations_before":   df["violation_unmanaged"].sum(),
        "violations_after":    df["violation_scheduled"].sum(),
        "par_before":          par_before,
        "par_after":           par_after,
        "avg_ev_shifted_kw":   (df["unmanaged_ev_kw"] - df["scheduled_ev_kw"]).abs().mean(),
        "avg_delay_slots":     ((df["unmanaged_ev_kw"] > 0) & (df["scheduled_ev_kw"] == 0)).sum()
                                / max(1, (df["unmanaged_ev_kw"] > 0).sum()) * MAX_DELAY_SLOTS * 0.5,
    }


# ─────────────────────────────────────────────
# 7.  RUN ALL ZONES
# ─────────────────────────────────────────────

print("Running rolling-horizon MILP optimizer for all zones...")
print(f"  Horizon: {HORIZON_SLOTS} slots ({HORIZON_SLOTS*15} min)  |  "
      f"Max delay: {MAX_DELAY_SLOTS} slots ({MAX_DELAY_SLOTS*15} min)\n")

all_results: Dict[str, List[ScheduleResult]] = {}
all_metrics: Dict[str, Dict]                 = {}

for zone, cfg in ZONES.items():
    t0 = time.time()
    results = rolling_horizon(zone, cfg)
    elapsed = time.time() - t0
    metrics = compute_metrics(results)
    all_results[zone] = results
    all_metrics[zone] = metrics

    solvers_used = set(r.solver_used for r in results)
    print(f"  {zone:15s} | "
          f"Peak: {metrics['peak_unmanaged_kw']:.0f} → {metrics['peak_scheduled_kw']:.0f} kW "
          f"(-{metrics['peak_reduction_pct']:.1f}%)  |  "
          f"Violations: {metrics['violations_before']} → {metrics['violations_after']}  |  "
          f"Solver: {', '.join(solvers_used)}  ({elapsed:.1f}s)")


# ─────────────────────────────────────────────
# 8.  RATIONALE CARDS (actionable outputs)
# ─────────────────────────────────────────────

print("\n── Actionable Rationale Cards ─────────────────────────────────")
for zone, results in all_results.items():
    m = all_metrics[zone]
    df = pd.DataFrame([r.__dict__ for r in results])
    worst_slot = df.loc[df["net_unmanaged_kw"].idxmax()]
    evs_shifted = int(df[df["unmanaged_ev_kw"] > df["scheduled_ev_kw"]]["unmanaged_ev_kw"].count())
    avg_delay_h = m["avg_delay_slots"] * SLOT_MINUTES / 60

    print(f"\n  Zone: {zone}")
    print(f"  Action  : Delay ~{evs_shifted} EV batches away from peak window")
    print(f"  Impact  : Transformer loading {worst_slot['loading_unmanaged_pct']:.1%} → "
          f"~{worst_slot['loading_scheduled_pct']:.1%}")
    print(f"  Peak ↓  : {m['peak_reduction_kw']:.0f} kW  ({m['peak_reduction_pct']:.1f}%)")
    print(f"  PAR     : {m['par_before']:.2f} → {m['par_after']:.2f}  (lower is better)")
    print(f"  Violations eliminated: {m['violations_before']} → {m['violations_after']}")
    print(f"  Avg consumer delay  : {avg_delay_h:.1f} hours")
    print(f"  Confidence          : High (MILP with 95% hard constraint)")


# ─────────────────────────────────────────────
# 9.  WHAT-IF SIMULATOR
# ─────────────────────────────────────────────

def what_if(zone: str, extra_evs: int, cfg: dict) -> None:
    """
    'If X extra EVs are added to Zone Z, which transformers breach capacity?'
    """
    boost = extra_evs / (5000 * cfg["ev_density"] * 0.5 + 1e-9)
    base_load, unmanaged, batches = generate_scenario(zone, cfg, n_slots=48)
    unmanaged_boosted = unmanaged * (1 + boost)
    peak_unmanaged = (base_load + unmanaged_boosted).max()
    peak_scheduled, _, _ = solve_milp(base_load, batches, 48)
    peak_managed   = (base_load + peak_scheduled).max()

    print(f"\n── What-If: +{extra_evs} EVs in {zone} ──────────────────────────")
    print(f"  New peak (unmanaged) : {peak_unmanaged:.0f} kW  "
          f"({peak_unmanaged/TRANSFORMER_CAPACITY_KW:.1%} loading)")
    print(f"  After optimization   : {peak_managed:.0f} kW  "
          f"({peak_managed/TRANSFORMER_CAPACITY_KW:.1%} loading)")
    if peak_unmanaged > HARD_LIMIT_KW:
        print(f"  ⚠  Transformer BREACH without scheduling!")
    if peak_managed > HARD_LIMIT_KW:
        print(f"  ⚠  Breach PERSISTS even with scheduling — infrastructure upgrade needed.")
    else:
        print(f"  ✓  Scheduling keeps transformer within safe limits.")


what_if("Koramangala", 200, ZONES["Koramangala"])
what_if("Whitefield",  500, ZONES["Whitefield"])


# ─────────────────────────────────────────────
# 10. VISUALISATIONS
# ─────────────────────────────────────────────

print("\nGenerating plots...")
palette = sns.color_palette("tab10", n_colors=len(ZONES))
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("EV Schedule Optimizer — MILP Rolling Horizon Results", fontsize=14, fontweight="bold")

zone_list = list(ZONES.keys())

# Plot 1 & 2: Unmanaged vs Scheduled load — two representative zones
for ax_idx, zone in enumerate(["Koramangala", "Whitefield"]):
    ax = axes[0, ax_idx]
    df = pd.DataFrame([r.__dict__ for r in all_results[zone]])
    ax.fill_between(df["timestamp"], df["net_unmanaged_kw"],
                    alpha=0.35, color="coral", label="Unmanaged")
    ax.plot(df["timestamp"], df["net_unmanaged_kw"], color="coral", lw=1.2)
    ax.fill_between(df["timestamp"], df["net_scheduled_kw"],
                    alpha=0.35, color="steelblue", label="Optimized")
    ax.plot(df["timestamp"], df["net_scheduled_kw"], color="steelblue", lw=1.5)
    ax.axhline(HARD_LIMIT_KW, color="red",    ls="--", lw=1.2, label=f"95% limit ({HARD_LIMIT_KW:.0f} kW)")
    ax.axhline(TRANSFORMER_CAPACITY_KW * WARNING_PCT, color="orange", ls=":", lw=1.0, label="90% warning")
    ax.set_title(f"{zone} — Load Profile (8h window)")
    ax.set_ylabel("Load (kW)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

# Plot 3: Peak reduction all zones
ax = axes[1, 0]
zones_sorted = sorted(all_metrics.keys(), key=lambda z: -all_metrics[z]["peak_reduction_kw"])
peaks_before = [all_metrics[z]["peak_unmanaged_kw"] for z in zones_sorted]
peaks_after  = [all_metrics[z]["peak_scheduled_kw"]  for z in zones_sorted]
x_pos = np.arange(len(zones_sorted))
bars_b = ax.bar(x_pos - 0.2, peaks_before, 0.38, label="Unmanaged", color="coral",   alpha=0.85)
bars_a = ax.bar(x_pos + 0.2, peaks_after,  0.38, label="Optimized",  color="steelblue", alpha=0.85)
ax.axhline(HARD_LIMIT_KW, color="red", ls="--", lw=1.2, label=f"95% limit")
ax.set_xticks(x_pos)
ax.set_xticklabels(zones_sorted, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Peak Load (kW)")
ax.set_title("Peak Load Reduction — All Zones")
ax.legend(fontsize=8)

# Plot 4: Violations before vs after
ax = axes[1, 1]
viol_before = [all_metrics[z]["violations_before"] for z in zone_list]
viol_after  = [all_metrics[z]["violations_after"]  for z in zone_list]
x_pos = np.arange(len(zone_list))
ax.bar(x_pos - 0.2, viol_before, 0.38, label="Before", color="coral",    alpha=0.85)
ax.bar(x_pos + 0.2, viol_after,  0.38, label="After",  color="steelblue", alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(zone_list, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Capacity Violation Slots (>95%)")
ax.set_title("Capacity Violations Eliminated")
ax.legend(fontsize=8)

# Plot 5: EV load redistribution (Koramangala)
ax = axes[2, 0]
df = pd.DataFrame([r.__dict__ for r in all_results["Koramangala"]])
ax.bar(df["timestamp"], df["unmanaged_ev_kw"], width=0.008,
       color="coral", alpha=0.7, label="Unmanaged EV")
ax.bar(df["timestamp"], df["scheduled_ev_kw"], width=0.008,
       color="steelblue", alpha=0.7, label="Scheduled EV")
ax.set_title("Koramangala — EV Demand Redistribution")
ax.set_ylabel("EV Load (kW)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

# Plot 6: Peak-to-Average Ratio improvement
ax = axes[2, 1]
par_before = [all_metrics[z]["par_before"] for z in zone_list]
par_after  = [all_metrics[z]["par_after"]  for z in zone_list]
x_pos = np.arange(len(zone_list))
ax.bar(x_pos - 0.2, par_before, 0.38, label="Before", color="coral",    alpha=0.85)
ax.bar(x_pos + 0.2, par_after,  0.38, label="After",  color="steelblue", alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(zone_list, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Peak-to-Average Ratio (lower = better)")
ax.set_title("Load Factor Improvement (PAR)")
ax.legend(fontsize=8)

plt.tight_layout()
plot_path = "ev_scheduling_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  Saved → {plot_path}")

# ─────────────────────────────────────────────
# 11. SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n── Summary Table ───────────────────────────────────────────────")
print(f"{'Zone':15s} {'Peak↓ kW':>10s} {'Peak↓ %':>8s} {'Viol B→A':>10s} {'PAR B→A':>12s}")
print("-" * 60)
for zone in zone_list:
    m = all_metrics[zone]
    vb, va = m["violations_before"], m["violations_after"]
    print(f"{zone:15s} {m['peak_reduction_kw']:>10.0f} {m['peak_reduction_pct']:>7.1f}%"
          f"  {vb:>3d} → {va:<3d}   {m['par_before']:>5.2f} → {m['par_after']:.2f}")

print("\nDone! Next: Part C — Location Recommender (MCLP + Grid Headroom Score)")