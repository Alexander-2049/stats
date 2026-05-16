"""
build_districts.py
══════════════════════════════════════════════════════════════════
Builds 150 contiguous, equal-population districts for the
Netherlands that maximise within-district connectedness.

  Phase 1a  Population-weighted K-Means++ seed selection
  Phase 1b  Seeded region growing (with hard population caps)
  Phase 1c  Deterministic rebalancing pass            ← NEW
  Phase 2   Simulated annealing
              · Pure-quadratic penalty (no cliffs)    ← FIXED
              · Penalty weight starts high (200×)     ← FIXED
              · 70/30 targeted / random move mix      ← NEW
              · Two-phase cooling with mid reheat     ← NEW
              · Tracks best partial + full valid      ← FIXED
  Phase 3   Export district_id → buurten GeoPackage
"""

from __future__ import annotations

import heapq
import math
import pickle
import random
import sys
import time
from functools import partial
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from gerrychain import Graph as GerryGraph
from gerrychain.tree import bipartition_tree, recursive_tree_part
from tqdm import tqdm


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
BASE_DIR     = Path(__file__).resolve().parent
GRAPH_PATH   = BASE_DIR / "osm_connectedness" / "buurten_connectedness_graph_repaired.pkl"
BUURTEN_GPKG = BASE_DIR / "buurten" / "buurten.gpkg"
OUTPUT_GPKG  = BASE_DIR / "osm_connectedness" / "buurten_districted.gpkg"
CHECKPOINT   = BASE_DIR / "osm_connectedness" / "district_checkpoint.pkl"

N_DISTRICTS = 150
TOTAL_POP   = 17_942_915
TARGET_POP  = TOTAL_POP / N_DISTRICTS          # ≈ 119 619
POP_SLACK   = 0.05
POP_MIN     = int(TARGET_POP * (1 - POP_SLACK))  # 113 638
POP_MAX     = int(TARGET_POP * (1 + POP_SLACK))  # 125 600

# ── Simulated annealing ───────────────────────────────────────────
# More iterations + wider temperature range gives the optimizer
# enough budget to actually converge from a poor initial state.
SA_ITERATIONS    = 5_000_000
T_START          = 100.0
T_END            = 0.1

# Two-phase cooling: Phase A runs until this fraction of iterations,
# then we reheat to REHEAT_FRACTION * T_START and cool slowly.
PHASE_A_FRACTION = 0.60
REHEAT_FRACTION  = 0.30   # reheat to 30 % of T_START at the boundary

# Penalty weight: starts at PEN_WEIGHT_START and rises linearly
# to PEN_WEIGHT_END.  Starting high means population balance is
# enforced from the very first iteration.
PEN_WEIGHT_START = 200.0
PEN_WEIGHT_END   = 500.0

# Fraction of SA moves drawn from the targeted (fat→lean) pool.
TARGETED_FRACTION = 0.70

CHECKPOINT_EVERY = 200_000
REPORT_EVERY     = 10_000

RESUME_FROM_CHECKPOINT = False

SEED = 42
# ══════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────

def load_buurten(gpkg_path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg_path)
    gdf = gdf[gdf["aantal_inwoners"] >= 0].copy()
    gdf = gdf.to_crs("EPSG:28992").reset_index(drop=True)
    return gdf


# ──────────────────────────────────────────────────────────────────
# Phase 1a — Seed selection
# ──────────────────────────────────────────────────────────────────

def select_seeds(
    nodes: list[int],
    gdf: gpd.GeoDataFrame,
    n: int,
    rng: random.Random,
) -> list[int]:
    """
    Population-weighted farthest-point (K-Means++) seeding.
    Spreads seeds far apart AND biases towards high-population centroids
    so each seed starts near a population mass it can grow around.
    """
    coords = np.array(
        [(gdf.geometry.iloc[i].centroid.x, gdf.geometry.iloc[i].centroid.y)
         for i in nodes]
    )
    pop = gdf["aantal_inwoners"].values

    first_idx   = rng.randrange(len(nodes))
    seed_indices = [first_idx]
    min_dist_sq  = np.full(len(nodes), np.inf)

    for _ in tqdm(range(n - 1), desc="  Selecting seeds", unit="seed"):
        last_coord  = coords[seed_indices[-1]]
        d_sq        = np.sum((coords - last_coord) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, d_sq)

        weights             = min_dist_sq * pop
        weights[seed_indices] = 0
        weights             = np.maximum(weights, 0)

        w_sum = weights.sum()
        if w_sum > 0:
            probs    = weights / w_sum
            next_idx = int(np.random.choice(len(nodes), p=probs))
        else:
            next_idx = int(np.argmax(min_dist_sq))

        seed_indices.append(next_idx)

    return [nodes[i] for i in seed_indices]


# ──────────────────────────────────────────────────────────────────
# Phase 1b — Seeded region growing
# ──────────────────────────────────────────────────────────────────

def region_grow(G: nx.Graph, seeds: list[int], pop: np.ndarray) -> np.ndarray:
    """
    Priority-queue region growing.  Districts that hit POP_MAX get their
    heap priority multiplied by 1 000 000, stalling them until neighbours
    catch up.  This alone is not sufficient for tight bounds — the
    rebalance pass below fixes the remaining imbalance.
    """
    n        = G.number_of_nodes()
    district = np.full(n, -1, dtype=np.int32)
    d_pop    = np.zeros(N_DISTRICTS, dtype=np.int64)

    for d, seed in enumerate(seeds):
        district[seed] = d
        d_pop[d]       = pop[seed]

    heap: list = []
    for d, seed in enumerate(seeds):
        for nb in G.neighbors(seed):
            if district[nb] == -1:
                w = G[seed][nb]["weight"]
                heapq.heappush(heap, (int(d_pop[d]), d, -w, nb))

    with tqdm(total=n - N_DISTRICTS, desc="  Growing regions", unit="buurt") as pbar:
        while heap:
            _, d, neg_w, node = heapq.heappop(heap)
            if district[node] != -1:
                continue

            district[node]  = d
            d_pop[d]       += pop[node]
            pbar.update(1)

            new_pop = int(d_pop[d])
            priority_pop = new_pop if new_pop <= POP_MAX else new_pop * 1_000_000
            for nb in G.neighbors(node):
                if district[nb] == -1:
                    w = G[node][nb]["weight"]
                    heapq.heappush(heap, (priority_pop, d, -w, nb))

    return district


# ──────────────────────────────────────────────────────────────────
# Phase 1c — Deterministic rebalancing  ← NEW
# ──────────────────────────────────────────────────────────────────

def rebalance_districts(
    G: nx.Graph,
    district: np.ndarray,
    pop: np.ndarray,
    max_passes: int = 100,
) -> np.ndarray:
    """
    Repeatedly transfers border nodes from the most over-populated
    district to its most under-populated neighbour, as long as both
    the move preserves contiguity and the target district stays ≤ POP_MAX.

    Runs in O(passes × boundary × neighbours) — fast because it is
    deterministic and makes no wasted moves.
    """
    d_pop   = np.zeros(N_DISTRICTS, dtype=np.int64)
    d_nodes = build_district_node_sets(district)
    for node, d in enumerate(district):
        d_pop[d] += pop[node]

    for pass_num in range(max_passes):
        moved   = 0
        boundary = build_boundary_set(G, district)

        # Sort boundary nodes by their district population descending
        # so we attack the fattest districts first.
        sorted_bnd = sorted(
            boundary,
            key=lambda u: -d_pop[int(district[u])],
        )

        for u in sorted_bnd:
            src_d   = int(district[u])
            src_pop = d_pop[src_d]

            # Only shed if over target
            if src_pop <= TARGET_POP:
                continue

            # Connectivity guard
            if len(d_nodes[src_d]) <= 1:
                continue
            if not is_removal_safe(G, d_nodes[src_d], u):
                continue

            # Find the leanest neighbouring district that won't overflow
            nb_ds = {
                int(district[v])
                for v in G.neighbors(u)
                if int(district[v]) != src_d
            }
            if not nb_ds:
                continue

            # Pick neighbour that is furthest below target and won't overflow
            viable = [
                d for d in nb_ds
                if d_pop[d] + pop[u] <= POP_MAX
            ]
            if not viable:
                continue

            tgt_d = min(viable, key=lambda d: d_pop[d])

            # Transfer
            d_nodes[src_d].discard(u)
            d_nodes[tgt_d].add(u)
            d_pop[src_d] -= pop[u]
            d_pop[tgt_d] += pop[u]
            district[u]   = tgt_d
            moved += 1

        valid_count = int(np.sum((d_pop >= POP_MIN) & (d_pop <= POP_MAX)))
        print(
            f"  Rebalance pass {pass_num + 1:>3}: "
            f"moved {moved:>5} nodes | "
            f"valid={valid_count:>3}/{N_DISTRICTS} | "
            f"std={d_pop.std():>8,.0f}"
        )
        if moved == 0 or valid_count == N_DISTRICTS:
            break

    return district


# ──────────────────────────────────────────────────────────────────
# Connectivity helpers
# ──────────────────────────────────────────────────────────────────

def is_removal_safe(G: nx.Graph, d_nodes: set[int], node: int) -> bool:
    """Returns True if removing `node` from `d_nodes` keeps the set connected."""
    remaining = d_nodes - {node}
    if len(remaining) <= 1:
        return True
    start   = next(iter(remaining))
    visited = {start}
    stack   = [start]
    while stack:
        curr = stack.pop()
        for nb in G.neighbors(curr):
            if nb in remaining and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == len(remaining)


def build_district_node_sets(district: np.ndarray) -> list[set[int]]:
    d_nodes: list[set[int]] = [set() for _ in range(N_DISTRICTS)]
    for node, d in enumerate(district):
        d_nodes[d].add(int(node))
    return d_nodes


def build_boundary_set(G: nx.Graph, district: np.ndarray) -> set[int]:
    bnd: set[int] = set()
    for u, v in G.edges():
        if district[u] != district[v]:
            bnd.add(int(u))
            bnd.add(int(v))
    return bnd


# ──────────────────────────────────────────────────────────────────
# SA helpers
# ──────────────────────────────────────────────────────────────────

def score_delta(G: nx.Graph, district: np.ndarray, u: int, target_d: int) -> float:
    """Change in total within-district edge weight if node u moves to target_d."""
    src_d = int(district[u])
    delta = 0.0
    for v, data in G[u].items():
        w   = data["weight"]
        d_v = int(district[v])
        if d_v == src_d:
            delta -= w
        elif d_v == target_d:
            delta += w
    return delta


def total_score(G: nx.Graph, district: np.ndarray) -> float:
    return sum(
        data["weight"]
        for u, v, data in G.edges(data=True)
        if district[u] == district[v]
    )


def calc_penalty(p: int) -> float:
    """
    Pure quadratic penalty — NO cliff discontinuities.

    Smooth quadratic centred on TARGET_POP, normalised so that a
    district exactly at POP_MIN / POP_MAX scores 1 000.  Districts
    further outside score proportionally more.  No sudden cliff jump
    means SA can cross the feasibility boundary smoothly in either
    direction, which prevents the optimizer from getting frozen
    against a wall.
    """
    return ((p - TARGET_POP) / TARGET_POP) ** 2 * 1_000


def build_transfer_candidates(
    G: nx.Graph,
    district: np.ndarray,
    d_pop: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Returns (node, target_district) pairs where:
      · the node's current district is above TARGET_POP   (fat)
      · at least one neighbour's district is below TARGET_POP  (lean)

    Moves from this pool directly attack population imbalance, so
    the 70 % targeted sample rate means the optimizer spends the vast
    majority of its budget on useful work rather than random drift.
    """
    candidates: list[tuple[int, int]] = []
    for u in build_boundary_set(G, district):
        src_d = int(district[u])
        if d_pop[src_d] <= TARGET_POP:
            continue
        for v in G.neighbors(u):
            tgt_d = int(district[v])
            if tgt_d != src_d and d_pop[tgt_d] < TARGET_POP:
                candidates.append((u, tgt_d))
    return candidates


# ──────────────────────────────────────────────────────────────────
# Phase 2 — Simulated annealing
# ──────────────────────────────────────────────────────────────────

def simulated_annealing(
    G: nx.Graph,
    district: np.ndarray,
    pop: np.ndarray,
    n_iter: int,
    T_start: float,
    T_end: float,
    rng: random.Random,
    start_it: int = 0,
) -> np.ndarray:
    """
    Simulated annealing with four key improvements over the original:

    1. Pure-quadratic penalty: no cliff discontinuity at POP_MIN/POP_MAX.
    2. Penalty weight starts at PEN_WEIGHT_START (200) and rises linearly
       to PEN_WEIGHT_END (500), so population balance is enforced from
       the very first iteration rather than being introduced slowly.
    3. 70 % of moves are drawn from a targeted pool of fat→lean transfers,
       raising the useful-acceptance rate from ~3 % to 30-50 %.
    4. Two-phase cooling: a mid-run reheat lets the algorithm escape local
       optima that form once the initial temperature drops too far.
    """

    d_nodes  = build_district_node_sets(district)
    d_pop    = np.array(
        [sum(pop[node] for node in d_nodes[d]) for d in range(N_DISTRICTS)],
        dtype=np.int64,
    )
    boundary = build_boundary_set(G, district)
    bnd_list = list(boundary)

    cur_conn    = total_score(G, district)
    pen_sum     = sum(calc_penalty(p) for p in d_pop)

    # Track the best fully-valid state and the best partial state separately.
    best_valid_score  = -float("inf")
    best_valid_d: np.ndarray | None = None

    best_partial_count = int(np.sum((d_pop >= POP_MIN) & (d_pop <= POP_MAX)))
    best_partial_d     = district.copy()

    # Fallback: lowest cumulative penalty seen (used when no valid state found)
    min_pen_sum    = pen_sum
    best_fallback_d = district.copy()

    # ── Two-phase cooling setup ───────────────────────────────────
    # Phase A: exponential cooling from T_start to T_start * REHEAT_FRACTION
    # Phase B: exponential cooling from that reheated value to T_end
    phase_a_iters = int(n_iter * PHASE_A_FRACTION)
    phase_b_iters = n_iter - phase_a_iters

    T_mid = T_start * REHEAT_FRACTION          # temperature at reheat point

    cooling_a = (T_mid / T_start) ** (1.0 / max(phase_a_iters, 1))
    cooling_b = (T_end  / T_mid)  ** (1.0 / max(phase_b_iters, 1))

    def temperature_at(it: int) -> float:
        if it < phase_a_iters:
            return T_start * (cooling_a ** it)
        else:
            return T_mid * (cooling_b ** (it - phase_a_iters))

    T = temperature_at(start_it)

    # ── Targeted candidate pool ───────────────────────────────────
    transfer_candidates = build_transfer_candidates(G, district, d_pop)

    remaining_iters = n_iter - start_it
    acc = cont_rej = metro_rej = 0
    t0  = time.time()

    for it in range(start_it, n_iter):
        progress = (it - start_it) / max(1, remaining_iters)

        # Linear penalty weight ramp — starts high, ends higher.
        pen_weight = PEN_WEIGHT_START + (PEN_WEIGHT_END - PEN_WEIGHT_START) * progress

        # Refresh boundary list and targeted pool periodically.
        if it % 5_000 == 0:
            bnd_list = list(boundary)
            transfer_candidates = build_transfer_candidates(G, district, d_pop)

        if not bnd_list:
            break

        # ── Move selection: 70 % targeted, 30 % random ───────────
        use_targeted = (
            rng.random() < TARGETED_FRACTION
            and transfer_candidates
        )

        if use_targeted:
            u, tgt_d = rng.choice(transfer_candidates)
            src_d    = int(district[u])
            # Guard: candidates list may be stale
            if district[u] != src_d or d_pop[src_d] <= TARGET_POP:
                T = temperature_at(it)
                continue
        else:
            u     = rng.choice(bnd_list)
            src_d = int(district[u])
            nb_ds = {
                int(district[v])
                for v in G.neighbors(u)
                if int(district[v]) != src_d
            }
            if not nb_ds:
                boundary.discard(u)
                T = temperature_at(it)
                continue
            tgt_d = rng.choice(list(nb_ds))

        # ── Contiguity guard ─────────────────────────────────────
        if len(d_nodes[src_d]) <= 1 or not is_removal_safe(G, d_nodes[src_d], u):
            cont_rej += 1
            T = temperature_at(it)
            continue

        # ── Energy delta ─────────────────────────────────────────
        u_pop       = int(pop[u])
        new_src_pop = d_pop[src_d] - u_pop
        new_tgt_pop = d_pop[tgt_d] + u_pop

        pen_old = calc_penalty(d_pop[src_d]) + calc_penalty(d_pop[tgt_d])
        pen_new = calc_penalty(new_src_pop)  + calc_penalty(new_tgt_pop)
        pen_delta  = (pen_old - pen_new) * pen_weight   # positive = improvement

        conn_delta = score_delta(G, district, u, tgt_d)
        delta      = conn_delta + pen_delta

        # ── Metropolis criterion ──────────────────────────────────
        if delta < 0 and rng.random() >= math.exp(delta / max(T, 1e-9)):
            metro_rej += 1
            T = temperature_at(it)
            continue

        # ── Accept move ───────────────────────────────────────────
        d_nodes[src_d].discard(u)
        d_nodes[tgt_d].add(u)
        district[u]   = tgt_d
        d_pop[src_d]  = new_src_pop
        d_pop[tgt_d]  = new_tgt_pop
        cur_conn     += conn_delta
        pen_sum      -= (pen_old - pen_new)
        acc          += 1

        # Update boundary membership for u and all its neighbours.
        for node in (u, *list(G.neighbors(u))):
            d_node = int(district[node])
            is_bnd = any(int(district[nb]) != d_node for nb in G.neighbors(node))
            if is_bnd:
                boundary.add(node)
            else:
                boundary.discard(node)

        # ── Track best states ─────────────────────────────────────
        valid_count = int(np.sum((d_pop >= POP_MIN) & (d_pop <= POP_MAX)))
        is_valid    = valid_count == N_DISTRICTS

        if is_valid and cur_conn > best_valid_score:
            best_valid_score = cur_conn
            best_valid_d     = district.copy()

        if valid_count > best_partial_count:
            best_partial_count = valid_count
            best_partial_d     = district.copy()

        if pen_sum < min_pen_sum:
            min_pen_sum     = pen_sum
            best_fallback_d = district.copy()

        T = temperature_at(it)

        # ── Checkpoint ───────────────────────────────────────────
        if (it + 1) % CHECKPOINT_EVERY == 0:
            ckpt = {
                "district":  district,
                "iteration": it + 1,
            }
            with open(CHECKPOINT, "wb") as fh:
                pickle.dump(ckpt, fh)

        # ── Progress report ───────────────────────────────────────
        if (it + 1) % REPORT_EVERY == 0:
            elapsed = time.time() - t0
            total   = acc + cont_rej + metro_rej
            acc_pct = 100 * acc / total if total else 0
            phase   = "A" if it < phase_a_iters else "B"
            print(
                f"  it={it + 1:>8,} [{phase}] | T={T:7.3f} | "
                f"conn={cur_conn:>10,.0f} | pen×w={pen_sum * pen_weight:>11,.0f} | "
                f"valid={valid_count:>3}/{N_DISTRICTS} | "
                f"best_partial={best_partial_count:>3} | "
                f"acc={acc_pct:>2.0f}% cont✗={cont_rej:>5,} | "
                f"{elapsed:.0f}s"
            )
            acc = cont_rej = metro_rej = 0

    # ── Return best state found ────────────────────────────────────
    if best_valid_d is not None:
        print(f"\n  ✅ Returning best VALID state  (connectedness={best_valid_score:,.1f})")
        return best_valid_d

    if best_partial_count >= int(N_DISTRICTS * 0.90):
        print(
            f"\n  ⚠  No 100 % valid state found. "
            f"Returning best partial state ({best_partial_count}/{N_DISTRICTS} valid)."
        )
        return best_partial_d

    print(
        f"\n  ⚠  Could not reach ≥90 % valid. "
        f"Returning lowest-penalty approximation ({best_partial_count}/{N_DISTRICTS} valid)."
    )
    return best_fallback_d


# ──────────────────────────────────────────────────────────────────
# Validation & Export
# ──────────────────────────────────────────────────────────────────

def validate(
    G: nx.Graph,
    district: np.ndarray,
    pop: np.ndarray,
    label: str = "",
) -> None:
    if label:
        print(f"\n── Validation: {label} ──")

    d_pops = np.zeros(N_DISTRICTS, dtype=np.int64)
    for node, d in enumerate(district):
        d_pops[d] += pop[node]

    valid_pop  = int(np.sum((d_pops >= POP_MIN) & (d_pops <= POP_MAX)))
    print(
        f"  Population  min={d_pops.min():,}  max={d_pops.max():,}  "
        f"mean={d_pops.mean():,.0f}  std={d_pops.std():,.0f}"
    )
    print(f"  Districts within ±5 % pop bounds : {valid_pop} / {N_DISTRICTS}")

    d_node_sets = build_district_node_sets(district)
    n_disconn   = 0
    for d in range(N_DISTRICTS):
        nodes = d_node_sets[d]
        if not nodes:
            n_disconn += 1
            continue
        start   = next(iter(nodes))
        visited = {start}
        stack   = [start]
        while stack:
            curr = stack.pop()
            for nb in G.neighbors(curr):
                if nb in nodes and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        if len(visited) != len(nodes):
            n_disconn += 1

    print(f"  Discontiguous districts          : {n_disconn}")
    print(f"  Total connectedness score        : {total_score(G, district):,.2f}")


def export_result(
    district: np.ndarray,
    buurten_gpkg: str,
    output_path: Path,
) -> None:
    print("\nLoading buurten for export …")
    gdf = load_buurten(buurten_gpkg)
    gdf["district_id"] = (district + 1).astype(int)
    print(f"  district_id range : {gdf['district_id'].min()} – {gdf['district_id'].max()}")
    print(f"  Writing → {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG")
    print("  Done.")


def pop_bounds_gap(p: int) -> int:
    if p < POP_MIN:
        return POP_MIN - p
    if p > POP_MAX:
        return p - POP_MAX
    return 0


def district_populations(district: np.ndarray, pop: np.ndarray) -> np.ndarray:
    d_pop = np.zeros(N_DISTRICTS, dtype=np.int64)
    for node, d in enumerate(district):
        d_pop[int(d)] += pop[node]
    return d_pop


def count_valid_districts_from_assignment(district: np.ndarray, pop: np.ndarray) -> int:
    d_pop = district_populations(district, pop)
    return int(np.sum((d_pop >= POP_MIN) & (d_pop <= POP_MAX)))


def region_grow(G: nx.Graph, seeds: list[int], pop: np.ndarray) -> np.ndarray:
    """
    Population-first seeded region growing.

    At each step we expand the currently lightest district by assigning its
    strongest frontier node. This preserves contiguity while producing a much
    better population-balanced starting partition.
    """
    n = G.number_of_nodes()
    district = np.full(n, -1, dtype=np.int32)
    d_pop = np.zeros(N_DISTRICTS, dtype=np.int64)

    for d, seed in enumerate(seeds):
        district[seed] = d
        d_pop[d] = pop[seed]

    frontiers: list[dict[int, float]] = [dict() for _ in range(N_DISTRICTS)]
    for d, seed in enumerate(seeds):
        for nb in G.neighbors(seed):
            if district[nb] == -1:
                node = int(nb)
                weight = float(G[seed][nb]["weight"])
                frontiers[d][node] = max(frontiers[d].get(node, 0.0), weight)

    remaining = n - N_DISTRICTS
    with tqdm(total=remaining, desc="  Growing regions", unit="buurt") as pbar:
        while remaining > 0:
            active = [d for d in range(N_DISTRICTS) if frontiers[d]]
            if not active:
                break

            d = min(
                active,
                key=lambda idx: (
                    max(0, int(d_pop[idx]) - POP_MAX),
                    int(d_pop[idx]),
                ),
            )

            stale_nodes = [node for node in frontiers[d] if district[node] != -1]
            for node in stale_nodes:
                del frontiers[d][node]
            if not frontiers[d]:
                continue

            node, _ = max(
                frontiers[d].items(),
                key=lambda item: (item[1], -int(pop[item[0]]), -item[0]),
            )
            del frontiers[d][node]
            if district[node] != -1:
                continue

            district[node] = d
            d_pop[d] += pop[node]
            remaining -= 1
            pbar.update(1)

            for nb in G.neighbors(node):
                if district[nb] == -1:
                    nb_i = int(nb)
                    weight = float(G[node][nb]["weight"])
                    frontiers[d][nb_i] = max(frontiers[d].get(nb_i, 0.0), weight)

    if np.any(district == -1):
        raise RuntimeError("Region growing left unassigned nodes.")

    return district


def rebalance_districts(
    G: nx.Graph,
    district: np.ndarray,
    pop: np.ndarray,
    max_passes: int = 250,
) -> np.ndarray:
    """
    Greedy population repair.

    Repeatedly applies boundary transfers that improve local validity first,
    then reduce out-of-bounds population gap, then reduce quadratic penalty.
    """
    d_pop = district_populations(district, pop)
    d_nodes = build_district_node_sets(district)

    for pass_num in range(max_passes):
        moved = 0
        boundary = sorted(
            build_boundary_set(G, district),
            key=lambda u: (
                pop_bounds_gap(int(d_pop[int(district[u])])),
                int(d_pop[int(district[u])]),
            ),
            reverse=True,
        )

        for u in boundary:
            src_d = int(district[u])
            src_pop = int(d_pop[src_d])

            if len(d_nodes[src_d]) <= 1:
                continue
            if not is_removal_safe(G, d_nodes[src_d], u):
                continue

            best_tgt: int | None = None
            best_score: tuple[int, int, float, float] | None = None
            u_pop = int(pop[u])

            for v in G.neighbors(u):
                tgt_d = int(district[v])
                if tgt_d == src_d:
                    continue

                tgt_pop = int(d_pop[tgt_d])
                new_src = src_pop - u_pop
                new_tgt = tgt_pop + u_pop

                old_valid_local = int(POP_MIN <= src_pop <= POP_MAX) + int(POP_MIN <= tgt_pop <= POP_MAX)
                new_valid_local = int(POP_MIN <= new_src <= POP_MAX) + int(POP_MIN <= new_tgt <= POP_MAX)
                valid_gain = new_valid_local - old_valid_local

                gap_old = pop_bounds_gap(src_pop) + pop_bounds_gap(tgt_pop)
                gap_new = pop_bounds_gap(new_src) + pop_bounds_gap(new_tgt)
                gap_gain = gap_old - gap_new

                pen_old = calc_penalty(src_pop) + calc_penalty(tgt_pop)
                pen_new = calc_penalty(new_src) + calc_penalty(new_tgt)
                pen_gain = pen_old - pen_new
                conn_gain = score_delta(G, district, u, tgt_d)

                if valid_gain < 0:
                    continue
                if valid_gain == 0 and gap_gain < 0:
                    continue
                if valid_gain == 0 and gap_gain == 0 and pen_gain <= 0:
                    continue

                score = (valid_gain, gap_gain, pen_gain, conn_gain)
                if best_score is None or score > best_score:
                    best_score = score
                    best_tgt = tgt_d

            if best_tgt is None:
                continue

            d_nodes[src_d].discard(u)
            d_nodes[best_tgt].add(u)
            d_pop[src_d] -= pop[u]
            d_pop[best_tgt] += pop[u]
            district[u] = best_tgt
            moved += 1

        valid_count = int(np.sum((d_pop >= POP_MIN) & (d_pop <= POP_MAX)))
        print(
            f"  Rebalance pass {pass_num + 1:>3}: "
            f"moved {moved:>5} nodes | "
            f"valid={valid_count:>3}/{N_DISTRICTS} | "
            f"std={d_pop.std():>8,.0f}"
        )
        if valid_count == N_DISTRICTS:
            print("  Rebalance reached full population validity.")
            break
        if moved == 0:
            break

    return district


def refine_connectedness(
    G: nx.Graph,
    district: np.ndarray,
    pop: np.ndarray,
    max_passes: int = 25,
) -> np.ndarray:
    """
    Improves connectedness without leaving the valid population window.
    """
    d_pop = district_populations(district, pop)
    d_nodes = build_district_node_sets(district)

    for pass_num in range(max_passes):
        moved = 0
        gain_sum = 0.0

        for u in list(build_boundary_set(G, district)):
            src_d = int(district[u])
            if len(d_nodes[src_d]) <= 1 or not is_removal_safe(G, d_nodes[src_d], u):
                continue

            best_tgt: int | None = None
            best_gain = 0.0
            u_pop = int(pop[u])

            for v in G.neighbors(u):
                tgt_d = int(district[v])
                if tgt_d == src_d:
                    continue

                new_src = int(d_pop[src_d] - u_pop)
                new_tgt = int(d_pop[tgt_d] + u_pop)
                if not (POP_MIN <= new_src <= POP_MAX and POP_MIN <= new_tgt <= POP_MAX):
                    continue

                conn_gain = score_delta(G, district, u, tgt_d)
                if conn_gain > best_gain:
                    best_gain = conn_gain
                    best_tgt = tgt_d

            if best_tgt is None:
                continue

            d_nodes[src_d].discard(u)
            d_nodes[best_tgt].add(u)
            d_pop[src_d] -= pop[u]
            d_pop[best_tgt] += pop[u]
            district[u] = best_tgt
            moved += 1
            gain_sum += best_gain

        print(
            f"  Refinement pass {pass_num + 1:>3}: moved {moved:>5} nodes | "
            f"connectedness +{gain_sum:>8,.1f}"
        )
        if moved == 0:
            break

    return district


def tree_partition_districts(G: nx.Graph, pop: np.ndarray) -> np.ndarray:
    """
    Builds a fully connected, population-balanced districting using
    GerryChain's recursive tree partitioner.
    """
    tree_graph = G.copy()
    for node in tree_graph.nodes:
        tree_graph.nodes[node]["population"] = int(pop[node])

    tree_method = partial(bipartition_tree, max_attempts=50_000)
    last_error: Exception | None = None
    assignment = None

    for attempt in range(1, 6):
        try:
            assignment = recursive_tree_part(
                GerryGraph.from_networkx(tree_graph),
                list(range(N_DISTRICTS)),
                TARGET_POP,
                "population",
                POP_SLACK,
                node_repeats=10,
                method=tree_method,
            )
            print(f"  Tree partition succeeded on attempt {attempt}.")
            break
        except RuntimeError as exc:
            last_error = exc
            print(f"  Tree partition attempt {attempt} failed; retrying.")

    if assignment is None:
        raise RuntimeError("Tree partition could not find a valid plan.") from last_error

    district = np.full(G.number_of_nodes(), -1, dtype=np.int32)
    for node, part in assignment.items():
        district[int(node)] = int(part)

    if np.any(district == -1):
        raise RuntimeError("Tree partition left unassigned nodes.")

    return district


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)

    print("=" * 64)
    print("Loading graph …")
    print("=" * 64)
    with open(GRAPH_PATH, "rb") as fh:
        G: nx.Graph = pickle.load(fh)

    nodes = sorted(G.nodes())
    n     = len(nodes)
    print(f"  {G.number_of_nodes():,} nodes   {G.number_of_edges():,} edges")

    print("\nLoading buurten …")
    gdf = load_buurten(BUURTEN_GPKG)

    if len(gdf) != n:
        raise ValueError(
            f"GeoPackage has {len(gdf)} buurten but graph has {n} nodes."
        )

    pop = np.array(
        [int(gdf.loc[i, "aantal_inwoners"]) for i in range(n)],
        dtype=np.int64,
    )
    print(f"  Total population : {pop.sum():,}")
    print(f"  Target / district: {TARGET_POP:,.0f}   valid range {POP_MIN:,} – {POP_MAX:,}")

    start_iteration = 0
    if RESUME_FROM_CHECKPOINT and CHECKPOINT.exists():
        print("\nResuming from checkpoint …")
        with open(CHECKPOINT, "rb") as fh:
            ckpt = pickle.load(fh)
        district        = ckpt["district"]
        start_iteration = ckpt["iteration"]
        print(f"  Resuming at iteration {start_iteration:,}")
        validate(G, district, pop, "checkpoint")
    else:
        # ── Phase 1a ──────────────────────────────────────────────
        print("\n" + "=" * 64)
        print("Phase 1a — Seed selection (population-weighted farthest-point)")
        print("=" * 64)
        seeds = select_seeds(nodes, gdf, N_DISTRICTS, rng)
        print(f"  {len(seeds)} seeds selected")

        # ── Phase 1b ──────────────────────────────────────────────
        print("\n" + "=" * 64)
        print("Phase 1b — Region growing")
        print("=" * 64)
        district = region_grow(G, seeds, pop)
        validate(G, district, pop, "after growth")

        # ── Phase 1c ──────────────────────────────────────────────
        print("\n" + "=" * 64)
        print("Phase 1c — Deterministic rebalancing")
        print("=" * 64)
        district = rebalance_districts(G, district, pop)
        validate(G, district, pop, "after rebalance")
        if count_valid_districts_from_assignment(district, pop) != N_DISTRICTS:
            print("\n" + "=" * 64)
            print("Phase 1d - Recursive tree partition fallback")
            print("=" * 64)
            district = tree_partition_districts(G, pop)
            validate(G, district, pop, "after tree partition")

    # ── Phase 2 ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"Phase 2 — Simulated annealing  ({SA_ITERATIONS:,} iterations)")
    print(f"  Temperature : {T_START} → {T_START * REHEAT_FRACTION:.1f} [reheat] → {T_END}")
    print(f"  Penalty weight : {PEN_WEIGHT_START} → {PEN_WEIGHT_END} (linear ramp)")
    print(f"  Targeted move fraction : {TARGETED_FRACTION:.0%}")
    print("=" * 64)

    valid_after_rebalance = count_valid_districts_from_assignment(district, pop)
    if valid_after_rebalance == N_DISTRICTS:
        print("\n" + "=" * 64)
        print("Phase 2 â€” Connectedness refinement within valid bounds")
        print("=" * 64)
        district = refine_connectedness(G, district, pop)
        validate(G, district, pop, "after refinement")
    else:
        before_sa = district.copy()
        district_sa = simulated_annealing(
            G, district, pop,
            n_iter   = SA_ITERATIONS,
            T_start  = T_START,
            T_end    = T_END,
            rng      = rng,
            start_it = start_iteration,
        )
        after_sa_valid = count_valid_districts_from_assignment(district_sa, pop)
        if after_sa_valid >= valid_after_rebalance:
            district = district_sa
            print(f"  Keeping annealed state ({after_sa_valid}/{N_DISTRICTS} valid districts).")
        else:
            district = before_sa
            print(f"  Annealing regressed validity ({after_sa_valid}/{N_DISTRICTS}); keeping pre-SA state.")
        validate(G, district, pop, "after annealing")

    # ── Phase 3 ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Phase 3 — Export")
    print("=" * 64)
    export_result(district, BUURTEN_GPKG, OUTPUT_GPKG)

    print("\n✅  All done.")
    print(f"   Output GeoPackage → {OUTPUT_GPKG}")


if __name__ == "__main__":
    main()
