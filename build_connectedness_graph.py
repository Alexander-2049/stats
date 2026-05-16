"""
build_connectedness_graph.py
────────────────────────────
Builds a buurt-level connectedness graph from OSM data.

Fixes applied vs. original:
  • MultiLineString → linemerge first, fall back to longest segment only if needed
  • Highway tag guarded against list values (pyrosm edge case)
  • Weight assignment vectorised (no row-wise apply)
  • Centroid map built without iterrows
  • Edge-building loop parallelised with multiprocessing.Pool (WKB serialisation)
  • Multimodal stacking kept intentional (a road with cycle lane scores higher)
  • OSM way-id deduplication NOT applied (multimodal boost is desired)
"""

import os
import time
import pickle
import multiprocessing as mp
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from shapely import wkb as shp_wkb
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from tqdm import tqdm

# ========================= CONFIG =========================
BUURTEN_GPKG = r"C:\Users\ivanm\OneDrive\Документы\buurten.gpkg"
OUTPUT_DIR   = Path("osm_connectedness")
WAYS_GEOJSON = OUTPUT_DIR / "osm" / "netherlands-ways.geojson"

WEIGHT_DICT = {
    'motorway':      5.0,  'motorway_link': 5.0,
    'trunk':         5.0,  'trunk_link':    5.0,
    'primary':       4.0,  'primary_link':  4.0,
    'secondary':     3.0,  'secondary_link':3.0,
    'tertiary':      2.5,  'tertiary_link': 2.5,
    'railway':       4.5,
    'ferry':         6.5,
    'cycle_foot':    2.0,
    'other_road':    1.0,
}

# Number of worker processes for the parallel edge-building step.
# None → use all logical cores. Reduce if RAM is tight.
N_WORKERS  = None
CHUNK_SIZE = 500   # ways per task chunk sent to each worker
# ==========================================================

OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Worker initialiser: injects the centroid map into each worker process.
# On Windows, fork() is unavailable so globals must be passed explicitly.
# ─────────────────────────────────────────────────────────────────────────────
_worker_centroid_map: dict[int, Point] = {}

def _worker_init(centroid_map: dict[int, Point]) -> None:
    global _worker_centroid_map
    _worker_centroid_map = centroid_map


def _process_way(args: tuple) -> list[tuple[int, int, float]]:
    """
    Process a single OSM way: project buurt centroids onto the way geometry,
    sort them by projection distance, and return consecutive (u, v, weight) edges.

    Args are (buurt_ids, weight, wkb_bytes) — geometry is WKB-serialised so it
    survives the inter-process pickle round-trip efficiently.
    """
    buurt_ids, weight, geom_wkb = args
    geom = shp_wkb.loads(geom_wkb)

    # Prefer a merged LineString; only fall back to longest segment if linemerge
    # cannot produce a single string (e.g. genuinely disconnected MultiLineString).
    if geom.geom_type == "MultiLineString":
        merged = linemerge(geom)
        geom   = merged if merged.geom_type == "LineString" \
                        else max(geom.geoms, key=lambda g: g.length)

    try:
        projections = sorted(
            (geom.project(_worker_centroid_map[bid]), bid)
            for bid in buurt_ids
        )
    except Exception:
        return []

    ordered = [p[1] for p in projections]
    return [
        (min(ordered[i], ordered[i + 1]),
         max(ordered[i], ordered[i + 1]),
         weight)
        for i in range(len(ordered) - 1)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    total_start = time.time()

    # ====================== BUURTEN ======================
    print("=" * 60)
    print("STEP 1/7 — Loading CBS buurten")
    print("=" * 60)
    t0 = time.time()
    buurten = gpd.read_file(BUURTEN_GPKG)
    print(f"  Raw rows loaded : {len(buurten):,}")
    buurten = buurten[buurten["aantal_inwoners"] >= 0].copy()
    print(f"  After water filter : {len(buurten):,}")
    buurten = buurten.to_crs("EPSG:28992").reset_index(drop=True)
    buurten["buurt_id"] = buurten.index
    print(f"  CRS → EPSG:28992  ({time.time()-t0:.1f}s)")

    # ====================== OSM LOADING ======================
    print()
    print("=" * 60)
    print("STEP 2/7 — Loading OSM ways from GeoJSON")
    print(f"  {WAYS_GEOJSON}")
    print("=" * 60)
    t0 = time.time()

    print("  Reading file (this may take 1-3 min for a national file)...",
          end=" ", flush=True)
    all_ways_raw = gpd.read_file(WAYS_GEOJSON, columns=["highway", "railway", "route"])
    print(f"✓  {len(all_ways_raw):,} ways  ({time.time()-t0:.0f}s)")

    print(f"  Columns available: {list(all_ways_raw.columns)}")

    # Classify each way into a source bucket used by the weight logic.
    # osmium export preserves all OSM tags as columns where present.
    print("  Classifying sources (road / rail / ferry)...", end=" ", flush=True)
    t0 = time.time()

    def _classify_source(row) -> str:
        railway = row.get("railway")
        if railway is not None and isinstance(railway, str) and railway != "":
            return "rail"
        route = row.get("route")
        if isinstance(route, str) and "ferry" in route.lower():
            return "ferry"
        return "roads"

    all_ways_raw["source"] = all_ways_raw.apply(_classify_source, axis=1)
    print(f"done ({time.time()-t0:.1f}s)")

    src_counts = all_ways_raw["source"].value_counts()
    for src, cnt in src_counts.items():
        print(f"    {src:<8}: {cnt:>10,} ways")

    ways_list = [all_ways_raw]

    # ====================== COMBINE & CLEAN ======================
    print()
    print("=" * 60)
    print("STEP 3/7 — Combining, cleaning, reprojecting ways")
    print("=" * 60)
    t0 = time.time()

    all_ways: gpd.GeoDataFrame = ways_list[0].copy()
    print(f"  Ways loaded : {len(all_ways):,}")

    all_ways = gpd.GeoDataFrame(all_ways, geometry="geometry", crs="EPSG:4326")
    print(f"  Reprojecting to EPSG:28992...", end=" ", flush=True)
    t0 = time.time()
    all_ways = all_ways.to_crs(buurten.crs)
    print(f"done ({time.time()-t0:.0f}s)")

    before = len(all_ways)
    valid_mask = all_ways.geometry.notna() & all_ways.geometry.is_valid
    all_ways   = all_ways[valid_mask].reset_index(drop=True)
    print(f"  Dropped {before - len(all_ways):,} null/invalid geometries")
    print(f"  Valid ways remaining        : {len(all_ways):,}")

    all_ways["way_idx"] = all_ways.index
    print(f"  Reproject + clean done in {time.time()-t0:.1f}s")

    # ====================== ASSIGN WEIGHTS ======================
    print()
    print("STEP 3b — Assigning weights (vectorised)...", end=" ", flush=True)
    t0 = time.time()

    def _hw_to_weight(h) -> float | None:
        """Map a highway tag value (possibly a list) to a weight, or None."""
        if h is None or (isinstance(h, float) and np.isnan(h)):
            return None
        tag = h[0] if isinstance(h, list) else str(h)
        return WEIGHT_DICT.get(tag)               # None if not found

    hw_weights = all_ways["highway"].map(_hw_to_weight) if "highway" in all_ways.columns \
                 else pd.Series(None, index=all_ways.index, dtype=object)

    source_weight_map = {
        "rail":    WEIGHT_DICT["railway"],
        "ferry":   WEIGHT_DICT["ferry"],
        "walking": WEIGHT_DICT["cycle_foot"],
        "cycling": WEIGHT_DICT["cycle_foot"],
        "roads":   None,    # roads fall back to highway tag
    }
    src_weights = all_ways["source"].map(source_weight_map)

    # Highway tag wins when present; source-derived weight is the fallback;
    # other_road is the final catch-all.
    all_ways["weight"] = (
        hw_weights
        .combine_first(src_weights)
        .fillna(WEIGHT_DICT["other_road"])
        .astype(float)
    )

    weight_summary = all_ways["weight"].value_counts().sort_index()
    print(f"done ({time.time()-t0:.1f}s)")
    print("  Weight distribution:")
    for w, cnt in weight_summary.items():
        print(f"    {w:.1f}  →  {cnt:>10,} ways")

    # ====================== SPATIAL JOIN ======================
    print()
    print("=" * 60)
    print("STEP 4/7 — Spatial join: ways ↔ buurten")
    print("=" * 60)
    t0 = time.time()

    joined = gpd.sjoin(
        all_ways[["way_idx", "weight", "geometry"]],
        buurten[["buurt_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    print(f"  Raw (way, buurt) pairs : {len(joined):,}  ({time.time()-t0:.0f}s)")

    # Keep only ways that touch ≥ 2 buurten — single-buurt ways produce no edges
    t0 = time.time()
    buurt_counts = joined.groupby("way_idx")["buurt_id"].transform("nunique")
    joined       = joined[buurt_counts >= 2].copy()
    n_multi      = joined["way_idx"].nunique()
    print(f"  Ways crossing ≥ 2 buurten : {n_multi:,}  (filter took {time.time()-t0:.1f}s)")
    print(f"  Remaining (way, buurt) pairs after filter : {len(joined):,}")

    # ====================== BUILD CENTROID MAP & TASK LIST ======================
    print()
    print("=" * 60)
    print("STEP 5/7 — Building edge task list")
    print("=" * 60)
    t0 = time.time()

    # Fast centroid map — no iterrows
    buurt_centroid_map: dict[int, Point] = {
        int(bid): geom.centroid
        for bid, geom in zip(buurten["buurt_id"], buurten.geometry)
    }
    print(f"  Centroid map built for {len(buurt_centroid_map):,} buurten")

    way_geom_map: dict[int, object] = all_ways.set_index("way_idx")["geometry"].to_dict()
    print(f"  Way geometry map built: {len(way_geom_map):,} entries")

    # Serialise each way's geometry once to WKB (fast inter-process transfer)
    print("  Serialising geometries to WKB and building task list...", end=" ", flush=True)
    tasks: list[tuple[list[int], float, bytes]] = []
    skipped_single = 0

    for way_idx, group in joined.groupby("way_idx"):
        buurt_ids = list(set(group["buurt_id"].tolist()))
        if len(buurt_ids) < 2:
            skipped_single += 1
            continue
        geom = way_geom_map[int(way_idx)]
        tasks.append((
            buurt_ids,
            float(group["weight"].iloc[0]),
            shp_wkb.dumps(geom),
        ))

    print(f"done")
    print(f"  Task list size     : {len(tasks):,} ways")
    print(f"  Skipped (1 buurt)  : {skipped_single:,}")
    print(f"  Task list built in {time.time()-t0:.1f}s")

    # ====================== PARALLEL EDGE BUILDING ======================
    print()
    print("=" * 60)
    print("STEP 6/7 — Building edges (parallel)")
    print("=" * 60)
    n_workers = N_WORKERS or os.cpu_count() or 1
    print(f"  Workers : {n_workers}   Chunk size : {CHUNK_SIZE}")
    print(f"  Tasks   : {len(tasks):,}")
    t0 = time.time()

    edge_weights: dict[tuple[int, int], float] = {}

    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(buurt_centroid_map,),
    ) as pool:
        results = list(tqdm(
            pool.imap(_process_way, tasks, chunksize=CHUNK_SIZE),
            total=len(tasks),
            desc="  Building edges",
            unit="way",
        ))

    # Accumulate
    print("  Accumulating edge weights...", end=" ", flush=True)
    t_acc = time.time()
    for edge_list in results:
        for u, v, w in edge_list:
            key = (u, v)
            edge_weights[key] = edge_weights.get(key, 0.0) + w
    print(f"done ({time.time()-t_acc:.1f}s)")

    print(f"  Unique edges found : {len(edge_weights):,}")
    print(f"  Edge building done in {time.time()-t0:.0f}s")

    if edge_weights:
        weights_arr = np.array(list(edge_weights.values()))
        print(f"  Edge weight stats  — "
              f"min={weights_arr.min():.2f}  "
              f"median={np.median(weights_arr):.2f}  "
              f"mean={weights_arr.mean():.2f}  "
              f"max={weights_arr.max():.2f}")

    # ====================== ASSEMBLE GRAPH ======================
    print()
    print("=" * 60)
    print("STEP 7/7 — Assembling graph & fixing isolated buurten")
    print("=" * 60)
    t0 = time.time()

    G = nx.Graph()
    G.add_nodes_from(buurten.index.tolist())
    G.add_edges_from(
        (u, v, {"weight": w}) for (u, v), w in edge_weights.items()
    )
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Degree distribution summary
    degrees = [G.degree(n) for n in G.nodes]
    print(f"  Degree stats — "
          f"min={min(degrees)}  "
          f"median={int(np.median(degrees))}  "
          f"max={max(degrees)}")

    # Connected components
    components = list(nx.connected_components(G))
    print(f"  Connected components : {len(components)}")
    if len(components) > 1:
        sizes = sorted([len(c) for c in components], reverse=True)
        print(f"    Largest: {sizes[0]}  2nd: {sizes[1] if len(sizes) > 1 else '—'}  "
              f"singletons: {sum(1 for s in sizes if s == 1)}")

    # ── Fix isolated buurten ──────────────────────────────────────
    isolated     = [n for n in G.nodes if G.degree(n) == 0]
    isolated_set = set(isolated)
    print(f"  Isolated (degree-0) buurten : {len(isolated)}")

    if isolated:
        print("  Connecting each isolated buurt to its nearest non-isolated neighbour...")
        centroids_arr = np.array([
            (buurten.geometry.iloc[i].centroid.x,
             buurten.geometry.iloc[i].centroid.y)
            for i in buurten.index
        ])
        kdtree = KDTree(centroids_arr)
        connected = 0
        for node in isolated:
            _, idxs = kdtree.query(centroids_arr[node], k=6)
            for cand in idxs[1:]:
                if int(cand) not in isolated_set:
                    G.add_edge(node, int(cand), weight=0.5)
                    connected += 1
                    break
        print(f"  Connected {connected} isolated buurten via KDTree fallback "
              f"(weight=0.5)")

        # Recount after fix
        still_isolated = [n for n in G.nodes if G.degree(n) == 0]
        print(f"  Still isolated after fix : {len(still_isolated)}")

    print(f"  Graph assembly done in {time.time()-t0:.1f}s")

    # ====================== SAVE GRAPH ======================
    graph_path = OUTPUT_DIR / "buurten_connectedness_graph.pkl"
    print(f"\n  Saving graph → {graph_path} ...", end=" ", flush=True)
    with open(graph_path, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print("done")

    # ====================== GIS VISUALISATION LAYER ======================
    print(f"  Building GIS edge layer...", end=" ", flush=True)
    t0 = time.time()
    centroid_coords = {i: buurten.geometry.iloc[i].centroid for i in buurten.index}
    edges_gdf = gpd.GeoDataFrame(
        [
            {
                "u":        u,
                "v":        v,
                "weight":   d["weight"],
                "geometry": LineString([centroid_coords[u], centroid_coords[v]]),
            }
            for u, v, d in G.edges(data=True)
        ],
        crs=buurten.crs,
    )
    gpkg_path = OUTPUT_DIR / "connectedness_edges.gpkg"
    edges_gdf.to_file(gpkg_path, driver="GPKG")
    print(f"done ({time.time()-t0:.1f}s)  →  {gpkg_path}")

    # ====================== SUMMARY ======================
    elapsed = time.time() - total_start
    m, s    = divmod(int(elapsed), 60)
    print()
    print("=" * 60)
    print(f"✅  DONE  (total wall time: {m}m {s}s)")
    print("=" * 60)
    print(f"   Graph  →  {graph_path}")
    print(f"   QGIS   →  {gpkg_path}")
    print()
    print("   To reload the graph later:")
    print("     import pickle")
    print(f"     with open(r'{graph_path}', 'rb') as fh:")
    print("         G = pickle.load(fh)")


# ─────────────────────────────────────────────────────────────────────────────
# Windows requires the multiprocessing guard around the entry point.
# On Linux/macOS this is a no-op.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.freeze_support()   # needed for frozen executables on Windows; harmless elsewhere
    main()