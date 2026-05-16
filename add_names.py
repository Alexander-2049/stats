# add_buurt_names_to_graph.py
# ────────────────────────────────────────────────────────────
# Adds buurt names + metadata from the CBS GeoPackage
# into the existing NetworkX graph as node attributes.
#
# Result:
#   G.nodes[node_id]["buurtnaam"]
#   G.nodes[node_id]["gemeentenaam"]
#   G.nodes[node_id]["wijknaam"]
#   G.nodes[node_id]["bu_code"]
#
# IMPORTANT:
# This assumes graph node IDs correspond to:
#     buurten.reset_index(drop=True)
#
# exactly like in your original graph-building script.
# ────────────────────────────────────────────────────────────

import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx

# ==========================================================
# CONFIG
# ==========================================================
GRAPH_PATH = Path("osm_connectedness/buurten_connectedness_graph.pkl")

OUTPUT_PATH = Path(
    "osm_connectedness/buurten_connectedness_graph_named.pkl"
)

BUURTEN_GPKG = r"C:\Users\ivanm\OneDrive\Документы\buurten.gpkg"
# ==========================================================


def main() -> None:
    print("=" * 60)
    print("LOADING GRAPH")
    print("=" * 60)

    with open(GRAPH_PATH, "rb") as fh:
        G: nx.Graph = pickle.load(fh)

    print(
        f"Graph loaded: "
        f"{G.number_of_nodes():,} nodes / "
        f"{G.number_of_edges():,} edges"
    )

    print("\nLoading buurten GeoPackage...")

    buurten = gpd.read_file(BUURTEN_GPKG)

    # MUST match original graph preprocessing exactly
    buurten = buurten[buurten["aantal_inwoners"] >= 0].copy()
    buurten = buurten.to_crs("EPSG:28992").reset_index(drop=True)

    print(f"Loaded {len(buurten):,} buurten")

    print("\nAdding metadata to graph nodes...")

    added = 0

    for idx, row in buurten.iterrows():

        if idx not in G:
            continue

        G.nodes[idx]["buurtnaam"] = row.get("buurtnaam")
        G.nodes[idx]["wijknaam"] = row.get("wijknaam")
        G.nodes[idx]["gemeentenaam"] = row.get("gemeentenaam")
        G.nodes[idx]["bu_code"] = row.get("bu_code")

        added += 1

    print(f"Metadata added to {added:,} graph nodes")

    print(f"\nSaving enriched graph → {OUTPUT_PATH}")

    with open(OUTPUT_PATH, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")


if __name__ == "__main__":
    main()