"""
repair_disconnected_islands.py
──────────────────────────────
Interactive helper script for repairing disconnected components in the
buurten connectedness graph.

Workflow:
1. Load your existing NetworkX graph
2. Find all connected components
3. Keep the largest component as the mainland
4. For every disconnected island/component:
      • show component stats
      • print a few node IDs from that component
      • ask the user for TWO buurt IDs:
            - one from the island
            - one from the mainland
      • create a ferry edge between them
5. Repeat until graph becomes fully connected
6. Save repaired graph

Typical use:
    island node id   = a buurt on Texel / Vlieland / etc.
    mainland node id = closest mainland coastal buurt

Suggested ferry weight:
    6.5  (same as your WEIGHT_DICT["ferry"])

You can also type:
    skip    → ignore this component for now
    quit    → stop immediately and save current progress
"""

import pickle
from pathlib import Path

import networkx as nx


# ==========================================================
# CONFIG
# ==========================================================
GRAPH_PATH = Path("osm_connectedness/buurten_connectedness_graph_named.pkl")
# Output graph after repairs
OUTPUT_PATH = Path("osm_connectedness/buurten_connectedness_graph_repaired.pkl")

# Default ferry edge weight
FERRY_WEIGHT = 6.5
# ==========================================================


def print_component_summary(G: nx.Graph) -> None:
    comps = sorted(nx.connected_components(G), key=len, reverse=True)

    print("\n" + "=" * 60)
    print("CONNECTED COMPONENT SUMMARY")
    print("=" * 60)

    for i, comp in enumerate(comps[:15]):
        print(
            f"{i:>2} | size={len(comp):>5} "
            f"| sample nodes={list(sorted(comp))[:10]}"
        )

    if len(comps) > 15:
        print(f"... and {len(comps)-15} more components")

    print("=" * 60)
    print(f"Total components: {len(comps)}")
    print(f"Largest component size: {len(comps[0])}")
    print()

def search_nodes(G, query, limit=20):
    """
    Search graph nodes by buurtnaam or gemeentenaam.
    """

    query = query.lower()
    matches = []

    for node, data in G.nodes(data=True):

        buurtnaam = str(data.get("buurtnaam", ""))
        gemeente = str(data.get("gemeentenaam", ""))

        text = f"{buurtnaam} {gemeente}".lower()

        if query in text:
            matches.append(
                (
                    node,
                    buurtnaam,
                    gemeente,
                )
            )

    matches = sorted(matches, key=lambda x: (x[2], x[1]))

    print("\nSearch results:")
    print("-" * 60)

    for node, buurtnaam, gemeente in matches[:limit]:
        print(f"{node:>5} | {buurtnaam} ({gemeente})")

    if not matches:
        print("No matches found.")

    print("-" * 60)

def ask_for_connection(
    G,
    island_component: set[int],
    mainland_component: set[int],
) -> tuple[int, int] | None:
    """
    Ask user which two nodes should be connected.

    Returns:
        (u, v) if valid
        None   if skipped
    """

    def format_nodes(G, nodes, limit=20):
        rows = []

        for n in list(sorted(nodes))[:limit]:
            data = G.nodes[n]

            buurtnaam = data.get("buurtnaam", "UNKNOWN")
            gemeente = data.get("gemeentenaam", "UNKNOWN")

            rows.append(
                f"{n:>5} | {buurtnaam} ({gemeente})"
            )

        return "\n".join(rows)

    print("Island sample buurten:")
    print(format_nodes(G, island_component))

    print()
    print("Mainland sample buurten:")
    print(format_nodes(G, mainland_component))

    while True:
        raw = input(
            "\nEnter TWO buurt IDs "
            "(island mainland)\n"
            "Commands:\n"
            "  search <text>\n"
            "  skip\n"
            "  quit\n"
            "> "
        ).strip()

        if raw.lower().startswith("search "):
            query = raw[7:].strip()

            if query:
                search_nodes(G, query)

            continue

        if raw.lower() == "skip":
            return None

        if raw.lower() == "quit":
            raise KeyboardInterrupt

        parts = raw.split()

        if len(parts) != 2:
            print("Please enter exactly two integers.")
            continue

        try:
            u = int(parts[0])
            v = int(parts[1])
        except ValueError:
            print("IDs must be integers.")
            continue

        if u not in island_component:
            print(f"{u} is NOT inside the disconnected island component.")
            continue

        if v not in mainland_component:
            print(f"{v} is NOT inside the mainland component.")
            continue

        return u, v


def main() -> None:
    print("=" * 60)
    print("LOADING GRAPH")
    print("=" * 60)

    with open(GRAPH_PATH, "rb") as fh:
        G: nx.Graph = pickle.load(fh)

    print(
        f"Loaded graph with "
        f"{G.number_of_nodes():,} nodes and "
        f"{G.number_of_edges():,} edges"
    )

    while True:
        components = sorted(
            nx.connected_components(G),
            key=len,
            reverse=True,
        )

        if len(components) == 1:
            print("\n✅ Graph is now fully connected.")
            break

        print_component_summary(G)

        mainland = components[0]

        # Find next disconnected component
        target_component = components[1]

        try:
            result = ask_for_connection(
                G,
                target_component,
                mainland,
            )

        except KeyboardInterrupt:
            print("\n\nStopping early and saving current progress...")
            break

        if result is None:
            print("Skipped this component.")
            continue

        island_node, mainland_node = result

        G.add_edge(
            island_node,
            mainland_node,
            weight=FERRY_WEIGHT,
            connection_type="manual_ferry",
        )

        print(
            f"\nAdded ferry edge:\n"
            f"  {island_node} ↔ {mainland_node}\n"
            f"  weight = {FERRY_WEIGHT}"
        )

    print("\n" + "=" * 60)
    print("FINAL GRAPH STATUS")
    print("=" * 60)

    final_components = list(nx.connected_components(G))

    print(f"Connected components : {len(final_components)}")
    print(f"Nodes                : {G.number_of_nodes():,}")
    print(f"Edges                : {G.number_of_edges():,}")

    print(f"\nSaving repaired graph → {OUTPUT_PATH}")

    with open(OUTPUT_PATH, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")


if __name__ == "__main__":
    main()