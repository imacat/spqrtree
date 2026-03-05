# Pure Python SPQR-Tree implementation.
# Authors:
#   imacat@mail.imacat.idv.tw (imacat), 2026/3/1
# AI assistance: Claude Code (Anthropic)

# Copyright (c) 2026 imacat.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Palm tree construction for the SPQR-Tree algorithm.

Implements a DFS-based palm tree computation as required by the
Gutwenger-Mutzel (2001) triconnected components algorithm.

A palm tree is a DFS spanning tree together with all back edges (fronds).
For each vertex v, we compute:

- ``dfs_num[v]``: DFS discovery order number (1-indexed).
- ``parent[v]``: parent vertex in the DFS tree (None for root).
- ``parent_edge[v]``: edge ID of the parent edge (None for root).
- ``tree_edges``: set of tree edge IDs.
- ``fronds``: set of frond (back edge) IDs.
- ``lowpt1[v]``: minimum DFS number reachable from the subtree of v
  (including v itself) via at most one frond.
- ``lowpt2[v]``: second minimum DFS number in the same candidate set,
  i.e., the smallest value strictly greater than lowpt1[v].
  Set to ``LOWPT_INF`` if no such value exists.
- ``nd[v]``: number of vertices in the subtree rooted at v (including v).
- ``first_child[v]``: first DFS tree child of v (None if v is a leaf).
- ``high[v]``: highest DFS number of a descendant of v (inclusive) that
  has a frond to a proper ancestor of v.  Zero if no such frond exists.
"""
from collections.abc import Hashable
from dataclasses import dataclass, field

from spqrtree._graph import Edge, MultiGraph

LOWPT_INF: int = 10 ** 9
"""Sentinel value representing positive infinity for lowpt computations."""


@dataclass
class PalmTree:
    """A DFS palm tree with low-point values for a connected multigraph.

    All vertex-keyed dictionaries are defined for every vertex in the
    graph.  Edge-ID sets cover every edge exactly once.
    """

    dfs_num: dict[Hashable, int]
    """DFS discovery number for each vertex (1-indexed)."""

    parent: dict[Hashable, Hashable | None]
    """Parent vertex in the DFS tree; None for the root."""

    parent_edge: dict[Hashable, int | None]
    """Edge ID of the tree edge to the parent; None for the root."""

    tree_edges: set[int]
    """Set of edge IDs classified as DFS tree edges."""

    fronds: set[int]
    """Set of edge IDs classified as fronds (back edges)."""

    lowpt1: dict[Hashable, int]
    """Minimum DFS number reachable from the subtree of each vertex."""

    lowpt2: dict[Hashable, int]
    """Second minimum DFS number reachable; LOWPT_INF if none."""

    nd: dict[Hashable, int]
    """Number of vertices in the subtree rooted at each vertex."""

    first_child: dict[Hashable, Hashable | None]
    """First DFS tree child of each vertex; None if the vertex is a leaf."""

    high: dict[Hashable, int]
    """Highest descendant DFS number with a frond to a proper ancestor."""

    children: dict[Hashable, list[Hashable]] = field(
        default_factory=dict)
    """Ordered list of DFS tree children for each vertex."""

    dfs_order: list[Hashable] = field(default_factory=list)
    """Vertices in DFS discovery order (root first)."""


def _is_frond_edge(
    w: Hashable,
    v: Hashable,
    eid: int,
    dfs_num: dict[Hashable, int],
    parent_edge: dict[Hashable, int | None],
) -> bool:
    """Check if edge *eid* from *v* to *w* is a frond (back edge).

    A frond goes from v to a proper ancestor w (lower DFS number)
    and is not the tree edge connecting v to its parent.

    :param w: The other endpoint of the edge.
    :param v: The current vertex.
    :param eid: The edge ID.
    :param dfs_num: DFS discovery numbers.
    :param parent_edge: Parent edge IDs.
    :return: True if the edge is a frond.
    """
    return dfs_num[w] < dfs_num[v] and eid != parent_edge[v]


def build_palm_tree(graph: MultiGraph, start: Hashable) -> PalmTree:
    """Build a DFS palm tree for the given graph starting at *start*.

    Performs an iterative DFS in adjacency-list order (edge insertion
    order).  Computes DFS numbers, parent links, tree/frond
    classification, lowpt1/lowpt2, nd, first_child, and high values.

    :param graph: The multigraph to traverse.
    :param start: The starting vertex for the DFS.
    :return: A fully populated PalmTree.
    :raises KeyError: If *start* is not in the graph.
    """
    dfs_num: dict[Hashable, int] = {}
    parent: dict[Hashable, Hashable | None] = {start: None}
    parent_edge: dict[Hashable, int | None] = {start: None}
    tree_edges: set[int] = set()
    fronds: set[int] = set()
    lowpt1: dict[Hashable, int] = {}
    lowpt2: dict[Hashable, int] = {}
    nd: dict[Hashable, int] = {}
    first_child: dict[Hashable, Hashable | None] = {}
    high: dict[Hashable, int] = {}
    children: dict[Hashable, list[Hashable]] = {
        v: [] for v in graph.vertices}
    dfs_order: list[Hashable] = []

    counter: int = 0

    # Stack entries: (vertex, iterator-index into adjacency list)
    stack: list[tuple[Hashable, int]] = [(start, 0)]
    visited: set[Hashable] = set()
    adj_lists: dict[Hashable, list[int]] = {
        v: graph.adj_edge_ids(v) for v in graph.vertices
    }

    while stack:
        v, idx = stack[-1]

        if v not in visited:
            visited.add(v)
            counter += 1
            dfs_num[v] = counter
            dfs_order.append(v)
            first_child[v] = None

        adj: list[int] = adj_lists[v]
        advanced: bool = False

        while idx < len(adj):
            eid: int = adj[idx]
            idx += 1
            e: Edge | None = graph.get_edge(eid)
            assert e is not None
            w: Hashable = e.other(v)

            if w not in visited:
                # Tree edge v -> w
                tree_edges.add(eid)
                parent[w] = v
                parent_edge[w] = eid
                children[v].append(w)
                if first_child[v] is None:
                    first_child[v] = w
                stack[-1] = (v, idx)
                stack.append((w, 0))
                advanced = True
                break
            elif _is_frond_edge(w, v, eid, dfs_num, parent_edge):
                # Frond v -> w (w is a proper ancestor of v)
                fronds.add(eid)
            # else: w is a descendant or parent edge (already handled)

        if not advanced:
            # All adjacencies of v processed; compute bottom-up values
            stack.pop()
            _compute_lowpt(
                v, dfs_num, children, fronds,
                graph, lowpt1, lowpt2, nd, high
            )

    return PalmTree(
        dfs_num=dfs_num,
        parent=parent,
        parent_edge=parent_edge,
        tree_edges=tree_edges,
        fronds=fronds,
        lowpt1=lowpt1,
        lowpt2=lowpt2,
        nd=nd,
        first_child=first_child,
        high=high,
        children=children,
        dfs_order=dfs_order,
    )


def _compute_lowpt(
    v: Hashable,
    dfs_num: dict[Hashable, int],
    children: dict[Hashable, list[Hashable]],
    fronds: set[int],
    graph: MultiGraph,
    lowpt1: dict[Hashable, int],
    lowpt2: dict[Hashable, int],
    nd: dict[Hashable, int],
    high: dict[Hashable, int],
) -> None:
    """Compute lowpt1, lowpt2, nd, and high for vertex v.

    Called in post-order (after all children of v are processed).
    Updates the dictionaries in place.

    :param v: The vertex being processed.
    :param dfs_num: DFS discovery numbers for all vertices.
    :param children: DFS tree children for all vertices.
    :param fronds: Set of frond edge IDs.
    :param graph: The multigraph.
    :param lowpt1: Output dictionary for lowpt1 values.
    :param lowpt2: Output dictionary for lowpt2 values.
    :param nd: Output dictionary for subtree sizes.
    :param high: Output dictionary for high values.
    :return: None
    """
    # Candidate multiset for low-point computation.
    # We only need to track the two smallest distinct values.
    lp1: int = dfs_num[v]
    lp2: int = LOWPT_INF

    def _update(val: int) -> None:
        """Update (lp1, lp2) with a new candidate value.

        :param val: A new candidate DFS number.
        :return: None
        """
        nonlocal lp1, lp2
        if val < lp1:
            lp2 = lp1
            lp1 = val
        elif lp1 < val < lp2:
            lp2 = val

    # Fronds from v directly
    for eid in graph.adj_edge_ids(v):
        if eid in fronds:
            e: Edge | None = graph.get_edge(eid)
            assert e is not None
            w: Hashable = e.other(v)
            _update(dfs_num[w])

    # Propagate from children
    nd_sum: int = 1
    high_max: int = 0
    for c in children[v]:
        _update(lowpt1[c])
        if lowpt2[c] != LOWPT_INF:
            _update(lowpt2[c])
        nd_sum += nd[c]
        if high[c] > high_max:
            high_max = high[c]

    lowpt1[v] = lp1
    lowpt2[v] = lp2
    nd[v] = nd_sum

    # high[v]: the highest DFS number of a descendant (or v) that has
    # a frond to a proper ancestor of v.
    # A frond (v, w) goes to a proper ancestor if dfs_num[w] < dfs_num[v].
    if any(eid in fronds for eid in graph.adj_edge_ids(v)):
        high_max = max(high_max, dfs_num[v])
    high[v] = high_max


def phi_key(
    v: Hashable,
    eid: int,
    pt: PalmTree,
    graph: MultiGraph,
) -> int:
    """Compute the sort key φ(e) for edge eid as defined by Gutwenger-Mutzel.

    The φ ordering governs the adjacency list order used during the main
    DFS in Algorithm 3 (PathSearch).  Smaller φ values come first.

    From Gutwenger-Mutzel (2001), p. 83:

    For a tree edge e = v→w (w is a child of v):

    - φ(e) = 3 * lowpt1[w]       if lowpt2[w] < dfs_num[v]
    - φ(e) = 3 * lowpt1[w] + 2   if lowpt2[w] >= dfs_num[v]

    For a frond e = v↪w (w is a proper ancestor of v):

    - φ(e) = 3 * dfs_num[w] + 1

    :param v: The vertex whose adjacency list is being sorted.
    :param eid: The edge ID to compute the key for.
    :param pt: The palm tree with DFS data.
    :param graph: The multigraph.
    :return: The integer sort key φ(e).
    """
    e: Edge | None = graph.get_edge(eid)
    assert e is not None
    w: Hashable = e.other(v)
    w_num: int = pt.dfs_num.get(w, 0)
    v_num: int = pt.dfs_num.get(v, 0)

    if eid in pt.tree_edges and pt.parent_edge.get(w) == eid:
        # Tree edge v→w where w is the child of v.
        if pt.lowpt2[w] < v_num:
            # Case 1: lowpt2(w) < dfs_num(v).
            return 3 * pt.lowpt1[w]
        # Case 3: lowpt2(w) >= dfs_num(v).
        return 3 * pt.lowpt1[w] + 2

    if eid in pt.fronds and w_num < v_num:
        # Frond from v to proper ancestor w (w has smaller DFS number).
        # Case 2: phi = 3 * dfs_num(w) + 1.
        return 3 * w_num + 1

    # Parent edge or edge to descendant (frond from the other direction):
    # assign a large key so it is sorted last.
    return 3 * (w_num + 1) * 3 + 3


def sort_adjacency_lists(
    graph: MultiGraph,
    pt: PalmTree,
) -> None:
    """Re-order each vertex's adjacency list by φ values in place.

    This must be called before running PathSearch (Algorithm 3) to
    ensure the DFS visits edges in the order required for the
    Gutwenger-Mutzel triconnected components algorithm.

    :param graph: The multigraph whose adjacency lists will be sorted.
    :param pt: A fully computed palm tree for the graph.
    :return: None
    """
    for v in graph.vertices:
        adj: list[int] = graph.adj_edge_ids(v)
        adj_sorted: list[int] = sorted(
            adj,
            key=lambda eid, _v=v: phi_key(_v, eid, pt, graph)
        )
        graph.set_adj_order(v, adj_sorted)
