# Pure Python SPQR-Tree implementation.
# Authors:
#   imacat@mail.imacat.idv.tw (imacat), 2026/3/2
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
"""Triconnected components for the SPQR-Tree algorithm.

Implements decomposition of a biconnected multigraph into its
triconnected split components: BONDs (parallel-edge groups),
POLYGONs (simple cycles), and TRICONNECTED subgraphs.

The algorithm proceeds in three phases:

1. **Multi-edge phase**: Each group of parallel edges becomes a BOND
   split component; they are replaced by a single virtual edge.
2. **PathSearch phase**: An iterative DFS using the Gutwenger-Mutzel
   (2001) algorithm detects separation pairs and creates split
   components using an edge stack (ESTACK) and triple stack (TSTACK).
3. **Merge phase**: Adjacent split components of the same type that
   share a virtual edge are merged (Algorithm 2).

Public API::

    from spqrtree._triconnected import (
        ComponentType, TriconnectedComponent,
        find_triconnected_components,
    )
"""
import bisect
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum

from spqrtree._graph import Edge, MultiGraph
from spqrtree._palm_tree import (
    LOWPT_INF,
    PalmTree,
    build_palm_tree,
    sort_adjacency_lists,
)

# Sentinel triple marking the end of a path segment on the TSTACK.
_EOS: tuple[int, int, int] = (-1, -1, -1)


class ComponentType(Enum):
    """Classification of a triconnected split component.

    :cvar BOND: A bond - two vertices connected by multiple parallel
        edges.
    :cvar POLYGON: A simple cycle - every vertex has degree 2 in the
        component.
    :cvar TRICONNECTED: A 3-connected subgraph.
    """

    BOND = "bond"
    """A bond component: two vertices connected by parallel edges."""

    POLYGON = "polygon"
    """A polygon component: a simple cycle."""

    TRICONNECTED = "triconnected"
    """A triconnected component: a 3-connected subgraph."""


@dataclass
class TriconnectedComponent:
    """A single triconnected split component.

    :param type: Classification as BOND, POLYGON, or TRICONNECTED.
    :param edges: All edges in this component (real and virtual).
        Virtual edges are shared with exactly one other component.
    """

    type: ComponentType
    """Classification of this component."""

    edges: list[Edge]
    """All edges in this component (real and virtual)."""


def _check_biconnected(graph: MultiGraph) -> None:
    """Check that a graph is biconnected (connected with no cut vertex).

    Uses a single DFS pass to verify connectivity and detect cut
    vertices via Tarjan's algorithm.  A non-root vertex v is a cut
    vertex if it has a child w such that lowpt1[w] >= dfs_num[v].
    The DFS root is a cut vertex if it has two or more children.

    :param graph: The multigraph to check.
    :raises ValueError: If the graph is not connected or has a cut
        vertex.
    """
    start: Hashable = next(iter(graph.vertices))
    pt: PalmTree = build_palm_tree(graph, start)

    # Check connectivity: DFS must visit all vertices.
    if len(pt.dfs_num) < graph.num_vertices():
        raise ValueError("graph is not connected")

    # Check for cut vertices.
    for v in graph.vertices:
        if pt.parent[v] is None:
            # Root: cut vertex if it has 2+ children.
            if len(pt.children[v]) >= 2:
                raise ValueError("graph has a cut vertex")
        else:
            # Non-root: cut vertex if any child w has
            # lowpt1[w] >= dfs_num[v].
            v_num: int = pt.dfs_num[v]
            for w in pt.children[v]:
                if pt.lowpt1[w] >= v_num:
                    raise ValueError(
                        "graph has a cut vertex")


def find_triconnected_components(
    graph: MultiGraph,
) -> list[TriconnectedComponent]:
    """Find all triconnected split components of a biconnected multigraph.

    The input must be a biconnected multigraph (possibly with parallel
    edges).  Returns a list of TriconnectedComponent objects, each
    classified as BOND, POLYGON, or TRICONNECTED.

    Each real (non-virtual) edge of the input appears in exactly one
    component.  Each virtual edge (added by the algorithm) appears in
    exactly two components, representing the two sides of a separation
    pair.

    :param graph: A biconnected multigraph.
    :return: A list of TriconnectedComponent objects.
    :raises ValueError: If the graph is not connected or has a
        cut vertex.
    """
    if graph.num_vertices() == 0 or graph.num_edges() == 0:
        return []

    _check_biconnected(graph)

    # Work on a copy to avoid modifying the caller's graph.
    g: MultiGraph = graph.copy()

    # Accumulated split components (each is a list of edge IDs).
    raw_comps: list[list[int]] = []

    # Set of virtual edge IDs added during decomposition.
    virtual_ids: set[int] = set()

    # Phase 1: Split off multi-edges.
    _phase_multiedge(g, raw_comps, virtual_ids)

    # If only 2 vertices remain (or fewer), skip PathSearch.
    if g.num_vertices() >= 2 and g.num_edges() >= 1:
        # Phase 2: PathSearch.
        _phase_pathsearch(g, raw_comps, virtual_ids)

    # Phase 3: Classify and merge.
    return _phase_classify_merge(raw_comps, virtual_ids, graph, g)


# ---------------------------------------------------------------------------
# Phase 1: Multi-edge splitting (Algorithm 1)
# ---------------------------------------------------------------------------

def _phase_multiedge(
    g: MultiGraph,
    raw_comps: list[list[int]],
    virtual_ids: set[int],
) -> None:
    """Replace groups of parallel edges with virtual edges (Algorithm 1).

    For each pair of vertices (u, v) with k >= 2 parallel edges,
    creates a BOND split component {e_1, ..., e_k, e'} where e' is a
    new virtual edge, then removes e_1, ..., e_k from g, leaving e'.

    :param g: The working multigraph (modified in place).
    :param raw_comps: List to append new split components to.
    :param virtual_ids: Set to add new virtual edge IDs to.
    :return: None
    """
    # Collect all unordered vertex pairs.
    seen_pairs: set[frozenset[Hashable]] = set()
    pairs: list[tuple[Hashable, Hashable]] = []
    for e in g.edges:
        key: frozenset[Hashable] = frozenset((e.u, e.v))
        if key not in seen_pairs:
            seen_pairs.add(key)
            pairs.append((e.u, e.v))

    for u, v in pairs:
        parallel: list[Edge] = g.edges_between(u, v)
        if len(parallel) < 2:
            continue
        # Only create a virtual edge if there are other edges in g
        # (i.e., the bond is embedded in a larger graph).
        if g.num_edges() > len(parallel):
            ve: Edge = g.add_edge(u, v, virtual=True)
            virtual_ids.add(ve.id)
            comp_eids: list[int] = [e.id for e in parallel] + [ve.id]
        else:
            # Entire graph is this bond: no virtual edge needed.
            comp_eids = [e.id for e in parallel]
        raw_comps.append(comp_eids)
        # Remove original parallel edges.
        for e in parallel:
            g.remove_edge(e.id)


def _classify_start_edge(
    eid: int,
    w: Hashable,
    v: Hashable,
    v_num: int,
    w_num: int,
    tree_edges: set[int],
    fronds: set[int],
    parent_edge: dict[Hashable, int | None],
) -> tuple[bool, bool]:
    """Classify an edge for the start-set computation.

    :param eid: Edge ID to classify.
    :param w: The other endpoint.
    :param v: The current vertex.
    :param v_num: DFS number of v.
    :param w_num: DFS number of w.
    :param tree_edges: Set of tree edge IDs.
    :param fronds: Set of frond edge IDs.
    :param parent_edge: Parent edge map.
    :return: (is_tree_arc, is_frond) tuple.
    """
    is_tree_arc: bool = (eid in tree_edges
                         and parent_edge.get(w) == eid)
    is_frond: bool = (eid in fronds
                      and w_num < v_num
                      and eid != parent_edge.get(v))
    return is_tree_arc, is_frond


def _compute_start_set(g: MultiGraph, pt: PalmTree) -> set[int]:
    """Compute start edges using the path-finder traversal.

    An edge starts a new path if ``new_path`` is True when the edge is
    first traversed in DFS order.  ``new_path`` starts True, is set to
    False after the first outgoing arc (tree edge or frond) is marked,
    and is set back to True after any frond is processed.  Backtracking
    never changes ``new_path``.

    This corresponds to the ``starts_path`` computation in the
    Gutwenger-Mutzel (2001) path_finder subroutine and determines which
    tree-edge arcs push a new EOS sentinel onto TSTACK during PathSearch.

    :param g: The multigraph with phi-sorted adjacency lists.
    :param pt: The palm tree computed for g.
    :return: Set of edge IDs that start a new path segment.
    """
    start_set: set[int] = set()
    if not g.vertices:
        return start_set

    dfs_num: dict[Hashable, int] = pt.dfs_num
    tree_edges: set[int] = pt.tree_edges
    fronds: set[int] = pt.fronds
    parent_edge: dict[Hashable, int | None] = pt.parent_edge
    root: Hashable = min(
        g.vertices, key=lambda v: dfs_num[v])

    adj_lists: dict[Hashable, list[int]] = {
        v: g.adj_edge_ids(v) for v in g.vertices}

    new_path: bool = True
    # Stack entries: (vertex, adj_index)
    stack: list[tuple[Hashable, int]] = [(root, 0)]

    while stack:
        v, idx = stack[-1]
        adj: list[int] = adj_lists[v]

        if idx >= len(adj):
            stack.pop()
            continue

        eid: int = adj[idx]
        stack[-1] = (v, idx + 1)

        if not g.has_edge(eid):
            continue

        e: Edge | None = g.get_edge(eid)
        assert e is not None
        w: Hashable = e.other(v)
        v_num: int = dfs_num[v]
        w_num: int = dfs_num.get(w, 0)
        is_tree_arc, is_frond = _classify_start_edge(
            eid, w, v, v_num, w_num,
            tree_edges, fronds, parent_edge)

        if is_tree_arc:
            # Tree edge v -> w (w is a child of v).
            if new_path:
                start_set.add(eid)
                new_path = False
            stack.append((w, 0))
        elif is_frond:
            # Frond v -> w (w is a proper ancestor of v).
            if new_path:
                start_set.add(eid)
            new_path = True
        # Parent edge or reverse-direction edge: skip.

    return start_set


def _renumber_palm_tree(pt: PalmTree) -> None:
    """Renumber a palm tree using the Gutwenger-Mutzel scheme.

    Assigns ``newnum[v] = counter - nd[v] + 1`` where *counter*
    starts at *n* and decrements on backtrack.  Within a vertex's
    children, the first child (in phi-sorted order) receives the
    **highest** DFS numbers and the last child the **lowest**.  This
    matches the numbering in Gutwenger-Mutzel (2001) §3.3 and is
    required for correct TSTACK triple ranges in PathSearch.

    Updates ``dfs_num``, ``lowpt1``, and ``lowpt2`` in place.

    :param pt: The palm tree to renumber.
    :return: None
    """
    n: int = len(pt.dfs_num)
    old_to_new: dict[int, int] = {}
    counter: int = n

    # Iterative DFS following children order.
    root: Hashable = next(
        v for v, p in pt.parent.items() if p is None)
    stack: list[tuple[Hashable, bool]] = [(root, False)]
    visited: set[Hashable] = set()

    while stack:
        v, returning = stack.pop()
        if returning:
            counter -= 1
            continue
        if v in visited:
            continue
        visited.add(v)
        old_num: int = pt.dfs_num[v]
        new_num: int = counter - pt.nd[v] + 1
        old_to_new[old_num] = new_num

        # Push backtrack marker.
        stack.append((v, True))
        # Push children in reverse order so first child is
        # processed first (popped last from reversed push).
        for child in reversed(pt.children.get(v, [])):
            stack.append((child, False))

    # Update dfs_num.
    for v in pt.dfs_num:
        pt.dfs_num[v] = old_to_new[pt.dfs_num[v]]
    # Update lowpt1 and lowpt2.
    for v in pt.lowpt1:
        pt.lowpt1[v] = old_to_new[pt.lowpt1[v]]
    for v in pt.lowpt2:
        old_val: int = pt.lowpt2[v]
        if old_val == LOWPT_INF:
            continue
        pt.lowpt2[v] = old_to_new.get(old_val, old_val)


# ---------------------------------------------------------------------------
# Phase 2: PathSearch (Algorithms 3 - 6)
# ---------------------------------------------------------------------------


class _PathSearcher:
    """PathSearch algorithm state and methods (Algorithms 3-6).

    Encapsulates the mutable state and sub-algorithms for the
    Gutwenger-Mutzel (2001) PathSearch.  Splitting the logic into
    methods keeps cognitive complexity manageable.
    """

    def __init__(
        self,
        g: MultiGraph,
        raw_comps: list[list[int]],
        virtual_ids: set[int],
    ) -> None:
        """Initialize the PathSearch state.

        Builds the palm tree, sorts adjacency lists, and sets up
        all mutable data structures needed by the algorithm.

        :param g: The working multigraph.
        :param raw_comps: Accumulator for split components.
        :param virtual_ids: Accumulator for virtual edge IDs.
        :return: None
        """
        self.g: MultiGraph = g
        """The working multigraph."""
        self.raw_comps: list[list[int]] = raw_comps
        """Accumulated raw split components."""
        self.virtual_ids: set[int] = virtual_ids
        """Set of virtual edge IDs."""

        # Build palm tree and sort adjacency lists.
        # Use the first remaining edge's endpoint as start vertex
        # for deterministic DFS ordering.  Edge IDs are sequential
        # integers, so g.edges[0] is always the lowest-ID edge,
        # regardless of Python hash seed.
        start: Hashable = g.edges[0].u
        pt: PalmTree = build_palm_tree(g, start)
        sort_adjacency_lists(g, pt)
        pt = build_palm_tree(g, start)
        _renumber_palm_tree(pt)

        self.dfs_num: dict[Hashable, int] = pt.dfs_num
        """DFS discovery numbers."""
        self.parent: dict[Hashable, Hashable | None] = pt.parent
        """DFS tree parent for each vertex."""
        self.lowpt1: dict[Hashable, int] = pt.lowpt1
        """Lowpt1 values."""
        self.lowpt2: dict[Hashable, int] = pt.lowpt2
        """Lowpt2 values."""
        self.nd: dict[Hashable, int] = pt.nd
        """Subtree sizes."""
        self.inv_dfs: dict[int, Hashable] = {
            n: v for v, n in pt.dfs_num.items()}
        """Inverse DFS map: DFS number to vertex."""
        self.cur_parent_edge: dict[Hashable, int | None] = dict(
            pt.parent_edge)
        """Current parent edge for each vertex."""
        self.cur_tree: set[int] = set(pt.tree_edges)
        """Current set of tree edge IDs."""
        self.cur_deg: dict[Hashable, int] = {
            v: g.degree(v) for v in g.vertices}
        """Current degree of each vertex."""
        self.cur_children: dict[Hashable, list[Hashable]] = {
            v: list(pt.children[v])
            for v in g.vertices}
        """Current DFS children for each vertex."""
        self.fronds: set[int] = pt.fronds
        """Set of frond edge IDs from palm tree."""

        # Build frond sources for _high().
        self.frond_srcs: dict[int, list[int]] = {
            pt.dfs_num[v]: [] for v in g.vertices}
        """Frond source DFS numbers by target."""
        self.in_high: dict[int, tuple[int, int]] = {}
        """Maps edge ID to (target_dfs, source_dfs) for
        highpt tracking."""
        for eid in pt.fronds:
            e: Edge | None = g.get_edge(eid)
            if e is None:
                continue
            if pt.dfs_num[e.u] > pt.dfs_num[e.v]:
                s: int = pt.dfs_num[e.u]
                d: int = pt.dfs_num[e.v]
            else:
                s = pt.dfs_num[e.v]
                d = pt.dfs_num[e.u]
            self.frond_srcs[d].append(s)
            self.in_high[eid] = (d, s)
        for lst in self.frond_srcs.values():
            lst.sort()

        self.start_set: set[int] = _compute_start_set(g, pt)
        """Edge IDs that start a new path segment."""
        self.estack: list[int] = []
        """Edge stack (ESTACK)."""
        self.tstack: list[tuple[int, int, int]] = []
        """Triple stack (TSTACK)."""
        self.adj_cache: dict[Hashable, list[int]] = {
            v: list(g.adj_edge_ids(v))
            for v in g.vertices}
        """Cached adjacency lists."""
        self.consumed: set[int] = set()
        """Edge IDs consumed by split components."""
        self.adj_len: dict[Hashable, int] = {}
        """Original adjacency list length per vertex for DFS
        iteration bounds."""
        self.y_accum: dict[Hashable, int] = {}
        """Accumulated TSTACK h-value per vertex, persisting
        across children (matching SageMath's y_dict)."""
        self.call_stack: list[tuple] = []
        """DFS call stack."""

    def run(self) -> None:
        """Execute the main PathSearch DFS loop.

        Processes visit and post-return frames iteratively.
        Any remaining edges on ESTACK form a final component.

        :return: None
        """
        root: Hashable = self.inv_dfs[1]
        self.adj_len[root] = len(
            self.adj_cache.get(root, []))
        self.y_accum[root] = 0
        self.call_stack = [('visit', root, 0)]

        while self.call_stack:
            frame: tuple = self.call_stack[-1]

            if frame[0] == 'post':
                self.call_stack.pop()
                self._process_post_frame(
                    frame[1], frame[2], frame[3], frame[4])
                continue

            _, v, idx = frame
            adj: list[int] = self.adj_cache.get(v, [])
            bound: int = self.adj_len.get(v, len(adj))

            if idx >= bound:
                self.call_stack.pop()
                continue

            eid: int = adj[idx]
            self.call_stack[-1] = ('visit', v, idx + 1)

            if not self.g.has_edge(eid):
                continue

            e: Edge | None = self.g.get_edge(eid)
            assert e is not None
            w: Hashable = e.other(v)
            v_num: int = self.dfs_num[v]
            w_num: int = self.dfs_num.get(w, -1)
            if w_num < 0:
                continue

            is_start: bool = eid in self.start_set

            is_tree_arc: bool = (
                eid in self.cur_tree
                and self.cur_parent_edge.get(w) == eid)
            is_out_frond: bool = (
                eid in self.fronds and w_num < v_num
                and eid != self.cur_parent_edge.get(v))

            if is_tree_arc:
                self._process_tree_edge(
                    v, eid, w, v_num, w_num, is_start)
            elif is_out_frond:
                self._process_frond(
                    v, eid, w, v_num, w_num, is_start)

        if self.estack:
            self.raw_comps.append(list(self.estack))

    def _high(self, w_num: int) -> int:
        """Return the largest frond-source DFS number.

        :param w_num: DFS number of vertex w.
        :return: Largest frond-source DFS number, or 0.
        """
        fl: list[int] = self.frond_srcs.get(w_num, [])
        return fl[-1] if fl else 0

    def _del_high(self, eid: int) -> None:
        """Remove the highpt entry for edge *eid*.

        If *eid* has a recorded ``in_high`` entry, removes
        the corresponding source from ``frond_srcs``.

        :param eid: The edge ID whose highpt entry to remove.
        :return: None
        """
        entry: tuple[int, int] | None = self.in_high.pop(
            eid, None)
        if entry is None:
            return
        target, source = entry
        fl: list[int] = self.frond_srcs.get(target, [])
        idx: int = bisect.bisect_left(fl, source)
        if idx < len(fl) and fl[idx] == source:
            fl.pop(idx)

    def _first_child_num(self, v: Hashable) -> int:
        """Return DFS number of v's first child, or 0.

        :param v: A vertex.
        :return: DFS number of the first child, or 0.
        """
        ch: list[Hashable] = self.cur_children.get(v, [])
        return self.dfs_num[ch[0]] if ch else 0

    def _remaining_deg(self, w: Hashable) -> int:
        """Return the number of non-consumed edges of *w*.

        Counts edges in the adjacency cache that have not
        been consumed by split components and still exist
        in the graph.

        :param w: A vertex.
        :return: Count of remaining incident edges.
        """
        count: int = 0
        for eid in self.adj_cache.get(w, []):
            if eid in self.consumed:
                continue
            if self.g.get_edge(eid) is None:
                continue
            count += 1
        return count

    def _temp_target_num(self, w: Hashable) -> int:
        """Return DFS number of the first remaining outgoing
        edge target from *w*.

        Scans the adjacency cache for the first edge that
        has not been consumed and is not the parent edge,
        returning the target DFS number for outgoing tree
        arcs (to children) or outgoing fronds (to ancestors).

        :param w: A vertex.
        :return: DFS number of the target, or 0.
        """
        w_num: int = self.dfs_num[w]
        peid: int | None = self.cur_parent_edge.get(w)
        for eid in self.adj_cache.get(w, []):
            if eid in self.consumed or eid == peid:
                continue
            ed: Edge | None = self.g.get_edge(eid)
            if ed is None:
                continue
            other: Hashable = ed.other(w)
            o_num: int = self.dfs_num.get(other, -1)
            if o_num < 0:
                continue
            is_tree: bool = eid in self.cur_tree
            if is_tree and o_num > w_num:
                return o_num
            if not is_tree and o_num < w_num:
                return o_num
        return 0

    def _add_virt(self, u: Hashable, v: Hashable) -> Edge:
        """Add a new virtual edge between u and v.

        Records the virtual edge ID.  Does NOT modify
        cur_deg.

        :param u: One endpoint.
        :param v: Other endpoint.
        :return: The new virtual Edge.
        """
        e: Edge = self.g.add_edge(u, v, virtual=True)
        self.virtual_ids.add(e.id)
        return e

    def _delete_tstack_above(
        self, threshold: int,
    ) -> list[tuple[int, int, int]]:
        """Pop TSTACK triples with *a* above *threshold*.

        Pops triples from TSTACK top while they are not EOS
        and their second element (a) exceeds *threshold*.

        :param threshold: The low-point threshold value.
        :return: List of deleted triples.
        """
        deleted: list[tuple[int, int, int]] = []
        while (
            self.tstack
            and self.tstack[-1] != _EOS
            and self.tstack[-1][1] > threshold
        ):
            deleted.append(self.tstack.pop())
        return deleted

    def _process_post_frame(
        self,
        v: Hashable,
        eid: int,
        w_orig: Hashable,
        is_start: bool,
    ) -> None:
        """Handle a post-return frame after recursing.

        Pushes the tree edge onto ESTACK, runs type-2 and
        type-1 checks, cleans up EOS triples, and pops
        stale triples.

        :param v: The parent vertex.
        :param eid: The tree edge ID.
        :param w_orig: The original child vertex.
        :param is_start: Whether this edge starts a path.
        :return: None
        """
        ed: Edge | None = self.g.get_edge(eid)
        if ed is not None:
            w: Hashable = ed.other(v)
        else:
            w = w_orig

        peid_w: int | None = self.cur_parent_edge.get(w)
        if peid_w is not None:
            self.estack.append(peid_w)

        w = self._check_type2(v, w)
        self._check_type1(v, w)

        if is_start:
            while (self.tstack
                   and self.tstack[-1] != _EOS):
                self.tstack.pop()
            if (self.tstack
                    and self.tstack[-1] == _EOS):
                self.tstack.pop()

        v_num: int = self.dfs_num[v]
        while (
            self.tstack
            and self.tstack[-1] != _EOS
            and self.tstack[-1][2] != v_num
            and self._high(v_num)
            > self.tstack[-1][0]
        ):
            self.tstack.pop()

    def _process_tree_edge(
        self,
        v: Hashable,
        eid: int,
        w: Hashable,
        v_num: int,
        w_num: int,
        is_start: bool,
    ) -> None:
        """Handle a tree edge v -> w in the visit frame.

        Updates TSTACK with a new triple if this is a start
        edge, then pushes post-return and visit frames.

        :param v: The current vertex.
        :param eid: The tree edge ID.
        :param w: The child vertex.
        :param v_num: DFS number of v.
        :param w_num: DFS number of w.
        :param is_start: Whether this edge starts a path.
        :return: None
        """
        if is_start:
            lp1w: int = self.lowpt1[w]
            y_acc: int = self.y_accum.get(v, 0)
            deleted: list[tuple[int, int, int]] = \
                self._delete_tstack_above(lp1w)
            if not deleted:
                self.tstack.append(
                    (w_num + self.nd[w] - 1, lp1w, v_num))
            else:
                y: int = y_acc
                for t in deleted:
                    y = max(y, t[0])
                last_b: int = deleted[-1][2]
                self.tstack.append((y, lp1w, last_b))
                self.y_accum[v] = y
            self.tstack.append(_EOS)

        self.call_stack.append(('post', v, eid, w, is_start))
        self.adj_cache[w] = list(self.g.adj_edge_ids(w))
        self.adj_len[w] = len(self.adj_cache[w])
        self.y_accum[w] = 0
        self.call_stack.append(('visit', w, 0))

    def _process_frond(
        self,
        v: Hashable,
        eid: int,
        w: Hashable,
        v_num: int,
        w_num: int,
        is_start: bool,
    ) -> None:
        """Handle a frond v -> w in the visit frame.

        Updates TSTACK if this is a start edge.  If the frond
        goes to the parent, creates a 2-component and replaces
        the parent edge with a virtual edge.

        :param v: The current vertex.
        :param eid: The frond edge ID.
        :param w: The ancestor vertex.
        :param v_num: DFS number of v.
        :param w_num: DFS number of w.
        :param is_start: Whether this edge starts a path.
        :return: None
        """
        if is_start:
            y_acc: int = self.y_accum.get(v, 0)
            deleted: list[tuple[int, int, int]] = \
                self._delete_tstack_above(w_num)
            if not deleted:
                self.tstack.append((v_num, w_num, v_num))
            else:
                y: int = y_acc
                for t in deleted:
                    y = max(y, t[0])
                last_b: int = deleted[-1][2]
                self.tstack.append((y, w_num, last_b))

        p_v: Hashable | None = self.parent.get(v)
        if p_v is not None and w == p_v:
            peid: int | None = self.cur_parent_edge.get(v)
            if peid is not None and self.g.has_edge(peid):
                self.raw_comps.append([eid, peid])
                self.consumed.add(eid)
                self.consumed.add(peid)
                ve: Edge = self._add_virt(v, w)
                self.cur_tree.add(ve.id)
                self.cur_parent_edge[v] = ve.id
                self.adj_cache[v] = list(
                    self.g.adj_edge_ids(v))
                self.adj_cache[w] = list(
                    self.g.adj_edge_ids(w))
            else:
                self.estack.append(eid)
        else:
            self.estack.append(eid)

    def _eval_type2_conds(
        self, v_num: int, w: Hashable,
    ) -> tuple[bool, bool]:
        """Evaluate the two type-2 loop conditions.

        Separates the boolean ``and``/``or`` sequences
        from ``_check_type2`` to keep cognitive complexity
        within limits.

        :param v_num: DFS number of the current vertex.
        :param w: The child vertex.
        :return: (top_cond, deg2_cond) tuple.
        """
        top_cond: bool = (
            self.tstack
            and self.tstack[-1] != _EOS
            and self.tstack[-1][1] == v_num)
        deg2_cond: bool = (
            self.cur_deg.get(w, 0) == 2
            and self._temp_target_num(w)
            > self.dfs_num[w])
        return top_cond, deg2_cond

    def _type2_try_parent_pop(
        self, v: Hashable, top_cond: bool,
    ) -> bool:
        """Try to pop a TSTACK triple whose parent is v.

        If the top-of-stack condition holds and the parent of
        b is v, pops the triple and returns True.  Otherwise
        returns False without modifying state.

        :param v: The current vertex.
        :param top_cond: Whether the TSTACK top condition holds.
        :return: True if a triple was popped, False otherwise.
        """
        if not top_cond:
            return False
        b: int = self.tstack[-1][2]
        b_v: Hashable = self.inv_dfs[b]
        if self.parent.get(b_v) != v:
            return False
        self.tstack.pop()
        return True

    def _type2_apply(
        self, v: Hashable, w: Hashable,
        deg2_cond: bool,
    ) -> Hashable | None:
        """Apply a type-2 split (deg2 or top variant).

        Delegates to ``_check_type2_deg2`` when *deg2_cond*
        is true, otherwise to ``_check_type2_top``.

        :param v: The current vertex.
        :param w: The current child vertex.
        :param deg2_cond: Whether the degree-2 condition holds.
        :return: New w vertex, or None.
        """
        if deg2_cond:
            return self._check_type2_deg2(v, w)
        return self._check_type2_top(v)

    def _check_type2(
        self, v: Hashable, w: Hashable,
    ) -> Hashable:
        """Check for type-2 separation pairs (Algorithm 5).

        Implements Algorithm 5 from Gutwenger-Mutzel (2001).
        May update w if the algorithm creates new virtual edges.

        Priority follows SageMath: (1) if top_cond and parent
        of b_v is v, pop TSTACK and continue; (2) if deg2_cond,
        do deg2 split; (3) otherwise do top split.

        :param v: The current vertex label.
        :param w: The child vertex (just processed).
        :return: Updated w (may change if edges merge).
        """
        v_num: int = self.dfs_num[v]
        if v_num == 1:
            return w

        while True:
            top_cond, deg2_cond = self._eval_type2_conds(
                v_num, w)
            if not top_cond and not deg2_cond:
                break
            if self._type2_try_parent_pop(v, top_cond):
                continue
            new_w: Hashable | None = self._type2_apply(
                v, w, deg2_cond)
            if new_w is not None:
                w = new_w
            elif deg2_cond:
                break

        return w

    def _check_type2_top(
        self, v: Hashable,
    ) -> Hashable | None:
        """Handle TSTACK-based type-2 separation pair.

        Pops the top triple and splits edges in the range
        [a, h].  The parent-check pop is handled by the
        caller.  Returns the new w vertex, or None to signal
        that the caller should continue (no-op pop).

        :param v: The current vertex.
        :return: New w vertex, or None for continue.
        """
        h, a, b = self.tstack[-1]
        self.tstack.pop()
        a_v: Hashable = self.inv_dfs[a]
        b_v = self.inv_dfs[b]
        comp_eids, e_ab = self._pop_range_edges(a, b, h)
        if not comp_eids:
            if e_ab is not None:
                self.estack.append(e_ab)
            return None
        if e_ab is not None:
            e_ab_ed: Edge | None = self.g.get_edge(e_ab)
            self._del_high(e_ab)
            self.consumed.add(e_ab)
            if e_ab_ed is not None:
                self.cur_deg[e_ab_ed.u] -= 1
                self.cur_deg[e_ab_ed.v] -= 1
        ve: Edge = self._add_virt(a_v, b_v)
        comp_eids.append(ve.id)
        self.raw_comps.append(comp_eids)
        if e_ab is not None:
            ve2: Edge = self._add_virt(a_v, b_v)
            self.raw_comps.append([e_ab, ve.id, ve2.id])
            ve = ve2
        self.estack.append(ve.id)
        self.adj_cache.setdefault(
            a_v, []).append(ve.id)
        self.cur_deg[a_v] += 1
        self.cur_deg[b_v] += 1
        self.cur_tree.add(ve.id)
        self.parent[b_v] = v
        self.cur_parent_edge[b_v] = ve.id
        return b_v

    def _pop_range_edges(
        self, a: int, b: int, h: int,
    ) -> tuple[list[int], int | None]:
        """Pop ESTACK edges within DFS range [a, h].

        Pops edges where both endpoints have DFS numbers in
        [a, h].  The edge connecting vertices a and b (if
        found) is separated out as *e_ab* and not consumed.

        :param a: Lower bound of DFS range.
        :param b: DFS number of the second separation vertex.
        :param h: Upper bound of DFS range.
        :return: (comp_eids, e_ab) — component edge IDs and
            the optional {a, b} edge ID.
        """
        comp_eids: list[int] = []
        e_ab: int | None = None
        while self.estack:
            eid: int = self.estack[-1]
            ed: Edge | None = self.g.get_edge(eid)
            if ed is None:
                self.estack.pop()
                continue
            eu: int = self.dfs_num.get(ed.u, -1)
            ev_: int = self.dfs_num.get(ed.v, -1)
            if not (a <= eu <= h and a <= ev_ <= h):
                break
            self.estack.pop()
            if {eu, ev_} == {a, b}:
                e_ab = eid
            else:
                self._del_high(eid)
                self.consumed.add(eid)
                self.cur_deg[ed.u] -= 1
                self.cur_deg[ed.v] -= 1
                comp_eids.append(eid)
        return comp_eids, e_ab

    def _check_type2_deg2(
        self, v: Hashable, w: Hashable,
    ) -> Hashable | None:
        """Handle degree-2 based type-2 separation pair.

        Pops two edges from ESTACK and creates a split
        component.  Returns the new w vertex, or None to
        signal that the caller should break.

        :param v: The current vertex.
        :param w: The current child vertex.
        :return: New w vertex, or None for break.
        """
        if len(self.estack) < 2:
            return None
        e1id: int = self.estack.pop()
        e2id: int = self.estack.pop()
        ed1: Edge | None = self.g.get_edge(e1id)
        ed2: Edge | None = self.g.get_edge(e2id)
        if ed1 is None or ed2 is None:
            if ed2 is not None:
                self.estack.append(e2id)
            if ed1 is not None:
                self.estack.append(e1id)
            return None
        verts: set[Hashable] = (
            {ed1.u, ed1.v, ed2.u, ed2.v} - {v, w})
        if not verts:
            self.estack.append(e2id)
            self.estack.append(e1id)
            return None
        b_v: Hashable = min(
            verts, key=lambda x: self.dfs_num.get(x, 0))
        self.consumed.add(e1id)
        self.consumed.add(e2id)
        ve: Edge = self._add_virt(v, b_v)
        self.cur_deg[v] -= 1
        self.cur_deg[b_v] -= 1
        comp_eids: list[int] = [e1id, e2id, ve.id]
        self.raw_comps.append(comp_eids)
        e_ab: int | None = None
        if self.estack:
            top_e: Edge | None = self.g.get_edge(
                self.estack[-1])
            if (top_e is not None
                    and {top_e.u, top_e.v} == {v, b_v}):
                e_ab = self.estack.pop()
                self._del_high(e_ab)
                self.consumed.add(e_ab)
        if e_ab is not None:
            ve2: Edge = self._add_virt(v, b_v)
            self.raw_comps.append([e_ab, ve.id, ve2.id])
            self.cur_deg[v] -= 1
            self.cur_deg[b_v] -= 1
            ve = ve2
        self.estack.append(ve.id)
        self.adj_cache.setdefault(
            v, []).append(ve.id)
        self.cur_deg[v] += 1
        self.cur_deg[b_v] += 1
        self.cur_tree.add(ve.id)
        self.parent[b_v] = v
        self.cur_parent_edge[b_v] = ve.id
        return b_v

    def _pop_subtree_edges(
        self, w_lo: int, w_hi: int,
    ) -> list[int]:
        """Pop ESTACK edges within a subtree DFS range.

        Pops edges whose at least one endpoint has a DFS
        number in [w_lo, w_hi].  Decrements degrees for
        consumed edges.

        :param w_lo: Lower bound of subtree DFS range.
        :param w_hi: Upper bound of subtree DFS range.
        :return: List of popped edge IDs.
        """
        comp_eids: list[int] = []
        while self.estack:
            eid: int = self.estack[-1]
            ed: Edge | None = self.g.get_edge(eid)
            if ed is None:
                self.estack.pop()
                continue
            eu: int = self.dfs_num.get(ed.u, -1)
            ev_: int = self.dfs_num.get(ed.v, -1)
            in_range: bool = ((w_lo <= eu <= w_hi)
                              or (w_lo <= ev_ <= w_hi))
            if not in_range:
                break
            self.estack.pop()
            self._del_high(eid)
            self.consumed.add(eid)
            comp_eids.append(eid)
            self.cur_deg[ed.u] -= 1
            self.cur_deg[ed.v] -= 1
        return comp_eids

    def _type1_try_combine(
        self, v: Hashable, lp1w_v: Hashable,
        ve: Edge,
    ) -> Edge:
        """Try to combine ESTACK top into a bond.

        If the ESTACK top edge connects {v, lp1w_v}, pops it
        and creates a bond component.  Returns the (possibly
        updated) virtual edge.

        :param v: The current vertex.
        :param lp1w_v: The lowpt1(w) vertex.
        :param ve: The current virtual edge.
        :return: Updated virtual edge (may be a new one).
        """
        if not self.estack:
            return ve
        top_e: Edge | None = self.g.get_edge(
            self.estack[-1])
        if (top_e is None
                or {top_e.u, top_e.v} != {v, lp1w_v}):
            return ve
        e_top: int = self.estack.pop()
        self.consumed.add(e_top)
        ve2: Edge = self._add_virt(v, lp1w_v)
        self.raw_comps.append(
            [e_top, ve.id, ve2.id])
        self.cur_deg[v] -= 1
        self.cur_deg[lp1w_v] -= 1
        if e_top in self.in_high:
            self.in_high[ve2.id] = \
                self.in_high.pop(e_top)
        return ve2

    def _type1_parent_bond(
        self, v: Hashable, lp1w_v: Hashable,
        ve: Edge,
    ) -> None:
        """Handle the lp1w == pv_num branch of type-1.

        Creates a new virtual edge replacing the parent edge,
        and appends a bond component.

        :param v: The current vertex.
        :param lp1w_v: The lowpt1(w) vertex.
        :param ve: The virtual edge for the split component.
        :return: None
        """
        peid: int | None = self.cur_parent_edge.get(v)
        ve2: Edge = self._add_virt(lp1w_v, v)
        self.cur_tree.add(ve2.id)
        self.cur_parent_edge[v] = ve2.id
        if peid is not None and self.g.has_edge(peid):
            self.consumed.add(peid)
            if peid in self.in_high:
                self.in_high[ve2.id] = \
                    self.in_high.pop(peid)
            self.raw_comps.append(
                [ve.id, peid, ve2.id])
        else:
            self.raw_comps.append([ve.id, ve2.id])

    def _check_type1(
        self, v: Hashable, w: Hashable,
    ) -> None:
        """Check for type-1 separation pair (Algorithm 6).

        Implements Algorithm 6 from Gutwenger-Mutzel (2001).

        :param v: The current vertex label.
        :param w: The child vertex (just processed).
        :return: None
        """
        v_num: int = self.dfs_num[v]
        w_num: int = self.dfs_num[w]
        lp1w: int = self.lowpt1[w]
        lp2w: int = self.lowpt2[w]

        if not (lp1w < v_num <= lp2w):
            return
        pv: Hashable | None = self.parent.get(v)
        pv_num: int = (self.dfs_num.get(pv, 0)
                       if pv is not None else 0)
        has_more: bool = _has_unvisited_arc(
            v, w_num, self.cur_children, self.dfs_num)
        if not (pv_num != 1 or has_more):
            return

        w_lo: int = w_num
        w_hi: int = w_num + self.nd[w] - 1
        comp_eids: list[int] = self._pop_subtree_edges(
            w_lo, w_hi)

        lp1w_v: Hashable = self.inv_dfs[lp1w]
        ve: Edge = self._add_virt(v, lp1w_v)
        comp_eids.append(ve.id)
        self.raw_comps.append(comp_eids)

        ve = self._type1_try_combine(v, lp1w_v, ve)

        if lp1w != pv_num:
            self.estack.append(ve.id)
            self.adj_cache.setdefault(
                v, []).append(ve.id)
            self.cur_deg[v] += 1
            self.cur_deg[lp1w_v] += 1
            self.cur_tree.add(ve.id)
            if (ve.id not in self.in_high
                    and self._high(lp1w) < v_num):
                bisect.insort(
                    self.frond_srcs[lp1w], v_num)
                self.in_high[ve.id] = (lp1w, v_num)
        else:
            self._type1_parent_bond(v, lp1w_v, ve)

        ch: list[Hashable] = self.cur_children.get(v, [])
        if w in ch:
            ch.remove(w)


def _phase_pathsearch(
    g: MultiGraph,
    raw_comps: list[list[int]],
    virtual_ids: set[int],
) -> None:
    """Detect separation pairs and split components via PathSearch.

    Implements Algorithms 3-6 from Gutwenger-Mutzel (2001).  Runs an
    iterative DFS over the phi-sorted adjacency lists, maintaining an
    edge stack (ESTACK) and a triple stack (TSTACK) to detect type-1
    and type-2 separation pairs.

    :param g: The working multigraph (modified in place).
    :param raw_comps: List to append new split components to.
    :param virtual_ids: Set to add new virtual edge IDs to.
    :return: None
    """
    searcher: _PathSearcher = _PathSearcher(
        g, raw_comps, virtual_ids)
    searcher.run()


def _has_unvisited_arc(
    v: Hashable,
    w_num: int,
    cur_children: dict[Hashable, list[Hashable]],
    dfs_num: dict[Hashable, int],
) -> bool:
    """Check if v has a tree child with DFS number > w_num.

    Used in Algorithm 6 to determine whether v is adjacent to a
    not-yet-visited tree arc (i.e., has another child after w).

    :param v: The vertex to check.
    :param w_num: DFS number of the current child w.
    :param cur_children: Current DFS children for each vertex.
    :param dfs_num: DFS discovery numbers.
    :return: True if v has another child with dfs_num > w_num.
    """
    for c in cur_children.get(v, []):
        if dfs_num.get(c, 0) > w_num:
            return True
    return False


# ---------------------------------------------------------------------------
# Phase 3: Classify and merge split components
# ---------------------------------------------------------------------------

def _make_edge_list(
    eids: list[int],
    all_edges: dict[int, Edge],
) -> list[Edge]:
    """Resolve edge IDs to Edge objects (skipping unknowns).

    :param eids: List of edge IDs.
    :param all_edges: Mapping from edge ID to Edge object.
    :return: List of Edge objects.
    """
    result: list[Edge] = []
    for eid in eids:
        if eid in all_edges:
            result.append(all_edges[eid])
    return result


def _classify_component(
    edges: list[Edge],
) -> ComponentType:
    """Classify a set of edges as BOND, POLYGON, or TRICONNECTED.

    :param edges: The edges of the component.
    :return: The ComponentType.
    """
    verts: set[Hashable] = set()
    for e in edges:
        verts.add(e.u)
        verts.add(e.v)
    if len(verts) == 2:
        return ComponentType.BOND
    # Polygon: every vertex has degree 2.
    deg_map: dict[Hashable, int] = dict.fromkeys(verts, 0)
    for e in edges:
        deg_map[e.u] += 1
        deg_map[e.v] += 1
    if all(d == 2 for d in deg_map.values()):
        return ComponentType.POLYGON
    return ComponentType.TRICONNECTED


def _phase_classify_merge(
    raw_comps: list[list[int]],
    virtual_ids: set[int],
    orig_graph: MultiGraph,
    work_graph: MultiGraph,
) -> list[TriconnectedComponent]:
    """Classify raw split components and merge same-type adjacent ones.

    Classifies each split component as BOND, POLYGON, or TRICONNECTED.
    Then merges adjacent components of the same type that share a virtual
    edge (Algorithm 2).

    :param raw_comps: Raw split components from Phases 1 and 2.
    :param virtual_ids: Set of virtual edge IDs.
    :param orig_graph: The original input graph.
    :param work_graph: The working graph after PathSearch.
    :return: Final list of TriconnectedComponent objects.
    """
    # Build a unified edge dictionary.
    all_edges: dict[int, Edge] = {}
    for e in orig_graph.edges:
        all_edges[e.id] = e
    for e in work_graph.edges:
        if e.id not in all_edges:
            all_edges[e.id] = e

    # Build classified components.
    comps: list[TriconnectedComponent] = []
    for eids in raw_comps:
        edges: list[Edge] = _make_edge_list(eids, all_edges)
        if len(edges) < 2:
            continue
        ctype: ComponentType = _classify_component(edges)
        comps.append(TriconnectedComponent(
            type=ctype, edges=edges))

    if not comps:
        return comps

    # Merge adjacent same-type components (Algorithm 2).
    return _merge_components(comps, virtual_ids)


def _uf_find(uf: list[int], x: int) -> int:
    """Find root of x's set in the union-find structure.

    Uses path splitting for amortised near-constant time.

    :param uf: The union-find parent array.
    :param x: Index to find.
    :return: Root index.
    """
    while uf[x] != x:
        uf[x] = uf[uf[x]]
        x = uf[x]
    return x


def _uf_union(uf: list[int], x: int, y: int) -> None:
    """Union the sets containing x and y.

    :param uf: The union-find parent array.
    :param x: First element.
    :param y: Second element.
    :return: None
    """
    rx: int = _uf_find(uf, x)
    ry: int = _uf_find(uf, y)
    if rx != ry:
        uf[rx] = ry


def _collect_merge_groups(
    comps: list[TriconnectedComponent],
    virtual_ids: set[int],
) -> tuple[dict[int, list[int]], set[int]]:
    """Identify groups of same-type components to merge.

    Uses union-find to group components that share virtual edges
    and have the same type.

    :param comps: Initial classified components.
    :param virtual_ids: Set of virtual edge IDs.
    :return: A (groups, internal_ves) tuple where groups maps
        representative index to list of member indices, and
        internal_ves is the set of virtual edge IDs consumed
        by merging.
    """
    n: int = len(comps)

    # Map: virtual edge ID -> list of component indices.
    ve_to_comps: dict[int, list[int]] = {}
    for i, comp in enumerate(comps):
        for e in comp.edges:
            if e.id in virtual_ids:
                ve_to_comps.setdefault(e.id, []).append(i)

    # Union-Find for merging groups.
    uf: list[int] = list(range(n))

    # Virtual edge IDs consumed by merging.
    internal_ves: set[int] = set()

    for veid, idxs in ve_to_comps.items():
        if len(idxs) < 2:
            continue
        types_set: set[ComponentType] = {
            comps[i].type for i in idxs}
        if len(types_set) == 1:
            for k in range(1, len(idxs)):
                _uf_union(uf, idxs[0], idxs[k])
            internal_ves.add(veid)

    # Group component indices by representative.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        r: int = _uf_find(uf, i)
        groups.setdefault(r, []).append(i)

    return groups, internal_ves


def _merge_group_edges(
    comps: list[TriconnectedComponent],
    group_idxs: list[int],
    internal_ves: set[int],
) -> TriconnectedComponent | None:
    """Merge edges from multiple same-type components.

    Combines all edges from the group, removing internal virtual
    edges that were shared between the merged components.

    :param comps: All classified components.
    :param group_idxs: Indices of components in this group.
    :param internal_ves: Virtual edge IDs to exclude.
    :return: A merged TriconnectedComponent, or None if fewer
        than 2 edges remain.
    """
    merged_type: ComponentType = comps[group_idxs[0]].type
    seen_eids: set[int] = set()
    merged_edges: list[Edge] = []
    for idx in group_idxs:
        for e in comps[idx].edges:
            if e.id in seen_eids:
                continue
            seen_eids.add(e.id)
            if e.id in internal_ves:
                continue
            merged_edges.append(e)
    if len(merged_edges) >= 2:
        return TriconnectedComponent(
            type=merged_type, edges=merged_edges)
    return None


def _build_merged_result(
    comps: list[TriconnectedComponent],
    groups: dict[int, list[int]],
    internal_ves: set[int],
) -> list[TriconnectedComponent]:
    """Build the final merged component list from groups.

    :param comps: All classified components.
    :param groups: Mapping from representative to member indices.
    :param internal_ves: Virtual edge IDs to exclude.
    :return: Merged list of TriconnectedComponent objects.
    """
    result: list[TriconnectedComponent] = []
    for group_idxs in groups.values():
        if len(group_idxs) == 1:
            idx: int = group_idxs[0]
            edges: list[Edge] = [
                e for e in comps[idx].edges
                if e.id not in internal_ves
            ]
            result.append(TriconnectedComponent(
                type=comps[idx].type, edges=edges))
            continue
        merged: TriconnectedComponent | None = \
            _merge_group_edges(
                comps, group_idxs, internal_ves)
        if merged is not None:
            result.append(merged)
    return result


def _merge_components(
    comps: list[TriconnectedComponent],
    virtual_ids: set[int],
) -> list[TriconnectedComponent]:
    """Merge adjacent same-type components sharing a virtual edge.

    Two or more components are adjacent if they share a virtual edge.
    When all components sharing a virtual edge are the same type
    (BOND+BOND or POLYGON+POLYGON), they are merged transitively: the
    shared virtual edge is removed and the remaining edges are combined.

    :param comps: Initial classified components.
    :param virtual_ids: Set of virtual edge IDs.
    :return: Merged list of TriconnectedComponent objects.
    """
    if len(comps) == 0:
        return comps

    groups: dict[int, list[int]]
    internal_ves: set[int]
    groups, internal_ves = _collect_merge_groups(
        comps, virtual_ids)

    return _build_merged_result(comps, groups, internal_ves)
