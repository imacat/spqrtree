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
"""SPQR-Tree construction from triconnected split components.

Builds an SPQR-tree from the triconnected split components of a
biconnected multigraph, following Di Battista & Tamassia (1996).

Each triconnected split component maps to an SPQR-tree node:
- BOND with 2 total edges -> Q-node (degenerate: a single edge).
- BOND with 3+ total edges -> P-node (parallel edges).
- POLYGON -> S-node (simple cycle).
- TRICONNECTED -> R-node (3-connected subgraph).

Adjacent components (sharing a virtual edge) are linked as
parent-child pairs in the tree.

Public API::

    from spqrtree._spqr import NodeType, SPQRNode, build_spqr_tree
"""
from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass, field
from enum import Enum

from spqrtree._graph import MultiGraph
from spqrtree._triconnected import (
    ComponentType,
    TriconnectedComponent,
    find_triconnected_components,
)


class NodeType(Enum):
    """SPQR-tree node types.

    :cvar Q: A Q-node represents a single edge (degenerate bond).
    :cvar S: An S-node represents a simple cycle (series).
    :cvar P: A P-node represents parallel edges (parallel).
    :cvar R: An R-node represents a 3-connected subgraph (rigid).
    """

    Q = "Q"
    """Q-node: a single real edge."""

    S = "S"
    """S-node: a simple cycle (polygon)."""

    P = "P"
    """P-node: parallel edges (bond with 3+ real edges)."""

    R = "R"
    """R-node: a 3-connected subgraph."""


@dataclass
class SPQRNode:
    """A node in the SPQR-tree.

    :param type: The node type (Q, S, P, or R).
    :param skeleton: The skeleton graph for this node, containing
        the real and virtual edges of the component.
    :param poles: The pair of poles (u, v) for this node, i.e., the
        two vertices of the virtual edge connecting this node to its
        parent (or the two endpoints for the root).
    :param parent: The parent SPQRNode, or None if this is the root.
    :param children: Ordered list of child SPQRNodes.
    """

    type: NodeType
    """The SPQR-tree node type."""

    skeleton: MultiGraph
    """The skeleton graph of this node."""

    poles: tuple[Hashable, Hashable]
    """The two pole vertices of this node."""

    parent: SPQRNode | None
    """Parent node, or None if this is the root."""

    children: list[SPQRNode] = field(default_factory=list)
    """Child nodes in the SPQR-tree."""


def build_spqr_tree(graph: MultiGraph) -> SPQRNode:
    """Build an SPQR-tree for a biconnected multigraph.

    Decomposes the graph into triconnected split components, then
    assembles them into an SPQR-tree.  Each component becomes a node
    with type Q (degenerate bond of 2 edges), P (bond of 3+ edges),
    S (polygon), or R (triconnected).

    :param graph: A biconnected multigraph.
    :return: The root SPQRNode of the SPQR-tree.
    """
    comps: list[TriconnectedComponent] = find_triconnected_components(
        graph)

    if not comps:
        # Degenerate: single edge graph.
        skel: MultiGraph = MultiGraph()
        for e in graph.edges:
            skel.add_edge(e.u, e.v, virtual=e.virtual)
        verts: list[Hashable] = list(graph.vertices)
        poles: tuple[Hashable, Hashable] = (
            verts[0],
            verts[1] if len(verts) > 1 else verts[0])
        return SPQRNode(
            type=NodeType.Q,
            skeleton=skel,
            poles=poles,
            parent=None,
        )

    if len(comps) == 1:
        return _make_single_node(comps[0], None)

    # Multiple components: build tree from adjacency.
    return _build_tree_from_components(comps)


def _comp_to_node_type(comp: TriconnectedComponent) -> NodeType:
    """Convert a TriconnectedComponent type to an SPQRNode NodeType.

    A BOND with exactly 2 total edges (one real + one virtual) is a
    Q-node (degenerate: a single edge).  A BOND with 3 or more total
    edges is a P-node (parallel class).

    :param comp: The triconnected component.
    :return: The corresponding NodeType.
    """
    if comp.type == ComponentType.BOND:
        if len(comp.edges) <= 2:
            return NodeType.Q
        return NodeType.P
    if comp.type == ComponentType.POLYGON:
        return NodeType.S
    return NodeType.R


def _make_skeleton(comp: TriconnectedComponent) -> MultiGraph:
    """Build a skeleton MultiGraph for a triconnected component.

    :param comp: The triconnected component.
    :return: A MultiGraph containing all edges of the component.
    """
    skel: MultiGraph = MultiGraph()
    for e in comp.edges:
        skel.add_edge(e.u, e.v, virtual=e.virtual)
    return skel


def _get_poles(
    comp: TriconnectedComponent,
) -> tuple[Hashable, Hashable]:
    """Determine the poles of a component.

    The poles are the two endpoints of the virtual edge in the
    component (if any), or the two endpoints of the first real edge.

    :param comp: The triconnected component.
    :return: A (u, v) pair of pole vertices.
    :raises RuntimeError: If the component has no edges.
    """
    for e in comp.edges:
        if e.virtual:
            return e.u, e.v
    # No virtual edge: use first edge endpoints.
    if comp.edges:
        return comp.edges[0].u, comp.edges[0].v
    raise RuntimeError("Component has no edges")


def _make_single_node(
    comp: TriconnectedComponent,
    parent_node: SPQRNode | None,
) -> SPQRNode:
    """Create a single SPQRNode from one TriconnectedComponent.

    :param comp: The triconnected component.
    :param parent_node: The parent SPQRNode, or None for root.
    :return: A new SPQRNode.
    """
    ntype: NodeType = _comp_to_node_type(comp)
    skel: MultiGraph = _make_skeleton(comp)
    poles: tuple[Hashable, Hashable] = _get_poles(comp)
    node: SPQRNode = SPQRNode(
        type=ntype,
        skeleton=skel,
        poles=poles,
        parent=parent_node,
    )
    return node


def _collect_ve_to_comps(
    comps: list[TriconnectedComponent],
) -> dict[int, list[int]]:
    """Map virtual edge IDs to component indices containing them.

    Scans all components for virtual edges, then builds a mapping
    from each virtual edge ID to the list of component indices
    that include that edge.

    :param comps: List of triconnected components.
    :return: Mapping from virtual edge ID to component indices.
    """
    virtual_edge_ids: set[int] = set()
    for comp in comps:
        for e in comp.edges:
            if e.virtual:
                virtual_edge_ids.add(e.id)

    ve_to_comps: dict[int, list[int]] = {}
    for i, comp in enumerate(comps):
        for e in comp.edges:
            if e.id in virtual_edge_ids:
                ve_to_comps.setdefault(e.id, []).append(i)
    return ve_to_comps


def _build_adj_and_root(
    comps: list[TriconnectedComponent],
    ve_to_comps: dict[int, list[int]],
) -> tuple[dict[int, list[tuple[int, int]]], int]:
    """Build component adjacency and choose the root component.

    Two components are adjacent if they share a virtual edge.  The
    root is the component with the most virtual-edge adjacencies
    (the most central node in the tree).

    :param comps: List of triconnected components.
    :param ve_to_comps: Virtual edge to component indices mapping.
    :return: A (adj, root_ci) tuple where adj maps component index
        to list of (neighbor_index, shared_ve_id) pairs, and
        root_ci is the chosen root component index.
    """
    adj: dict[int, list[tuple[int, int]]] = {
        i: [] for i in range(len(comps))
    }
    for veid, idxs in ve_to_comps.items():
        if len(idxs) == 2:
            i, j = idxs
            adj[i].append((j, veid))
            adj[j].append((i, veid))

    adj_count: list[int] = [0] * len(comps)
    for idxs in ve_to_comps.values():
        if len(idxs) == 2:
            adj_count[idxs[0]] += 1
            adj_count[idxs[1]] += 1
    root_ci = max(range(len(comps)), key=lambda i: adj_count[i])
    return adj, root_ci


def _build_tree_from_components(
    comps: list[TriconnectedComponent],
) -> SPQRNode:
    """Build an SPQR-tree from a list of triconnected components.

    Components sharing a virtual edge are adjacent in the tree.
    The component with the most virtual-edge adjacencies is the root.

    :param comps: List of triconnected components.
    :return: The root SPQRNode.
    """
    ve_to_comps: dict[int, list[int]] = _collect_ve_to_comps(comps)
    adj: dict[int, list[tuple[int, int]]]
    root_ci: int
    adj, root_ci = _build_adj_and_root(
        comps, ve_to_comps)

    # Build the tree by BFS from the chosen root component.
    nodes: list[SPQRNode | None] = [None] * len(comps)
    visited: set[int] = set()
    bfs_queue: deque[tuple[int, SPQRNode | None]] = deque([
        (root_ci, None)])

    while bfs_queue:
        ci, parent_node = bfs_queue.popleft()
        if ci in visited:
            continue
        visited.add(ci)

        comp: TriconnectedComponent = comps[ci]
        ntype: NodeType = _comp_to_node_type(comp)
        skel: MultiGraph = _make_skeleton(comp)
        poles: tuple[Hashable, Hashable] = _get_poles(comp)
        node: SPQRNode = SPQRNode(
            type=ntype,
            skeleton=skel,
            poles=poles,
            parent=parent_node,
        )
        if parent_node is not None:
            parent_node.children.append(node)
        nodes[ci] = node

        for cj, _ in adj[ci]:
            if cj not in visited:
                bfs_queue.append((cj, node))

    # Handle any disconnected components (shouldn't happen for
    # biconnected graphs, but be defensive).
    for i in range(len(comps)):
        if nodes[i] is None:
            comp = comps[i]
            ntype = _comp_to_node_type(comp)
            skel = _make_skeleton(comp)
            poles = _get_poles(comp)
            nodes[i] = SPQRNode(
                type=ntype,
                skeleton=skel,
                poles=poles,
                parent=None,
            )

    # Return the chosen root node.
    root: SPQRNode | None = nodes[root_ci]
    assert root is not None
    return root
