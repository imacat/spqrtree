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
"""SPQR-Tree: a pure Python implementation.

This package provides an SPQR-Tree data structure for biconnected graphs,
based on the algorithm by Gutwenger & Mutzel (2001) with corrections to
Hopcroft-Tarjan (1973), and data structure definitions from
Di Battista & Tamassia (1996).

Public API::

    from spqrtree import Edge, MultiGraph
    from spqrtree import NodeType, SPQRNode, build_spqr_tree
    from spqrtree import (
        ComponentType,
        TriconnectedComponent,
        find_triconnected_components,
    )
"""
from collections import deque

from spqrtree._graph import Edge, MultiGraph
from spqrtree._spqr import NodeType, SPQRNode, build_spqr_tree
from spqrtree._triconnected import (
    ComponentType,
    TriconnectedComponent,
    find_triconnected_components,
)

VERSION: str = "0.0.0"
"""The package version."""
__all__: list[str] = [
    "SPQRTree", "SPQRNode", "NodeType",
    "Edge", "MultiGraph",
    "build_spqr_tree",
    "ComponentType", "TriconnectedComponent",
    "find_triconnected_components",
]


class SPQRTree:
    """An SPQR-Tree for a biconnected graph.

    Constructs the SPQR-Tree from a biconnected multigraph,
    using the Gutwenger-Mutzel triconnected components algorithm.
    """

    def __init__(self, graph: MultiGraph | dict) -> None:
        """Initialize the SPQR-Tree from a graph.

        :param graph: A MultiGraph or dict representing the input graph.
        :return: None
        """
        if isinstance(graph, dict):
            g: MultiGraph = MultiGraph()
            seen: set[frozenset] = set()
            for u, neighbors in graph.items():
                g.add_vertex(u)
                for v in neighbors:
                    pair: frozenset = frozenset((u, v))
                    if pair not in seen:
                        seen.add(pair)
                        g.add_edge(u, v)
        else:
            g = graph
        self._root: SPQRNode = build_spqr_tree(g)
        """The root node of the SPQR-Tree."""

    @property
    def root(self) -> SPQRNode:
        """Return the root node of the SPQR-Tree.

        :return: The root SPQRNode.
        """
        return self._root

    def nodes(self) -> list[SPQRNode]:
        """Return all nodes of the SPQR-Tree in BFS order.

        :return: A list of all SPQRNode objects.
        """
        result: list[SPQRNode] = []
        bfs_queue: deque[SPQRNode] = deque([self._root])
        while bfs_queue:
            node: SPQRNode = bfs_queue.popleft()
            result.append(node)
            for child in node.children:
                bfs_queue.append(child)
        return result
