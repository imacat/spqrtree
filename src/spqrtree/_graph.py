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
"""Internal multigraph data structure for the SPQR-Tree algorithm.

Provides Edge and MultiGraph classes supporting parallel edges and
virtual edges, identified by integer IDs.
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass


@dataclass
class Edge:
    """An edge in a multigraph, identified by a unique integer ID.

    Supports parallel edges (multiple edges between the same pair of
    vertices) and virtual edges used internally by the SPQR-Tree
    algorithm.
    """

    id: int
    """Unique integer identifier for this edge."""

    u: Hashable
    """One endpoint of the edge."""

    v: Hashable
    """The other endpoint of the edge."""

    virtual: bool = False
    """Whether this is a virtual edge (used in SPQR skeletons)."""

    def endpoints(self) -> tuple[Hashable, Hashable]:
        """Return both endpoints as a tuple.

        :return: A tuple (u, v) of the two endpoints.
        """
        return self.u, self.v

    def other(self, vertex: Hashable) -> Hashable:
        """Return the endpoint opposite to the given vertex.

        :param vertex: One endpoint of this edge.
        :return: The other endpoint.
        :raises ValueError: If vertex is not an endpoint of this edge.
        """
        if vertex == self.u:
            return self.v
        if vertex == self.v:
            return self.u
        raise ValueError(
            f"Vertex {vertex} is not an endpoint of edge {self.id}"
        )


class MultiGraph:
    """An undirected multigraph supporting parallel edges and virtual edges.

    Vertices are arbitrary hashable values. Edges are identified by
    unique integer IDs. Supports parallel edges (multiple edges between
    the same vertex pair).
    """

    def __init__(self) -> None:
        """Initialize an empty multigraph.

        :return: None
        """
        self._vertices: set[Hashable] = set()
        """The set of vertices in this graph."""

        self._edges: dict[int, Edge] = {}
        """Map from edge ID to Edge object."""

        self._adj: dict[Hashable, list[int]] = {}
        """Adjacency list: vertex -> list of edge IDs."""

        self._next_edge_id: int = 0
        """Counter for assigning unique edge IDs."""

    @property
    def vertices(self) -> set[Hashable]:
        """Return the set of all vertices.

        :return: The set of vertices.
        """
        return self._vertices

    @property
    def edges(self) -> list[Edge]:
        """Return a list of all edges in the graph.

        :return: List of Edge objects.
        """
        return list(self._edges.values())

    def add_vertex(self, v: Hashable) -> None:
        """Add a vertex to the graph.  No-op if already present.

        :param v: The vertex to add.
        :return: None
        """
        if v not in self._vertices:
            self._vertices.add(v)
            self._adj[v] = []

    def remove_vertex(self, v: Hashable) -> None:
        """Remove a vertex and all its incident edges from the graph.

        :param v: The vertex to remove.
        :return: None
        :raises KeyError: If the vertex does not exist.
        """
        if v not in self._vertices:
            raise KeyError(f"Vertex {v} not in graph")
        edge_ids: list[int] = list(self._adj[v])
        for eid in edge_ids:
            self.remove_edge(eid)
        self._vertices.remove(v)
        del self._adj[v]

    def add_edge(
        self, u: Hashable, v: Hashable, virtual: bool = False
    ) -> Edge:
        """Add an edge between two vertices and return it.

        Automatically adds vertices u and v if not already present.

        :param u: One endpoint vertex.
        :param v: The other endpoint vertex.
        :param virtual: Whether this is a virtual edge.
        :return: The newly created Edge object.
        """
        self.add_vertex(u)
        self.add_vertex(v)
        eid: int = self._next_edge_id
        self._next_edge_id += 1
        e: Edge = Edge(id=eid, u=u, v=v, virtual=virtual)
        self._edges[eid] = e
        self._adj[u].append(eid)
        if u != v:
            self._adj[v].append(eid)
        return e

    def remove_edge(self, edge_id: int) -> None:
        """Remove an edge from the graph by its ID.

        :param edge_id: The integer ID of the edge to remove.
        :return: None
        :raises KeyError: If the edge ID does not exist.
        """
        if edge_id not in self._edges:
            raise KeyError(f"Edge {edge_id} not in graph")
        e: Edge = self._edges.pop(edge_id)
        self._adj[e.u].remove(edge_id)
        if e.u != e.v:
            self._adj[e.v].remove(edge_id)

    def get_edge(self, edge_id: int) -> Edge | None:
        """Return the edge with the given ID, or None if not found.

        :param edge_id: The integer ID of the edge to look up.
        :return: The Edge object, or None if no such edge exists.
        """
        return self._edges.get(edge_id)

    def has_edge(self, edge_id: int) -> bool:
        """Check whether an edge with the given ID exists.

        :param edge_id: The integer ID to check.
        :return: True if the edge exists, False otherwise.
        """
        return edge_id in self._edges

    def neighbors(self, v: Hashable) -> list[Hashable]:
        """Return a list of distinct vertices adjacent to v.

        :param v: A vertex.
        :return: List of adjacent vertices (no duplicates).
        :raises KeyError: If vertex v does not exist.
        """
        if v not in self._vertices:
            raise KeyError(f"Vertex {v} not in graph")
        seen: set[Hashable] = set()
        result: list[Hashable] = []
        for eid in self._adj[v]:
            nbr: Hashable = self._edges[eid].other(v)
            if nbr not in seen:
                seen.add(nbr)
                result.append(nbr)
        return result

    def incident_edges(self, v: Hashable) -> list[Edge]:
        """Return all edges incident to vertex v.

        :param v: A vertex.
        :return: List of Edge objects incident to v.
        :raises KeyError: If vertex v does not exist.
        """
        if v not in self._vertices:
            raise KeyError(f"Vertex {v} not in graph")
        return [self._edges[eid] for eid in self._adj[v]]

    def edges_between(self, u: Hashable, v: Hashable) -> list[Edge]:
        """Return all edges between vertices u and v.

        :param u: One vertex.
        :param v: The other vertex.
        :return: List of Edge objects between u and v.
        """
        return [
            self._edges[eid]
            for eid in self._adj.get(u, [])
            if self._edges[eid].other(u) == v
        ]

    def degree(self, v: Hashable) -> int:
        """Return the degree of vertex v (number of incident edges).

        :param v: A vertex.
        :return: The number of edges incident to v.
        :raises KeyError: If vertex v does not exist.
        """
        if v not in self._vertices:
            raise KeyError(f"Vertex {v} not in graph")
        return len(self._adj[v])

    def num_vertices(self) -> int:
        """Return the number of vertices in the graph.

        :return: Vertex count.
        """
        return len(self._vertices)

    def num_edges(self) -> int:
        """Return the number of edges in the graph.

        :return: Edge count.
        """
        return len(self._edges)

    def copy(self) -> MultiGraph:
        """Return a shallow copy of this graph with independent structure.

        Vertices and edge IDs are preserved. Edge objects are copied.

        :return: A new MultiGraph with the same structure.
        """
        g: MultiGraph = MultiGraph()
        g._next_edge_id = self._next_edge_id
        for v in self._vertices:
            g._vertices.add(v)
            g._adj[v] = list(self._adj[v])
        for eid, e in self._edges.items():
            g._edges[eid] = Edge(
                id=e.id, u=e.u, v=e.v, virtual=e.virtual
            )
        return g

    def adj_edge_ids(self, v: Hashable) -> list[int]:
        """Return the list of edge IDs in the adjacency list of v.

        The order reflects the current adjacency list ordering, which
        may be modified by the palm tree construction for algorithmic
        correctness.

        :param v: A vertex.
        :return: List of edge IDs adjacent to v, in adjacency list order.
        :raises KeyError: If vertex v does not exist.
        """
        if v not in self._vertices:
            raise KeyError(f"Vertex {v} not in graph")
        return list(self._adj[v])

    def set_adj_order(self, v: Hashable, edge_ids: list[int]) -> None:
        """Set the adjacency list order for vertex v.

        Used by the palm tree algorithm to reorder edges for correct
        DFS traversal order per Gutwenger-Mutzel Section 4.2.

        :param v: A vertex.
        :param edge_ids: The ordered list of edge IDs for v's adjacency.
        :return: None
        :raises KeyError: If vertex v does not exist.
        """
        if v not in self._vertices:
            raise KeyError(f"Vertex {v} not in graph")
        self._adj[v] = list(edge_ids)
