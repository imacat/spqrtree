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
"""Tests for the MultiGraph data structure (_graph.py)."""
import unittest
from collections.abc import Hashable

from spqrtree._graph import Edge, MultiGraph


class TestEdge(unittest.TestCase):
    """Tests for the Edge dataclass."""

    def test_edge_creation(self) -> None:
        """Test basic edge creation with required attributes."""
        e: Edge = Edge(id=0, u=1, v=2)
        self.assertEqual(e.id, 0)
        self.assertEqual(e.u, 1)
        self.assertEqual(e.v, 2)
        self.assertFalse(e.virtual)

    def test_edge_virtual(self) -> None:
        """Test creating a virtual edge."""
        e: Edge = Edge(id=1, u=3, v=4, virtual=True)
        self.assertTrue(e.virtual)

    def test_edge_endpoints(self) -> None:
        """Test that endpoints method returns both endpoints."""
        e: Edge = Edge(id=0, u=1, v=2)
        self.assertEqual(e.endpoints(), (1, 2))

    def test_edge_other(self) -> None:
        """Test that other() returns the opposite endpoint."""
        e: Edge = Edge(id=0, u=1, v=2)
        self.assertEqual(e.other(1), 2)
        self.assertEqual(e.other(2), 1)


class TestMultiGraphVertices(unittest.TestCase):
    """Tests for vertex operations on MultiGraph."""

    def setUp(self) -> None:
        """Set up a fresh MultiGraph for each test."""
        self.g: MultiGraph = MultiGraph()
        """The graph under test."""

    def test_add_vertex(self) -> None:
        """Test adding a single vertex."""
        self.g.add_vertex(1)
        self.assertIn(1, self.g.vertices)

    def test_add_multiple_vertices(self) -> None:
        """Test adding multiple vertices."""
        for v in [1, 2, 3, 4]:
            self.g.add_vertex(v)
        self.assertEqual(set(self.g.vertices), {1, 2, 3, 4})

    def test_add_duplicate_vertex(self) -> None:
        """Test that adding a duplicate vertex is a no-op."""
        self.g.add_vertex(1)
        self.g.add_vertex(1)
        self.assertEqual(len(self.g.vertices), 1)

    def test_remove_vertex(self) -> None:
        """Test removing a vertex also removes its edges."""
        self.g.add_vertex(1)
        self.g.add_vertex(2)
        self.g.add_edge(1, 2)
        self.g.remove_vertex(1)
        self.assertNotIn(1, self.g.vertices)
        self.assertEqual(len(self.g.edges), 0)

    def test_vertex_count(self) -> None:
        """Test vertex count after additions."""
        for v in range(5):
            self.g.add_vertex(v)
        self.assertEqual(self.g.num_vertices(), 5)


class TestMultiGraphEdges(unittest.TestCase):
    """Tests for edge operations on MultiGraph."""

    def setUp(self) -> None:
        """Set up a fresh MultiGraph with two vertices."""
        self.g: MultiGraph = MultiGraph()
        """The graph under test."""
        self.g.add_vertex(1)
        self.g.add_vertex(2)
        self.g.add_vertex(3)

    def test_add_edge(self) -> None:
        """Test adding an edge between two vertices."""
        e: Edge = self.g.add_edge(1, 2)
        self.assertIn(e.id, {e2.id for e2 in self.g.edges})
        self.assertEqual(e.u, 1)
        self.assertEqual(e.v, 2)

    def test_add_edge_returns_edge(self) -> None:
        """Test that add_edge returns an Edge object."""
        e: Edge = self.g.add_edge(1, 2)
        self.assertIsInstance(e, Edge)

    def test_add_parallel_edges(self) -> None:
        """Test adding parallel edges between the same pair."""
        e1: Edge = self.g.add_edge(1, 2)
        e2: Edge = self.g.add_edge(1, 2)
        self.assertNotEqual(e1.id, e2.id)
        self.assertEqual(len(self.g.edges_between(1, 2)), 2)

    def test_remove_edge(self) -> None:
        """Test removing a specific edge by ID."""
        e: Edge = self.g.add_edge(1, 2)
        self.g.remove_edge(e.id)
        self.assertEqual(
            len(self.g.edges_between(1, 2)), 0)

    def test_remove_one_parallel_edge(self) -> None:
        """Test removing one of several parallel edges."""
        e1: Edge = self.g.add_edge(1, 2)
        e2: Edge = self.g.add_edge(1, 2)
        self.g.remove_edge(e1.id)
        remaining: list[Edge] = (
            self.g.edges_between(1, 2))
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].id, e2.id)

    def test_edge_count(self) -> None:
        """Test total edge count."""
        self.g.add_edge(1, 2)
        self.g.add_edge(2, 3)
        self.g.add_edge(1, 2)
        self.assertEqual(self.g.num_edges(), 3)

    def test_add_virtual_edge(self) -> None:
        """Test adding a virtual edge."""
        e: Edge = self.g.add_edge(1, 2, virtual=True)
        self.assertTrue(e.virtual)

    def test_edges_property(self) -> None:
        """Test that edges property returns all edges."""
        e1: Edge = self.g.add_edge(1, 2)
        e2: Edge = self.g.add_edge(2, 3)
        ids: set[int] = {e.id for e in self.g.edges}
        self.assertIn(e1.id, ids)
        self.assertIn(e2.id, ids)


class TestMultiGraphNeighbors(unittest.TestCase):
    """Tests for neighbor/adjacency operations on MultiGraph."""

    def setUp(self) -> None:
        """Set up a triangle graph (K3)."""
        self.g: MultiGraph = MultiGraph()
        """The graph under test."""
        for v in [1, 2, 3]:
            self.g.add_vertex(v)
        self.g.add_edge(1, 2)
        self.g.add_edge(2, 3)
        self.g.add_edge(1, 3)

    def test_neighbors(self) -> None:
        """Test neighbors returns all adjacent vertices."""
        nbrs: list[Hashable] = self.g.neighbors(1)
        self.assertEqual(set(nbrs), {2, 3})

    def test_neighbors_with_parallel_edges(self) -> None:
        """Test neighbors returns unique vertices with parallel edges."""
        self.g.add_edge(1, 2)
        nbrs: list[Hashable] = self.g.neighbors(1)
        self.assertEqual(set(nbrs), {2, 3})

    def test_incident_edges(self) -> None:
        """Test incident_edges returns edges incident to a vertex."""
        edges: list[Edge] = self.g.incident_edges(1)
        self.assertEqual(len(edges), 2)

    def test_edges_between(self) -> None:
        """Test edges_between returns edges between two vertices."""
        edges: list[Edge] = self.g.edges_between(1, 2)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].u, 1)
        self.assertEqual(edges[0].v, 2)

    def test_degree(self) -> None:
        """Test degree counts incident edges (with multiplicity)."""
        self.assertEqual(self.g.degree(1), 2)
        self.g.add_edge(1, 2)
        self.assertEqual(self.g.degree(1), 3)


class TestMultiGraphCopy(unittest.TestCase):
    """Tests for copying operations on MultiGraph."""

    def test_copy_is_independent(self) -> None:
        """Test that a copy is independent from the original."""
        g: MultiGraph = MultiGraph()
        g.add_vertex(1)
        g.add_vertex(2)
        g.add_edge(1, 2)
        g2: MultiGraph = g.copy()
        g2.add_vertex(3)
        self.assertNotIn(3, g.vertices)

    def test_copy_has_same_structure(self) -> None:
        """Test that a copy has the same vertices and edges."""
        g: MultiGraph = MultiGraph()
        for v in [1, 2, 3]:
            g.add_vertex(v)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g2: MultiGraph = g.copy()
        self.assertEqual(set(g2.vertices), {1, 2, 3})
        self.assertEqual(g2.num_edges(), 2)
