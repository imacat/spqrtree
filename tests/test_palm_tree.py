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
"""Tests for the palm tree construction (_palm_tree.py).

All DFS results are based on insertion-order adjacency traversal and
DFS from vertex 1.  Edge insertion order is specified in each test.
"""
import unittest
from collections.abc import Hashable

from spqrtree._graph import Edge, MultiGraph
from spqrtree._palm_tree import PalmTree, build_palm_tree, phi_key


def _make_k3() -> tuple[MultiGraph, list[int]]:
    """Build the triangle graph K3 (vertices 1,2,3; edges 1-2, 2-3, 1-3).

    :return: A tuple (graph, edge_ids) where edge_ids is
             [e0.id, e1.id, e2.id].
    """
    g: MultiGraph = MultiGraph()
    for v in [1, 2, 3]:
        g.add_vertex(v)
    e0: Edge = g.add_edge(1, 2)
    e1: Edge = g.add_edge(2, 3)
    e2: Edge = g.add_edge(1, 3)
    return g, [e0.id, e1.id, e2.id]


def _make_p3() -> tuple[MultiGraph, list[int]]:
    """Build the path graph P3 (vertices 1,2,3; edges 1-2, 2-3).

    :return: A tuple (graph, edge_ids) where edge_ids is [e0.id, e1.id].
    """
    g: MultiGraph = MultiGraph()
    for v in [1, 2, 3]:
        g.add_vertex(v)
    e0: Edge = g.add_edge(1, 2)
    e1: Edge = g.add_edge(2, 3)
    return g, [e0.id, e1.id]


def _make_c4() -> tuple[MultiGraph, list[int]]:
    """Build the 4-cycle C4 (vertices 1,2,3,4; edges 1-2,2-3,3-4,4-1).

    :return: A tuple (graph, edge_ids) in insertion order.
    """
    g: MultiGraph = MultiGraph()
    for v in [1, 2, 3, 4]:
        g.add_vertex(v)
    e0: Edge = g.add_edge(1, 2)
    e1: Edge = g.add_edge(2, 3)
    e2: Edge = g.add_edge(3, 4)
    e3: Edge = g.add_edge(4, 1)
    return g, [e0.id, e1.id, e2.id, e3.id]


class TestPalmTreeType(unittest.TestCase):
    """Tests that build_palm_tree returns a PalmTree instance."""

    def test_returns_palm_tree(self) -> None:
        """Test that build_palm_tree returns a PalmTree object."""
        g: MultiGraph
        g, _ = _make_k3()
        pt: PalmTree = build_palm_tree(g, 1)
        self.assertIsInstance(pt, PalmTree)


class TestPalmTreePath(unittest.TestCase):
    """Tests for palm tree on a path graph P3 (no back edges)."""

    def setUp(self) -> None:
        """Build palm tree for P3 starting at vertex 1."""
        g: MultiGraph
        g, eids = _make_p3()
        self.eids: list[int] = eids
        """The edge IDs of the P3 graph."""
        self.pt: PalmTree = build_palm_tree(g, 1)
        """The palm tree for the graph."""

    def test_dfs_num_root(self) -> None:
        """Test that the start vertex has DFS number 1."""
        self.assertEqual(self.pt.dfs_num[1], 1)

    def test_dfs_num_order(self) -> None:
        """Test that DFS numbers are assigned 1, 2, 3 in traversal order."""
        nums: list[int] = sorted(self.pt.dfs_num.values())
        self.assertEqual(nums, [1, 2, 3])

    def test_tree_edges_count(self) -> None:
        """Test that there are n-1 = 2 tree edges."""
        self.assertEqual(len(self.pt.tree_edges), 2)

    def test_no_fronds(self) -> None:
        """Test that there are no fronds on a tree path."""
        self.assertEqual(len(self.pt.fronds), 0)

    def test_all_edges_are_tree_edges(self) -> None:
        """Test that both edges are classified as tree edges."""
        e0, e1 = self.eids
        self.assertIn(e0, self.pt.tree_edges)
        self.assertIn(e1, self.pt.tree_edges)

    def test_parent_root(self) -> None:
        """Test that the root has no parent (None)."""
        self.assertIsNone(self.pt.parent.get(1))

    def test_nd_values(self) -> None:
        """Test that ND values are correct for P3."""
        # DFS from 1: 1→2→3 (since 2 is first in adj[1])
        self.assertEqual(self.pt.nd[1], 3)

    def test_nd_leaf(self) -> None:
        """Test that a leaf vertex has ND = 1."""
        # Vertex 3 is a leaf in P3
        self.assertEqual(self.pt.nd[3], 1)

    def test_lowpt1_values(self) -> None:
        """Test lowpt1 values for P3 (all vertices reach only themselves)."""
        for v in [1, 2, 3]:
            self.assertLessEqual(
                self.pt.lowpt1[v], self.pt.dfs_num[v]
            )

    def test_lowpt1_no_fronds(self) -> None:
        """Test that lowpt1[v] == dfs_num[v] when no fronds exist."""
        for v in [1, 2, 3]:
            self.assertEqual(self.pt.lowpt1[v], self.pt.dfs_num[v])


class TestPalmTreeTriangle(unittest.TestCase):
    """Tests for palm tree on the triangle graph K3."""

    def setUp(self) -> None:
        """Build palm tree for K3 starting at vertex 1."""
        g: MultiGraph
        g, eids = _make_k3()
        self.eids: list[int] = eids
        """The edge IDs of the K3 graph."""
        self.pt: PalmTree = build_palm_tree(g, 1)
        """The palm tree for the graph."""

    def test_dfs_num_root(self) -> None:
        """Test that vertex 1 has DFS number 1."""
        self.assertEqual(self.pt.dfs_num[1], 1)

    def test_dfs_num_all_assigned(self) -> None:
        """Test that all vertices get a DFS number."""
        self.assertEqual(set(self.pt.dfs_num.keys()), {1, 2, 3})

    def test_tree_edge_count(self) -> None:
        """Test that there are n-1 = 2 tree edges."""
        self.assertEqual(len(self.pt.tree_edges), 2)

    def test_frond_count(self) -> None:
        """Test that there is exactly 1 frond."""
        self.assertEqual(len(self.pt.fronds), 1)

    def test_frond_is_back_edge(self) -> None:
        """Test that the frond e2 (1-3) is classified as a frond."""
        # e2 = edges[2] = the third edge added (1-3)
        e0, e1, e2 = self.eids
        self.assertIn(e2, self.pt.fronds)
        self.assertIn(e0, self.pt.tree_edges)
        self.assertIn(e1, self.pt.tree_edges)

    def test_lowpt1_all_reach_root(self) -> None:
        """Test that all vertices have lowpt1 = 1 via frond."""
        for v in [1, 2, 3]:
            self.assertEqual(self.pt.lowpt1[v], 1)

    def test_nd_root(self) -> None:
        """Test that the root has nd = 3."""
        self.assertEqual(self.pt.nd[1], 3)

    def test_nd_leaf(self) -> None:
        """Test that the DFS leaf (vertex 3) has nd = 1."""
        # Vertex 3 is visited last in K3 with DFS from 1
        leaf: Hashable = next(
            v for v, n in self.pt.nd.items() if n == 1
        )
        self.assertEqual(self.pt.nd[leaf], 1)

    def test_first_child_of_root(self) -> None:
        """Test that root vertex 1 has a first child."""
        self.assertIsNotNone(self.pt.first_child.get(1))

    def test_first_child_of_leaf(self) -> None:
        """Test that the DFS leaf has no first child."""
        leaf: Hashable = next(
            v for v, n in self.pt.nd.items() if n == 1
        )
        self.assertIsNone(self.pt.first_child.get(leaf))

    def test_lowpt1_le_dfs_num(self) -> None:
        """Test that lowpt1[v] <= dfs_num[v] for all v."""
        for v in [1, 2, 3]:
            self.assertLessEqual(self.pt.lowpt1[v], self.pt.dfs_num[v])

    def test_lowpt1_le_lowpt2(self) -> None:
        """Test that lowpt1[v] <= lowpt2[v] for all v."""
        for v in [1, 2, 3]:
            self.assertLessEqual(self.pt.lowpt1[v], self.pt.lowpt2[v])


class TestPalmTreeC4(unittest.TestCase):
    """Tests for palm tree on the 4-cycle C4."""

    def setUp(self) -> None:
        """Build palm tree for C4 starting at vertex 1."""
        g: MultiGraph
        g, eids = _make_c4()
        self.eids: list[int] = eids
        """The edge IDs of the C4 graph."""
        self.pt: PalmTree = build_palm_tree(g, 1)
        """The palm tree for the graph."""

    def test_dfs_num_root(self) -> None:
        """Test that vertex 1 has DFS number 1."""
        self.assertEqual(self.pt.dfs_num[1], 1)

    def test_dfs_num_all_assigned(self) -> None:
        """Test that all 4 vertices get DFS numbers."""
        self.assertEqual(set(self.pt.dfs_num.keys()), {1, 2, 3, 4})

    def test_tree_edge_count(self) -> None:
        """Test that there are n-1 = 3 tree edges."""
        self.assertEqual(len(self.pt.tree_edges), 3)

    def test_frond_count(self) -> None:
        """Test that there is exactly 1 frond."""
        self.assertEqual(len(self.pt.fronds), 1)

    def test_frond_classification(self) -> None:
        """Test that e3 (4-1) is a frond and e0,e1,e2 are tree edges."""
        e0, e1, e2, e3 = self.eids
        self.assertIn(e0, self.pt.tree_edges)
        self.assertIn(e1, self.pt.tree_edges)
        self.assertIn(e2, self.pt.tree_edges)
        self.assertIn(e3, self.pt.fronds)

    def test_nd_root(self) -> None:
        """Test that root vertex 1 has nd = 4."""
        self.assertEqual(self.pt.nd[1], 4)

    def test_nd_leaf(self) -> None:
        """Test that the DFS leaf has nd = 1."""
        leaf: Hashable = next(
            v for v, n in self.pt.nd.items() if n == 1
        )
        self.assertEqual(self.pt.nd[leaf], 1)

    def test_lowpt1_all_reach_root(self) -> None:
        """Test that all vertices have lowpt1 = 1 due to the frond."""
        for v in [1, 2, 3, 4]:
            self.assertEqual(self.pt.lowpt1[v], 1)

    def test_lowpt2_intermediate(self) -> None:
        """Test that lowpt2 values are consistent (lowpt1 <= lowpt2)."""
        for v in [1, 2, 3, 4]:
            self.assertLessEqual(self.pt.lowpt1[v], self.pt.lowpt2[v])

    def test_parent_structure(self) -> None:
        """Test that the parent structure is a valid tree rooted at 1."""
        # Root has no parent
        self.assertIsNone(self.pt.parent.get(1))
        # All other vertices have a parent
        for v in [2, 3, 4]:
            self.assertIsNotNone(self.pt.parent.get(v))


class TestPalmTreeLowptSpecific(unittest.TestCase):
    """Tests for specific lowpt1/lowpt2 values on P3."""

    def setUp(self) -> None:
        """Build palm tree for P3 (path 1-2-3) starting at 1."""
        g: MultiGraph
        g, _ = _make_p3()
        self.pt: PalmTree = build_palm_tree(g, 1)
        """The palm tree for the graph."""

    def test_exact_dfs_nums(self) -> None:
        """Test exact DFS numbers for the path graph."""
        self.assertEqual(self.pt.dfs_num[1], 1)
        self.assertEqual(self.pt.dfs_num[2], 2)
        self.assertEqual(self.pt.dfs_num[3], 3)

    def test_exact_lowpt1(self) -> None:
        """Test exact lowpt1 values for P3 (no back edges)."""
        # Without fronds, lowpt1[v] = dfs_num[v]
        self.assertEqual(self.pt.lowpt1[3], 3)
        self.assertEqual(self.pt.lowpt1[2], 2)
        self.assertEqual(self.pt.lowpt1[1], 1)

    def test_exact_nd(self) -> None:
        """Test exact nd values for P3."""
        self.assertEqual(self.pt.nd[3], 1)
        self.assertEqual(self.pt.nd[2], 2)
        self.assertEqual(self.pt.nd[1], 3)

    def test_exact_parent(self) -> None:
        """Test exact parent mapping for P3."""
        self.assertIsNone(self.pt.parent.get(1))
        self.assertEqual(self.pt.parent[2], 1)
        self.assertEqual(self.pt.parent[3], 2)

    def test_exact_first_child(self) -> None:
        """Test exact first_child mapping for P3."""
        self.assertEqual(self.pt.first_child[1], 2)
        self.assertEqual(self.pt.first_child[2], 3)
        self.assertIsNone(self.pt.first_child.get(3))


class TestPhiKeyP3(unittest.TestCase):
    """Tests for phi_key correctness on the path graph P3.

    P3: vertices 1,2,3; edges 1-2 (tree), 2-3 (tree).
    DFS from 1: dfs_num = {1:1, 2:2, 3:3}.
    No fronds; lowpt1[v] = dfs_num[v] for all v.
    """

    def setUp(self) -> None:
        """Build palm tree for P3 and gather data for phi_key tests."""
        g: MultiGraph
        g, eids = _make_p3()
        self.g: MultiGraph = g
        """The P3 graph under test."""
        self.eids: list[int] = eids
        """The edge IDs of the P3 graph."""
        self.pt: PalmTree = build_palm_tree(g, 1)
        """The palm tree for the graph."""

    def test_tree_edge_1_2_case3_formula(self) -> None:
        """Test phi_key for tree edge 1-2 uses case 3 formula.

        For tree edge v=1, w=2:
          lowpt1[2]=2, lowpt2[2]=INF, dfs_num[1]=1.
          lowpt2[2] >= dfs_num[1] -> case 3 -> phi = 3*lowpt1[2]+2 = 8.
        """
        e0: int = self.eids[0]
        key: int = phi_key(
            v=1, eid=e0, pt=self.pt, graph=self.g,
        )
        # lowpt1[2]=2, case 3: 3*2+2 = 8
        self.assertEqual(key, 3 * 2 + 2)

    def test_tree_edge_2_3_case3_formula(self) -> None:
        """Test phi_key for tree edge 2-3 uses case 3 formula.

        For tree edge v=2, w=3:
          lowpt1[3]=3, lowpt2[3]=INF, dfs_num[2]=2.
          lowpt2[3] >= dfs_num[2] -> case 3 -> phi = 3*lowpt1[3]+2 = 11.
        """
        e1: int = self.eids[1]
        key: int = phi_key(
            v=2, eid=e1, pt=self.pt, graph=self.g,
        )
        # lowpt1[3]=3, case 3: 3*3+2 = 11
        self.assertEqual(key, 3 * 3 + 2)

    def test_tree_edge_case3_greater_than_case1(self) -> None:
        """Test that case-3 phi > case-1 phi for ordering.

        Case 1: phi = 3*lowpt1[w]
        Case 3: phi = 3*lowpt1[w]+2
        Case 3 must be strictly greater than case 1 for same lowpt1.
        """
        e0: int = self.eids[0]
        key_v1: int = phi_key(
            v=1, eid=e0, pt=self.pt, graph=self.g,
        )
        # Case 3 value (8) > case 1 value (6) for lowpt1[2]=2
        self.assertGreater(key_v1, 3 * 2)


class TestPhiKeyK3Frond(unittest.TestCase):
    """Tests for phi_key correctness on the triangle K3.

    K3: vertices 1,2,3; edges 1-2 (tree), 2-3 (tree), 1-3 (frond).
    DFS from 1: dfs_num = {1:1, 2:2, 3:3}.
    Frond: 1-3 (from vertex 3 to ancestor 1).
    """

    def setUp(self) -> None:
        """Build palm tree for K3 and gather data for phi_key tests."""
        g: MultiGraph
        g, eids = _make_k3()
        self.g: MultiGraph = g
        """The K3 graph under test."""
        self.eids: list[int] = eids
        """The edge IDs of the K3 graph."""
        self.pt: PalmTree = build_palm_tree(g, 1)
        """The palm tree for the graph."""

    def test_frond_phi_uses_w_dfs_num(self) -> None:
        """Test phi_key for frond e2 (1-3) from v=3 uses dfs_num[w=1].

        Frond from v=3 to ancestor w=1: dfs_num[w]=1.
        phi = 3 * dfs_num[w] + 1 = 3 * 1 + 1 = 4.
        """
        e2: int = self.eids[2]  # edge 1-3 (frond)
        # The frond is traversed from vertex 3 toward ancestor 1.
        # We need to identify which end is the frond source.
        # In the palm tree, frond is from v=3 (dfs_num=3) to w=1
        # (dfs_num=1).
        # phi_key called with v=3, w=1 (ancestor), eid=e2.
        key: int = phi_key(
            v=3, eid=e2, pt=self.pt, graph=self.g,
        )
        # Correct formula: 3 * dfs_num[w=1] + 1 = 3*1+1 = 4
        self.assertEqual(key, 3 * 1 + 1)

    def test_frond_phi_different_from_v_formula(self) -> None:
        """Test that frond phi uses w (not v) DFS number.

        The buggy formula uses dfs_num[v] (= 3) giving 3*3+2 = 11.
        The correct formula uses dfs_num[w] (= 1) giving 3*1+1 = 4.
        These must differ.
        """
        e2: int = self.eids[2]
        key: int = phi_key(
            v=3, eid=e2, pt=self.pt, graph=self.g,
        )
        # The buggy value would be 3*dfs_num[v=3]+2 = 3*3+2 = 11.
        # The correct value is 3*dfs_num[w=1]+1 = 4.
        self.assertNotEqual(key, 11)
        self.assertEqual(key, 4)

    def test_frond_phi_less_than_tree_edge_case3(self) -> None:
        """Test ordering: frond phi < tree-edge case-3 phi.

        The frond phi (4) should be less than a case-3 tree-edge phi
        with the same lowpt1 (3*1+2=5), ensuring correct DFS order.
        """
        e2: int = self.eids[2]
        frond_key: int = phi_key(
            v=3, eid=e2, pt=self.pt, graph=self.g,
        )
        # Frond phi = 3*1+1=4; case-3 tree-edge phi at lowpt1=1 = 3*1+2=5
        self.assertLess(frond_key, 3 * 1 + 2)

    def test_tree_edge_case1_condition(self) -> None:
        """Test phi_key for tree edge where lowpt2[w] < dfs_num[v].

        For K3, after sorting: tree edge v=1, w=2:
          lowpt1[2]=1, lowpt2[2]=INF, dfs_num[1]=1.
          lowpt2[2]=INF >= dfs_num[1]=1 -> case 3 -> phi = 3*1+2 = 5.
        """
        e0: int = self.eids[0]  # edge 1-2 (tree edge)
        key: int = phi_key(
            v=1, eid=e0, pt=self.pt, graph=self.g,
        )
        # lowpt1[2]=1, lowpt2[2]=INF >= dfs_num[1]=1 -> case 3:
        # phi = 3*1+2 = 5
        self.assertEqual(key, 3 * 1 + 2)
