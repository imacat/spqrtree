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
"""Tests for the SPQR-tree (_spqr.py).

Tests cover: triangle K3 (S-node), K4 (R-node), C4 (S-node),
two parallel edges (Q-node), and three parallel edges (P-node).
"""
import time
import unittest
from collections import deque
from collections.abc import Hashable

from spqrtree._graph import Edge, MultiGraph
from spqrtree._spqr import NodeType, SPQRNode, build_spqr_tree


def _make_k3() -> MultiGraph:
    """Build the triangle graph K3 (vertices 1,2,3).

    :return: A MultiGraph representing K3.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(1, 3)
    return g


def _make_c4() -> MultiGraph:
    """Build the 4-cycle C4 (vertices 1,2,3,4).

    :return: A MultiGraph representing C4.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 1)
    return g


def _make_k4() -> MultiGraph:
    """Build the complete graph K4 (6 edges among vertices 1,2,3,4).

    :return: A MultiGraph representing K4.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(1, 4)
    g.add_edge(2, 3)
    g.add_edge(2, 4)
    g.add_edge(3, 4)
    return g


def _make_two_parallel() -> MultiGraph:
    """Build a graph with 2 parallel edges between vertices 1 and 2.

    :return: A MultiGraph with two parallel edges.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(1, 2)
    g.add_edge(1, 2)
    return g


def _make_three_parallel() -> MultiGraph:
    """Build a graph with 3 parallel edges between vertices 1 and 2.

    :return: A MultiGraph with three parallel edges.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(1, 2)
    g.add_edge(1, 2)
    g.add_edge(1, 2)
    return g


def _collect_all_nodes(root: SPQRNode) -> list[SPQRNode]:
    """Collect all nodes in the SPQR-tree by BFS.

    :param root: The root SPQRNode.
    :return: List of all SPQRNode objects in BFS order.
    """
    result: list[SPQRNode] = []
    bfs_queue: deque[SPQRNode] = deque([root])
    while bfs_queue:
        node: SPQRNode = bfs_queue.popleft()
        result.append(node)
        for child in node.children:
            bfs_queue.append(child)
    return result


class TestSPQRNodeStructure(unittest.TestCase):
    """Tests for SPQRNode structure and attributes."""

    def test_spqrnode_has_type(self) -> None:
        """Test that SPQRNode has a type attribute."""
        root: SPQRNode = build_spqr_tree(_make_k3())
        self.assertIsInstance(root.type, NodeType)

    def test_spqrnode_has_skeleton(self) -> None:
        """Test that SPQRNode has a skeleton graph."""
        root: SPQRNode = build_spqr_tree(_make_k3())
        self.assertIsInstance(root.skeleton, MultiGraph)

    def test_spqrnode_has_poles(self) -> None:
        """Test that SPQRNode has a poles attribute."""
        root: SPQRNode = build_spqr_tree(_make_k3())
        self.assertIsInstance(root.poles, tuple)
        self.assertEqual(len(root.poles), 2)

    def test_spqrnode_has_children(self) -> None:
        """Test that SPQRNode has a children list."""
        root: SPQRNode = build_spqr_tree(_make_k3())
        self.assertIsInstance(root.children, list)

    def test_spqrnode_parent_is_none_for_root(self) -> None:
        """Test that the root SPQRNode has parent = None."""
        root: SPQRNode = build_spqr_tree(_make_k3())
        self.assertIsNone(root.parent)

    def test_children_parent_links(self) -> None:
        """Test that children have correct parent links."""
        root: SPQRNode = build_spqr_tree(_make_k3())
        for child in root.children:
            self.assertIs(child.parent, root)


class TestSPQRK3(unittest.TestCase):
    """Tests for the SPQR-tree of the triangle K3."""

    def setUp(self) -> None:
        """Build the SPQR-tree for K3."""
        self.root: SPQRNode = build_spqr_tree(_make_k3())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_returns_spqrnode(self) -> None:
        """Test that build_spqr_tree returns an SPQRNode."""
        self.assertIsInstance(self.root, SPQRNode)

    def test_root_is_s_node(self) -> None:
        """Test that K3 produces an S-node (POLYGON) as root."""
        self.assertEqual(self.root.type, NodeType.S)

    def test_node_types_are_valid(self) -> None:
        """Test that all node types are valid NodeType values."""
        for node in self.all_nodes:
            self.assertIsInstance(node.type, NodeType)
            self.assertIn(node.type, list(NodeType))


class TestSPQRK4(unittest.TestCase):
    """Tests for the SPQR-tree of the complete graph K4."""

    def setUp(self) -> None:
        """Build the SPQR-tree for K4."""
        self.root: SPQRNode = build_spqr_tree(_make_k4())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_root_is_r_node(self) -> None:
        """Test that K4 produces a single R-node."""
        self.assertEqual(self.root.type, NodeType.R)

    def test_skeleton_has_vertices(self) -> None:
        """Test that the R-node skeleton has vertices."""
        self.assertGreater(self.root.skeleton.num_vertices(), 0)


class TestSPQRC4(unittest.TestCase):
    """Tests for the SPQR-tree of the 4-cycle C4."""

    def setUp(self) -> None:
        """Build the SPQR-tree for C4."""
        self.root: SPQRNode = build_spqr_tree(_make_c4())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_root_is_s_node(self) -> None:
        """Test that C4 produces an S-node (POLYGON) as root."""
        self.assertEqual(self.root.type, NodeType.S)

    def test_node_types_are_valid(self) -> None:
        """Test that all node types are valid NodeType values."""
        for node in self.all_nodes:
            self.assertIsInstance(node.type, NodeType)


class TestSPQRTwoParallel(unittest.TestCase):
    """Tests for the SPQR-tree of two parallel edges."""

    def setUp(self) -> None:
        """Build the SPQR-tree for 2 parallel edges."""
        self.root: SPQRNode = \
            build_spqr_tree(_make_two_parallel())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_root_is_q_node(self) -> None:
        """Test that 2 parallel edges produce a single Q-node."""
        # 2 parallel edges: BOND with 2 edges -> Q-node.
        self.assertEqual(self.root.type, NodeType.Q)

    def test_no_children(self) -> None:
        """Test that a Q-node has no children."""
        self.assertEqual(len(self.root.children), 0)


class TestSPQRThreeParallel(unittest.TestCase):
    """Tests for the SPQR-tree of three parallel edges."""

    def setUp(self) -> None:
        """Build the SPQR-tree for 3 parallel edges."""
        self.root: SPQRNode = \
            build_spqr_tree(_make_three_parallel())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_root_is_p_node(self) -> None:
        """Test that 3 parallel edges produce a single P-node."""
        self.assertEqual(self.root.type, NodeType.P)

    def test_node_types_are_valid(self) -> None:
        """Test that all node types are valid NodeType values."""
        for node in self.all_nodes:
            self.assertIsInstance(node.type, NodeType)


class TestSPQRInvariants(unittest.TestCase):
    """Tests for global SPQR-tree invariants across all graphs."""

    def _check_parent_links(self, root: SPQRNode) -> None:
        """Check that parent-child links are consistent.

        :param root: The SPQR-tree root.
        :return: None
        """
        for node in _collect_all_nodes(root):
            for child in node.children:
                self.assertIs(
                    child.parent,
                    node,
                    f"Child {child.type} has wrong parent",
                )

    def _check_skeleton_edges(self, root: SPQRNode) -> None:
        """Check that each node's skeleton has at least 1 edge.

        :param root: The SPQR-tree root.
        :return: None
        """
        for node in _collect_all_nodes(root):
            self.assertGreater(
                node.skeleton.num_edges(),
                0,
                f"Node {node.type} has empty skeleton",
            )

    def test_k3_parent_links(self) -> None:
        """Test parent link invariant for K3."""
        self._check_parent_links(build_spqr_tree(_make_k3()))

    def test_c4_parent_links(self) -> None:
        """Test parent link invariant for C4."""
        self._check_parent_links(build_spqr_tree(_make_c4()))

    def test_k4_parent_links(self) -> None:
        """Test parent link invariant for K4."""
        self._check_parent_links(build_spqr_tree(_make_k4()))

    def test_two_parallel_parent_links(self) -> None:
        """Test parent link invariant for 2 parallel edges."""
        self._check_parent_links(build_spqr_tree(_make_two_parallel()))

    def test_three_parallel_parent_links(self) -> None:
        """Test parent link invariant for 3 parallel edges."""
        self._check_parent_links(
            build_spqr_tree(_make_three_parallel())
        )

    def test_k3_skeleton_edges(self) -> None:
        """Test skeleton edge invariant for K3."""
        self._check_skeleton_edges(build_spqr_tree(_make_k3()))

    def test_c4_skeleton_edges(self) -> None:
        """Test skeleton edge invariant for C4."""
        self._check_skeleton_edges(build_spqr_tree(_make_c4()))

    def test_k4_skeleton_edges(self) -> None:
        """Test skeleton edge invariant for K4."""
        self._check_skeleton_edges(build_spqr_tree(_make_k4()))

    def test_two_parallel_skeleton_edges(self) -> None:
        """Test skeleton edge invariant for 2 parallel edges."""
        self._check_skeleton_edges(
            build_spqr_tree(_make_two_parallel())
        )

    def test_three_parallel_skeleton_edges(self) -> None:
        """Test skeleton edge invariant for 3 parallel edges."""
        self._check_skeleton_edges(
            build_spqr_tree(_make_three_parallel())
        )


def _make_diamond() -> MultiGraph:
    """Build the diamond graph (K4 minus one edge).

    Vertices 1,2,3,4; edges: 1-2, 1-3, 2-3, 2-4, 3-4.

    :return: A MultiGraph representing the diamond graph.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    g.add_edge(2, 4)
    g.add_edge(3, 4)
    return g


def _make_theta() -> MultiGraph:
    """Build the theta graph: two vertices connected by 3 paths.

    Vertices 1-5; edges: 1-3, 3-2, 1-4, 4-2, 1-5, 5-2.

    :return: A MultiGraph representing the theta graph.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(1, 3)
    g.add_edge(3, 2)
    g.add_edge(1, 4)
    g.add_edge(4, 2)
    g.add_edge(1, 5)
    g.add_edge(5, 2)
    return g


def _make_prism() -> MultiGraph:
    """Build the triangular prism graph.

    Two triangles connected by 3 edges. This graph is 3-connected.

    :return: A MultiGraph representing the triangular prism.
    """
    g: MultiGraph = MultiGraph()
    # Top triangle
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(1, 3)
    # Bottom triangle
    g.add_edge(4, 5)
    g.add_edge(5, 6)
    g.add_edge(4, 6)
    # Connectors
    g.add_edge(1, 4)
    g.add_edge(2, 5)
    g.add_edge(3, 6)
    return g


class TestSPQRDiamond(unittest.TestCase):
    """Tests for the SPQR-tree of the diamond graph."""

    def setUp(self) -> None:
        """Build the SPQR-tree for the diamond graph."""
        self.root: SPQRNode = \
            build_spqr_tree(_make_diamond())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_at_least_two_nodes(self) -> None:
        """Test that diamond produces at least 2 SPQR-tree nodes."""
        self.assertGreaterEqual(
            len(self.all_nodes),
            2,
            "Diamond has a separation pair, expect >=2 SPQR nodes",
        )

    def test_node_types_are_valid(self) -> None:
        """Test that all node types are valid NodeType values."""
        for node in self.all_nodes:
            self.assertIsInstance(node.type, NodeType)
            self.assertIn(node.type, list(NodeType))

    def test_no_ss_adjacency(self) -> None:
        """Test that no S-node is adjacent to another S-node."""
        _assert_no_ss_pp(self, self.root, NodeType.S)

    def test_no_pp_adjacency(self) -> None:
        """Test that no P-node is adjacent to another P-node."""
        _assert_no_ss_pp(self, self.root, NodeType.P)

    def test_parent_links(self) -> None:
        """Test that all parent-child links are consistent."""
        for node in self.all_nodes:
            for child in node.children:
                self.assertIs(child.parent, node)


class TestSPQRTheta(unittest.TestCase):
    """Tests for the SPQR-tree of the theta graph."""

    def setUp(self) -> None:
        """Build the SPQR-tree for the theta graph."""
        self.root: SPQRNode = \
            build_spqr_tree(_make_theta())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_root_is_p_node(self) -> None:
        """Test that the theta graph produces a P-node as root.

        The theta graph has a separation pair {1,2} with 3 paths
        between them, so the root should be a P-node.
        """
        self.assertEqual(
            self.root.type,
            NodeType.P,
            "Theta graph has 3 parallel paths between 1 and 2, "
            "expect P-node at root",
        )

    def test_node_types_are_valid(self) -> None:
        """Test that all node types are valid NodeType values."""
        for node in self.all_nodes:
            self.assertIsInstance(node.type, NodeType)

    def test_no_ss_adjacency(self) -> None:
        """Test that no S-node is adjacent to another S-node."""
        _assert_no_ss_pp(self, self.root, NodeType.S)

    def test_no_pp_adjacency(self) -> None:
        """Test that no P-node is adjacent to another P-node."""
        _assert_no_ss_pp(self, self.root, NodeType.P)

    def test_parent_links(self) -> None:
        """Test that all parent-child links are consistent."""
        for node in self.all_nodes:
            for child in node.children:
                self.assertIs(child.parent, node)


class TestSPQRPrism(unittest.TestCase):
    """Tests for the SPQR-tree of the triangular prism graph."""

    def setUp(self) -> None:
        """Build the SPQR-tree for the triangular prism."""
        self.root: SPQRNode = \
            build_spqr_tree(_make_prism())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_single_r_node(self) -> None:
        """Test that the prism produces a single R-node."""
        self.assertEqual(
            len(self.all_nodes),
            1,
            "Prism is 3-connected, expect single R-node",
        )
        self.assertEqual(self.root.type, NodeType.R)

    def test_no_children(self) -> None:
        """Test that the single R-node has no children."""
        self.assertEqual(len(self.root.children), 0)

    def test_skeleton_has_nine_edges(self) -> None:
        """Test that the R-node skeleton contains 9 edges."""
        self.assertEqual(self.root.skeleton.num_edges(), 9)


def _assert_no_ss_pp(
    tc: unittest.TestCase,
    root: SPQRNode,
    ntype: NodeType,
) -> None:
    """Assert that no node of *ntype* is adjacent to another of same type.

    In the SPQR-tree, S-nodes must not be adjacent to S-nodes, and
    P-nodes must not be adjacent to P-nodes.

    :param tc: The TestCase instance for assertions.
    :param root: The SPQR-tree root node.
    :param ntype: The NodeType to check (S or P).
    :return: None
    """
    for node in _collect_all_nodes(root):
        if node.type == ntype:
            for child in node.children:
                tc.assertNotEqual(
                    child.type,
                    ntype,
                    f"{ntype.value}-{ntype.value} adjacency found "
                    f"in SPQR-tree (not allowed)",
                )
            if node.parent is not None:
                tc.assertNotEqual(
                    node.parent.type,
                    ntype,
                    f"{ntype.value}-{ntype.value} adjacency found "
                    f"in SPQR-tree (not allowed)",
                )


class TestSPQRNoSSPPInvariants(unittest.TestCase):
    """Tests that no S-S or P-P adjacency occurs for all graphs."""

    def _check_tree(self, g: MultiGraph) -> None:
        """Build SPQR-tree and check S-S and P-P invariants.

        :param g: The input multigraph.
        :return: None
        """
        root: SPQRNode = build_spqr_tree(g)
        _assert_no_ss_pp(self, root, NodeType.S)
        _assert_no_ss_pp(self, root, NodeType.P)

    def test_k3_no_ss_pp(self) -> None:
        """Test no S-S or P-P adjacency for K3."""
        self._check_tree(_make_k3())

    def test_c4_no_ss_pp(self) -> None:
        """Test no S-S or P-P adjacency for C4."""
        self._check_tree(_make_c4())

    def test_k4_no_ss_pp(self) -> None:
        """Test no S-S or P-P adjacency for K4."""
        self._check_tree(_make_k4())

    def test_two_parallel_no_ss_pp(self) -> None:
        """Test no S-S or P-P adjacency for 2 parallel edges."""
        self._check_tree(_make_two_parallel())

    def test_three_parallel_no_ss_pp(self) -> None:
        """Test no S-S or P-P adjacency for 3 parallel edges."""
        self._check_tree(_make_three_parallel())

    def test_diamond_no_ss_pp(self) -> None:
        """Test no S-S or P-P adjacency for the diamond graph."""
        self._check_tree(_make_diamond())

    def test_theta_no_ss_pp(self) -> None:
        """Test no S-S or P-P adjacency for the theta graph."""
        self._check_tree(_make_theta())

    def test_prism_no_ss_pp(self) -> None:
        """Test no S-S or P-P adjacency for the triangular prism."""
        self._check_tree(_make_prism())


def _count_real_edges_in_tree(root: SPQRNode) -> int:
    """Count real (non-virtual) edges across all SPQR-tree skeletons.

    Each real edge should appear in exactly one node's skeleton.

    :param root: The SPQR-tree root.
    :return: Total count of real edges summed over all nodes.
    """
    total: int = 0
    for node in _collect_all_nodes(root):
        for e in node.skeleton.edges:
            if not e.virtual:
                total += 1
    return total


def _check_spqr_invariants(
    tc: unittest.TestCase,
    g: MultiGraph,
    root: SPQRNode,
) -> None:
    """Check all SPQR-tree invariants for a given graph.

    Verifies: parent-child links consistent, each node has >= 1 skeleton
    edge, real edge count preserved, no S-S or P-P adjacency.

    :param tc: The TestCase instance for assertions.
    :param g: The original input graph.
    :param root: The SPQR-tree root.
    :return: None
    """
    nodes: list[SPQRNode] = _collect_all_nodes(root)
    # Parent-child links.
    for node in nodes:
        for child in node.children:
            tc.assertIs(child.parent, node)
    # Each node skeleton has at least 1 edge.
    for node in nodes:
        tc.assertGreater(node.skeleton.num_edges(), 0)
    # Real edge count.
    tc.assertEqual(
        _count_real_edges_in_tree(root),
        g.num_edges(),
        "Real edge count mismatch between SPQR-tree and original",
    )
    # No S-S adjacency.
    _assert_no_ss_pp(tc, root, NodeType.S)
    # No P-P adjacency.
    _assert_no_ss_pp(tc, root, NodeType.P)


def _make_wikipedia_example() -> MultiGraph:
    """Build the Wikipedia SPQR-tree example graph.

    21 edges, 13 vertices. Used in the Wikipedia SPQR_tree article.

    :return: A MultiGraph representing the Wikipedia example.
    """
    g: MultiGraph = MultiGraph()
    edges: list[tuple[int, int]] = [
        (1, 2), (1, 4), (1, 8), (1, 12),
        (3, 4), (2, 3), (2, 13), (3, 13),
        (4, 5), (4, 7), (5, 6), (5, 8), (5, 7), (6, 7),
        (8, 11), (8, 9), (8, 12), (9, 10), (9, 11), (9, 12), (10, 12),
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g


def _make_ht_example() -> MultiGraph:
    """Build the Hopcroft-Tarjan (1973) example graph.

    23 edges, 13 vertices. Used in [HT1973].

    :return: A MultiGraph representing the HT1973 example.
    """
    g: MultiGraph = MultiGraph()
    edges: list[tuple[int, int]] = [
        (1, 2), (1, 4), (1, 8), (1, 12), (1, 13),
        (2, 3), (2, 13), (3, 4), (3, 13),
        (4, 5), (4, 7), (5, 6), (5, 7), (5, 8), (6, 7),
        (8, 9), (8, 11), (8, 12), (9, 10), (9, 11), (9, 12),
        (10, 11), (10, 12),
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g


def _make_gm_example() -> MultiGraph:
    """Build the Gutwenger-Mutzel (2001) example graph.

    28 edges, 16 vertices. Used in [GM2001].

    :return: A MultiGraph representing the GM2001 example.
    """
    g: MultiGraph = MultiGraph()
    edges: list[tuple[int, int]] = [
        (1, 2), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5),
        (4, 6), (5, 7), (5, 8), (5, 14), (6, 8), (7, 14),
        (8, 9), (8, 10), (8, 11), (8, 12), (9, 10),
        (10, 13), (10, 14), (10, 15), (10, 16),
        (11, 12), (11, 13), (12, 13),
        (14, 15), (14, 16), (15, 16),
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g


def _make_multiedge_complex() -> MultiGraph:
    """Build a complex graph with multi-edges embedded in a larger graph.

    5 vertices, 7 edges; two pairs of parallel edges (1-5 and 2-3)
    embedded in a cycle.

    :return: A MultiGraph with embedded parallel edges.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 5)
    g.add_edge(1, 5)
    g.add_edge(1, 5)
    return g


class TestSPQRWikipediaExample(unittest.TestCase):
    """Tests for the SPQR-tree of the Wikipedia example graph."""

    def setUp(self) -> None:
        """Build the SPQR-tree for the Wikipedia example."""
        self.g: MultiGraph = _make_wikipedia_example()
        """The Wikipedia example graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for the Wikipedia example."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_at_least_two_nodes(self) -> None:
        """Test that the Wikipedia example produces multiple nodes."""
        self.assertGreaterEqual(
            len(self.all_nodes),
            2,
            "Wikipedia example has separation pairs, expect >=2 nodes",
        )


class TestSPQRHTExample(unittest.TestCase):
    """Tests for the SPQR-tree of the Hopcroft-Tarjan 1973 example."""

    def setUp(self) -> None:
        """Build the SPQR-tree for the HT1973 example."""
        self.g: MultiGraph = _make_ht_example()
        """The HT1973 example graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for the HT1973 example."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_at_least_two_nodes(self) -> None:
        """Test that the HT1973 example produces multiple nodes."""
        self.assertGreaterEqual(
            len(self.all_nodes),
            2,
            "HT1973 example has separation pairs, expect >=2 nodes",
        )


class TestSPQRGMExample(unittest.TestCase):
    """Tests for the SPQR-tree of the Gutwenger-Mutzel 2001 example."""

    def setUp(self) -> None:
        """Build the SPQR-tree for the GM2001 example."""
        self.g: MultiGraph = _make_gm_example()
        """The GM2001 example graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for the GM2001 example."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_at_least_two_nodes(self) -> None:
        """Test that the GM2001 example produces multiple nodes."""
        self.assertGreaterEqual(
            len(self.all_nodes),
            2,
            "GM2001 example has separation pairs, expect >=2 nodes",
        )


class TestSPQRMultiEdgeComplex(unittest.TestCase):
    """Tests for the SPQR-tree of a complex multi-edge graph.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for the complex multi-edge graph."""
        self.g: MultiGraph = _make_multiedge_complex()
        """The complex multi-edge graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for the multi-edge graph."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_has_p_node(self) -> None:
        """Test that multi-edges produce P-nodes in the tree."""
        p_nodes: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.P
        ]
        self.assertGreaterEqual(
            len(p_nodes), 1,
            "Multi-edge graph should have at least one P-node",
        )

    def test_has_s_node(self) -> None:
        """Test that the cycle backbone produces an S-node."""
        s_nodes: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.S
        ]
        self.assertGreaterEqual(
            len(s_nodes), 1,
            "Multi-edge graph should have at least one S-node",
        )

    def test_exact_node_structure(self) -> None:
        """Test exact SPQR-tree node counts: 2 P-nodes, 1 S-node.

        Each parallel pair forms a BOND: 2 real + 1 virtual = 3
        edges -> P-node.  The backbone cycle forms an S-node.
        """
        p_count: int = sum(
            1 for n in self.all_nodes
            if n.type == NodeType.P
        )
        s_count: int = sum(
            1 for n in self.all_nodes
            if n.type == NodeType.S
        )
        self.assertEqual(
            p_count, 2,
            f"Expected 2 P-nodes, got {p_count}",
        )
        self.assertEqual(
            s_count, 1,
            f"Expected 1 S-node, got {s_count}",
        )


def _make_single_edge() -> MultiGraph:
    """Build a graph with a single edge (vertices 0 and 1).

    This is the minimal biconnected graph. Expected: Q-node.

    :return: A MultiGraph with one edge.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(0, 1)
    return g


def _make_c5() -> MultiGraph:
    """Build the 5-cycle C5 (vertices 0-4).

    :return: A MultiGraph representing C5.
    """
    g: MultiGraph = MultiGraph()
    for i in range(5):
        g.add_edge(i, (i + 1) % 5)
    return g


def _make_c6() -> MultiGraph:
    """Build the 6-cycle C6 (vertices 0-5).

    :return: A MultiGraph representing C6.
    """
    g: MultiGraph = MultiGraph()
    for i in range(6):
        g.add_edge(i, (i + 1) % 6)
    return g


def _make_c6_with_chord() -> MultiGraph:
    """Build C6 plus chord (0,3): 7 edges, 6 vertices.

    The chord creates separation pair {0,3} yielding 3 SPQR nodes.

    :return: A MultiGraph representing C6 plus a chord.
    """
    g: MultiGraph = MultiGraph()
    for i in range(6):
        g.add_edge(i, (i + 1) % 6)
    g.add_edge(0, 3)
    return g


def _make_k5() -> MultiGraph:
    """Build the complete graph K5 (10 edges, vertices 0-4).

    K5 is 4-connected, hence a single R-node in the SPQR-tree.

    :return: A MultiGraph representing K5.
    """
    g: MultiGraph = MultiGraph()
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(i, j)
    return g


def _make_petersen() -> MultiGraph:
    """Build the Petersen graph (10 vertices, 15 edges).

    The Petersen graph is 3-connected, so it yields a single R-node.

    :return: A MultiGraph representing the Petersen graph.
    """
    g: MultiGraph = MultiGraph()
    for i in range(5):
        g.add_edge(i, (i + 1) % 5)
    for i in range(5):
        g.add_edge(i, i + 5)
    for u, v in [(5, 7), (7, 9), (9, 6), (6, 8), (8, 5)]:
        g.add_edge(u, v)
    return g


def _make_petersen_augmented() -> MultiGraph:
    """Build the Petersen graph with each edge subdivided by a path.

    For each original Petersen edge (u,v), two intermediate vertices
    w1 and w2 are added and a path u-w1-w2-v is inserted alongside
    the original edge.  Result: 40 vertices, 60 edges.

    Expected: 31 nodes (15 P + 15 S + 1 R).

    :return: The augmented Petersen multigraph.
    """
    g: MultiGraph = _make_petersen()
    petersen_edges: list[tuple[int, int]] = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
    ]
    next_v: int = 10
    for u, v in petersen_edges:
        w1: int = next_v
        w2: int = next_v + 1
        next_v += 2
        g.add_edge(u, w1)
        g.add_edge(w1, w2)
        g.add_edge(w2, v)
    return g


def _make_three_k4_cliques() -> MultiGraph:
    """Build graph: 3 K4 cliques sharing poles {0, 1}.

    Vertices 0-7; poles are 0 and 1; each clique K4(0,1,a,b) adds
    6 edges among {0,1,a,b}.  The edge (0,1) appears 3 times.
    Expected: 4 nodes (1 P + 3 R).

    :return: A MultiGraph with three K4 cliques sharing poles.
    """
    g: MultiGraph = MultiGraph()
    for a, b in [(2, 3), (4, 5), (6, 7)]:
        for u, v in [
            (0, 1), (0, a), (0, b), (1, a), (1, b), (a, b)
        ]:
            g.add_edge(u, v)
    return g


def _make_three_long_paths() -> MultiGraph:
    """Build graph: 3 paths of length 3 between vertices 0 and 1.

    Vertices 0-7; paths: 0-2-3-1, 0-4-5-1, 0-6-7-1.
    Expected: 4 nodes (1 P + 3 S).

    :return: A MultiGraph with three length-3 paths.
    """
    g: MultiGraph = MultiGraph()
    for a, b in [(2, 3), (4, 5), (6, 7)]:
        g.add_edge(0, a)
        g.add_edge(a, b)
        g.add_edge(b, 1)
    return g


def _count_nodes_by_type(root: SPQRNode) -> dict[str, int]:
    """Count SPQR-tree nodes by type.

    :param root: The SPQR-tree root.
    :return: Dict mapping type value string to count.
    """
    counts: dict[str, int] = {}
    for node in _collect_all_nodes(root):
        key: str = node.type.value
        counts[key] = counts.get(key, 0) + 1
    return counts


class TestSPQRSingleEdge(unittest.TestCase):
    """Tests for the SPQR-tree of a single-edge graph.

    A single edge is the degenerate case: one Q-node, no children.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for a single edge."""
        self.root: SPQRNode = \
            build_spqr_tree(_make_single_edge())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_root_is_q_node(self) -> None:
        """Test that a single edge produces a Q-node root."""
        self.assertEqual(self.root.type, NodeType.Q)

    def test_single_node_total(self) -> None:
        """Test that there is exactly 1 node in the tree."""
        self.assertEqual(len(self.all_nodes), 1)

    def test_no_children(self) -> None:
        """Test that the Q-node has no children."""
        self.assertEqual(len(self.root.children), 0)

    def test_skeleton_has_one_edge(self) -> None:
        """Test that the Q-node skeleton has exactly 1 edge."""
        self.assertEqual(self.root.skeleton.num_edges(), 1)


class TestSPQRC5(unittest.TestCase):
    """Tests for the SPQR-tree of the 5-cycle C5."""

    def setUp(self) -> None:
        """Build the SPQR-tree for C5."""
        self.root: SPQRNode = \
            build_spqr_tree(_make_c5())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_root_is_s_node(self) -> None:
        """Test that C5 produces a single S-node."""
        self.assertEqual(self.root.type, NodeType.S)

    def test_single_node_total(self) -> None:
        """Test that there is exactly 1 node in the tree."""
        self.assertEqual(len(self.all_nodes), 1)

    def test_skeleton_has_five_edges(self) -> None:
        """Test that the S-node skeleton has 5 edges."""
        self.assertEqual(self.root.skeleton.num_edges(), 5)


class TestSPQRC6(unittest.TestCase):
    """Tests for the SPQR-tree of the 6-cycle C6.

    Expected: single S-node (the entire cycle).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for C6."""
        self.root: SPQRNode = \
            build_spqr_tree(_make_c6())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_root_is_s_node(self) -> None:
        """Test that C6 produces a single S-node."""
        self.assertEqual(self.root.type, NodeType.S)

    def test_single_node_total(self) -> None:
        """Test that C6 yields exactly 1 SPQR node."""
        self.assertEqual(len(self.all_nodes), 1)

    def test_no_children(self) -> None:
        """Test that the root S-node has no children."""
        self.assertEqual(len(self.root.children), 0)

    def test_skeleton_has_six_edges(self) -> None:
        """Test that the S-node skeleton has 6 edges."""
        self.assertEqual(self.root.skeleton.num_edges(), 6)


class TestSPQRC6Chord(unittest.TestCase):
    """Tests for the SPQR-tree of C6 plus chord (0,3).

    The chord (0,3) creates separation pair {0,3} yielding 3 nodes:
    1 P-node (chord bond) + 2 S-nodes (the two 4-cycle halves).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for C6 with chord."""
        self.g: MultiGraph = _make_c6_with_chord()
        """The C6 with chord graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for C6 plus chord."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_three_nodes_total(self) -> None:
        """Test that C6 plus chord yields exactly 3 SPQR nodes."""
        self.assertEqual(
            len(self.all_nodes),
            3,
            f"C6+chord should have 3 SPQR nodes, "
            f"got {len(self.all_nodes)}",
        )

    def test_one_p_node(self) -> None:
        """Test that there is exactly 1 P-node (the chord bond).

        The chord (0,3) forms a BOND: 1 real edge + 2 virtual edges
        (one to each polygon side) = 3 total edges -> P-node.
        """
        p: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.P
        ]
        self.assertEqual(
            len(p), 1,
            f"Expected 1 P-node, got {len(p)}",
        )

    def test_two_s_nodes(self) -> None:
        """Test that there are exactly 2 S-nodes (the two paths)."""
        s: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.S
        ]
        self.assertEqual(
            len(s), 2,
            f"Expected 2 S-nodes, got {len(s)}",
        )


class TestSPQRK5(unittest.TestCase):
    """Tests for the SPQR-tree of the complete graph K5.

    K5 is 4-connected, so it yields a single R-node.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for K5."""
        self.root: SPQRNode = \
            build_spqr_tree(_make_k5())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_root_is_r_node(self) -> None:
        """Test that K5 produces a single R-node."""
        self.assertEqual(self.root.type, NodeType.R)

    def test_single_node_total(self) -> None:
        """Test that there is exactly 1 node in the tree."""
        self.assertEqual(len(self.all_nodes), 1)

    def test_skeleton_has_ten_edges(self) -> None:
        """Test that the R-node skeleton has 10 edges."""
        self.assertEqual(self.root.skeleton.num_edges(), 10)


class TestSPQRPetersen(unittest.TestCase):
    """Tests for the SPQR-tree of the Petersen graph.

    The Petersen graph is 3-connected, yielding a single R-node.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for the Petersen graph."""
        self.root: SPQRNode = \
            build_spqr_tree(_make_petersen())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_root_is_r_node(self) -> None:
        """Test that the Petersen graph produces a single R-node."""
        self.assertEqual(self.root.type, NodeType.R)

    def test_single_node_total(self) -> None:
        """Test that there is exactly 1 node in the tree."""
        self.assertEqual(len(self.all_nodes), 1)

    def test_no_children(self) -> None:
        """Test that the R-node has no children."""
        self.assertEqual(len(self.root.children), 0)

    def test_skeleton_has_fifteen_edges(self) -> None:
        """Test that the R-node skeleton has 15 edges."""
        self.assertEqual(self.root.skeleton.num_edges(), 15)


class TestSPQRThreeK4Cliques(unittest.TestCase):
    """Tests for the SPQR-tree of 3 K4 cliques sharing poles {0,1}.

    Expected: 4 nodes: 1 P-node (3-way parallel at 0-1) and 3
    R-nodes (one per K4 clique).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for the three-K4-cliques graph."""
        self.g: MultiGraph = _make_three_k4_cliques()
        """The three-K4-cliques graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for three K4 cliques."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_four_nodes_total(self) -> None:
        """Test that 3 K4 cliques yield exactly 4 SPQR nodes."""
        self.assertEqual(
            len(self.all_nodes),
            4,
            f"Expected 4 SPQR nodes, got {len(self.all_nodes)}",
        )

    def test_one_p_node(self) -> None:
        """Test that there is exactly 1 P-node (3+ parallel at 0-1)."""
        p: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.P
        ]
        self.assertEqual(
            len(p), 1,
            f"Expected 1 P-node, got {len(p)}",
        )

    def test_three_r_nodes(self) -> None:
        """Test that there are exactly 3 R-nodes (one per K4)."""
        r: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.R
        ]
        self.assertEqual(
            len(r), 3,
            f"Expected 3 R-nodes, got {len(r)}",
        )


class TestSPQRThreeLongPaths(unittest.TestCase):
    """Tests for the SPQR-tree of 3 length-3 paths between 0 and 1.

    Expected: 4 nodes: 1 P-node (3-way connection at poles) and 3
    S-nodes (one per length-3 path).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for the three-long-paths graph."""
        self.g: MultiGraph = _make_three_long_paths()
        """The three-long-paths graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for three long paths."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_four_nodes_total(self) -> None:
        """Test that three length-3 paths yield exactly 4 SPQR nodes."""
        self.assertEqual(
            len(self.all_nodes),
            4,
            f"Expected 4 SPQR nodes, got {len(self.all_nodes)}",
        )

    def test_one_p_node(self) -> None:
        """Test that there is exactly 1 P-node (3-way connection)."""
        p: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.P
        ]
        self.assertEqual(
            len(p), 1,
            f"Expected 1 P-node, got {len(p)}",
        )

    def test_three_s_nodes(self) -> None:
        """Test that there are exactly 3 S-nodes (one per path)."""
        s: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.S
        ]
        self.assertEqual(
            len(s), 3,
            f"Expected 3 S-nodes, got {len(s)}",
        )

    def test_s_node_skeletons_have_four_edges(self) -> None:
        """Test that each S-node skeleton has 4 edges.

        Each length-3 path has 3 real edges plus 1 virtual edge
        connecting its poles = 4 total edges in the skeleton.
        """
        for node in self.all_nodes:
            if node.type == NodeType.S:
                self.assertEqual(
                    node.skeleton.num_edges(),
                    4,
                    f"S-node skeleton should have 4 edges "
                    f"(3 real + 1 virtual), got "
                    f"{node.skeleton.num_edges()}",
                )


class TestSPQRPetersenAugmented(unittest.TestCase):
    """Tests for the SPQR-tree of the augmented Petersen graph.

    Each Petersen edge (u,v) gets a parallel path u-w1-w2-v added.
    Expected: 31 nodes (15 P + 15 S + 1 R).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for the augmented Petersen graph."""
        self.g: MultiGraph = _make_petersen_augmented()
        """The augmented Petersen graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for augmented Petersen."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_thirty_one_nodes_total(self) -> None:
        """Test that the tree has 31 nodes total.

        Expected: 15 P + 15 S + 1 R = 31 total.
        """
        self.assertEqual(
            len(self.all_nodes),
            31,
            f"Augmented Petersen should have 31 nodes, "
            f"got {len(self.all_nodes)}: "
            f"{_count_nodes_by_type(self.root)}",
        )

    def test_one_r_node(self) -> None:
        """Test that there is exactly 1 R-node (Petersen skeleton)."""
        r: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.R
        ]
        self.assertEqual(
            len(r), 1,
            f"Expected 1 R-node, got {len(r)}",
        )

    def test_fifteen_s_nodes(self) -> None:
        """Test that there are exactly 15 S-nodes."""
        s: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.S
        ]
        self.assertEqual(
            len(s), 15,
            f"Expected 15 S-nodes, got {len(s)}",
        )

    def test_fifteen_p_nodes(self) -> None:
        """Test that there are exactly 15 P-nodes.

        Each original Petersen edge (u,v) forms a BOND with the
        parallel path: 1 real + 2 virtual = 3 edges -> P-node.
        """
        p: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.P
        ]
        self.assertEqual(
            len(p), 15,
            f"Expected 15 P-nodes, got {len(p)}",
        )


def _make_k33() -> MultiGraph:
    """Build K_{3,3} (9 edges, vertices 0-5).

    K_{3,3} is 3-connected (triconnected), so its SPQR-tree is a
    single R-node.

    :return: A MultiGraph representing K_{3,3}.
    """
    g: MultiGraph = MultiGraph()
    for i in range(3):
        for j in range(3, 6):
            g.add_edge(i, j)
    return g


def _make_w4() -> MultiGraph:
    """Build wheel W4: hub vertex 0, rim vertices 1-4.

    W4 has 8 edges (4 spokes + 4 rim) and is 3-connected, so its
    SPQR-tree is a single R-node.

    :return: A MultiGraph representing W4.
    """
    g: MultiGraph = MultiGraph()
    for i in range(1, 5):
        g.add_edge(0, i)
        g.add_edge(i, i % 4 + 1)
    return g


def _make_k3_doubled() -> MultiGraph:
    """Build K3 with each edge doubled (6 edges, vertices 1-3).

    Each pair of parallel edges forms a BOND.  The triangle forms a
    POLYGON.  Expected: 4 nodes (3 P + 1 S).

    :return: A MultiGraph representing doubled K3.
    """
    g: MultiGraph = MultiGraph()
    for u, v in [(1, 2), (2, 3), (1, 3)]:
        g.add_edge(u, v)
        g.add_edge(u, v)
    return g


def _make_four_parallel() -> MultiGraph:
    """Build 4 parallel edges between vertices 1 and 2.

    Four parallel edges form a single BOND component with 4 real
    edges -> P-node.

    :return: A MultiGraph with 4 parallel edges.
    """
    g: MultiGraph = MultiGraph()
    for _ in range(4):
        g.add_edge(1, 2)
    return g


def _make_five_parallel() -> MultiGraph:
    """Build 5 parallel edges between vertices 1 and 2.

    Five parallel edges form a single BOND component with 5 real
    edges -> P-node.

    :return: A MultiGraph with 5 parallel edges.
    """
    g: MultiGraph = MultiGraph()
    for _ in range(5):
        g.add_edge(1, 2)
    return g


def _make_three_long_paths_doubled() -> MultiGraph:
    """Build 3 length-3 paths with all edges doubled.

    Vertices 0-7; paths: 0-2-3-1, 0-4-5-1, 0-6-7-1, each edge
    doubled.  Expected: 13 nodes (3 S + 10 P).

    :return: A MultiGraph with doubled three-length-3 paths.
    """
    g: MultiGraph = MultiGraph()
    for a, b in [(2, 3), (4, 5), (6, 7)]:
        for u, v in [(0, a), (a, b), (b, 1)]:
            g.add_edge(u, v)
            g.add_edge(u, v)
    return g


def _make_graph6_sage_docstring() -> MultiGraph:
    """Build the 13-vertex, 23-edge graph from graph6 'LlCG{O@?GBoMw?'.

    This biconnected graph has multiple separation pairs.  Expected:
    12 nodes (2 R + 5 S + 5 P).

    :return: A MultiGraph with 13 vertices and 23 edges.
    """
    g: MultiGraph = MultiGraph()
    edges: list[tuple[int, int]] = [
        (0, 1), (1, 2), (0, 3), (2, 3), (3, 4), (4, 5),
        (3, 6), (4, 6), (5, 6), (0, 7), (4, 7), (7, 8),
        (8, 9), (7, 10), (8, 10), (9, 10), (0, 11), (7, 11),
        (8, 11), (9, 11), (0, 12), (1, 12), (2, 12),
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g


def _make_petersen_augmented_twice() -> MultiGraph:
    """Build Petersen graph with two rounds of path augmentation.

    Round 1: for each of the 15 Petersen edges (u,v), add a parallel
    path u-w1-w2-v alongside.  Round 2: for each of the 60 round-1
    edges, add another parallel path alongside.
    Result: 160 vertices, 240 edges.
    Expected: 136 nodes (60 P + 75 S + 1 R).

    :return: The doubly-augmented Petersen multigraph.
    """
    g: MultiGraph = MultiGraph()
    for i in range(5):
        g.add_edge(i, (i + 1) % 5)
    for i in range(5):
        g.add_edge(i, i + 5)
    for u, v in [(5, 7), (7, 9), (9, 6), (6, 8), (8, 5)]:
        g.add_edge(u, v)
    petersen_edges: list[tuple[int, int]] = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
    ]
    next_v: int = 10
    for u, v in petersen_edges:
        g.add_edge(u, next_v)
        g.add_edge(next_v, next_v + 1)
        g.add_edge(next_v + 1, v)
        next_v += 2
    round1_edges: list[tuple[Hashable, Hashable]] = [
        (e.u, e.v) for e in g.edges
    ]
    for u, v in round1_edges:
        g.add_edge(u, next_v)
        g.add_edge(next_v, next_v + 1)
        g.add_edge(next_v + 1, v)
        next_v += 2
    return g


class TestSPQRK33(unittest.TestCase):
    """Tests for the SPQR-tree of K_{3,3}.

    K_{3,3} is 3-connected, so it decomposes into a single
    TRICONNECTED component -> single R-node.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for K_{3,3}."""
        self.g: MultiGraph = _make_k33()
        """The K_{3,3} graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for K_{3,3}."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_single_r_node(self) -> None:
        """Test that K_{3,3} produces exactly 1 R-node."""
        self.assertEqual(
            len(self.all_nodes), 1,
            f"K33 should have 1 node, got {len(self.all_nodes)}",
        )
        self.assertEqual(
            self.root.type, NodeType.R,
            f"K33 root should be R-node, got {self.root.type}",
        )

    def test_nine_real_edges_in_skeleton(self) -> None:
        """Test that the R-node skeleton has 9 real edges."""
        real: list[Edge] = [
            e for e in self.root.skeleton.edges
            if not e.virtual
        ]
        self.assertEqual(
            len(real), 9,
            f"K33 skeleton should have 9 real edges, got {len(real)}",
        )


class TestSPQRW4(unittest.TestCase):
    """Tests for the SPQR-tree of wheel W4.

    W4 (hub=0, rim=1-4) is 3-connected, so it decomposes into a
    single TRICONNECTED component -> single R-node.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for wheel W4."""
        self.g: MultiGraph = _make_w4()
        """The W4 wheel graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for W4."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_single_r_node(self) -> None:
        """Test that W4 produces exactly 1 R-node."""
        self.assertEqual(
            len(self.all_nodes), 1,
            f"W4 should have 1 node, got {len(self.all_nodes)}",
        )
        self.assertEqual(
            self.root.type, NodeType.R,
            f"W4 root should be R-node, got {self.root.type}",
        )

    def test_eight_real_edges_in_skeleton(self) -> None:
        """Test that the R-node skeleton has 8 real edges."""
        real: list[Edge] = [
            e for e in self.root.skeleton.edges
            if not e.virtual
        ]
        self.assertEqual(
            len(real), 8,
            f"W4 skeleton should have 8 real edges, got {len(real)}",
        )


class TestSPQRK3Doubled(unittest.TestCase):
    """Tests for the SPQR-tree of K3 with each edge doubled.

    Each pair of doubled edges forms a BOND (-> P-node).  The
    triangle K3 forms a POLYGON (-> S-node).
    Expected: 4 nodes (3 P + 1 S).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for doubled K3."""
        self.g: MultiGraph = _make_k3_doubled()
        """The doubled K3 graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for doubled K3."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_four_nodes_total(self) -> None:
        """Test that doubled K3 produces exactly 4 nodes.

        Expected: 4 total (3 P + 1 S).
        """
        self.assertEqual(
            len(self.all_nodes), 4,
            f"Doubled K3 should have 4 nodes, "
            f"got {len(self.all_nodes)}: "
            f"{_count_nodes_by_type(self.root)}",
        )

    def test_one_s_node_three_p_nodes(self) -> None:
        """Test that doubled K3 has 1 S-node and 3 P-nodes."""
        s: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.S
        ]
        p: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.P
        ]
        self.assertEqual(
            len(s), 1,
            f"Expected 1 S-node, got {len(s)}",
        )
        self.assertEqual(
            len(p), 3,
            f"Expected 3 P-nodes, got {len(p)}",
        )


class TestSPQRFourParallel(unittest.TestCase):
    """Tests for the SPQR-tree of 4 parallel edges.

    Four parallel edges form a single BOND with 4 real edges
    -> single P-node.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for 4 parallel edges."""
        self.g: MultiGraph = _make_four_parallel()
        """The 4-parallel-edges graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for 4 parallel edges."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_single_p_node(self) -> None:
        """Test that 4 parallel edges produce exactly 1 P-node."""
        self.assertEqual(
            len(self.all_nodes), 1,
            f"4-parallel should have 1 node, "
            f"got {len(self.all_nodes)}",
        )
        self.assertEqual(
            self.root.type, NodeType.P,
            f"4-parallel root should be P-node, "
            f"got {self.root.type}",
        )

    def test_four_real_edges_in_skeleton(self) -> None:
        """Test that the P-node skeleton has 4 real edges."""
        real: list[Edge] = [
            e for e in self.root.skeleton.edges
            if not e.virtual
        ]
        self.assertEqual(
            len(real), 4,
            f"4-parallel skeleton should have 4 real edges, "
            f"got {len(real)}",
        )


class TestSPQRFiveParallel(unittest.TestCase):
    """Tests for the SPQR-tree of 5 parallel edges.

    Five parallel edges form a single BOND with 5 real edges
    -> single P-node.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for 5 parallel edges."""
        self.g: MultiGraph = _make_five_parallel()
        """The 5-parallel-edges graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for 5 parallel edges."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_single_p_node(self) -> None:
        """Test that 5 parallel edges produce exactly 1 P-node."""
        self.assertEqual(
            len(self.all_nodes), 1,
            f"5-parallel should have 1 node, "
            f"got {len(self.all_nodes)}",
        )
        self.assertEqual(
            self.root.type, NodeType.P,
            f"5-parallel root should be P-node, "
            f"got {self.root.type}",
        )

    def test_five_real_edges_in_skeleton(self) -> None:
        """Test that the P-node skeleton has 5 real edges."""
        real: list[Edge] = [
            e for e in self.root.skeleton.edges
            if not e.virtual
        ]
        self.assertEqual(
            len(real), 5,
            f"5-parallel skeleton should have 5 real edges, "
            f"got {len(real)}",
        )


class TestSPQRThreeLongPathsDoubled(unittest.TestCase):
    """Tests for the SPQR-tree of three doubled length-3 paths.

    Three length-3 paths (0-a-b-1) with all edges doubled.  Each
    pair of doubled edges forms a P-node, and each path length-3
    forms an S-node.  Expected: 13 nodes (3 S + 10 P).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for doubled three-length-3 paths."""
        self.g: MultiGraph = \
            _make_three_long_paths_doubled()
        """The doubled three-long-paths graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for doubled long paths."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_thirteen_nodes_total(self) -> None:
        """Test that doubled three-paths has 13 total nodes.

        Expected: 13 total (3 S + 10 P).
        """
        self.assertEqual(
            len(self.all_nodes), 13,
            f"Doubled three-paths should have 13 nodes, "
            f"got {len(self.all_nodes)}: "
            f"{_count_nodes_by_type(self.root)}",
        )

    def test_three_s_nodes_ten_p_nodes(self) -> None:
        """Test that the tree has 3 S-nodes and 10 P-nodes."""
        s: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.S
        ]
        p: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.P
        ]
        self.assertEqual(
            len(s), 3,
            f"Expected 3 S-nodes, got {len(s)}",
        )
        self.assertEqual(
            len(p), 10,
            f"Expected 10 P-nodes, got {len(p)}",
        )


class TestSPQRSageDocstringGraph(unittest.TestCase):
    """Tests for the SPQR-tree of the 13V/23E graph (graph6 'LlCG{O@?GBoMw?').

    This biconnected graph has multiple separation pairs and yields
    12 SPQR nodes (2 R + 5 S + 5 P).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for the 13-vertex docstring graph."""
        self.g: MultiGraph = \
            _make_graph6_sage_docstring()
        """The 13V/23E docstring graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for the 13V/23E graph."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_twelve_nodes_total(self) -> None:
        """Test that the 13V/23E graph has 12 SPQR nodes total.

        Expected: 12 total (2 R + 5 S + 5 P).
        Matches SageMath's ``spqr_tree`` output.
        """
        self.assertEqual(
            len(self.all_nodes), 12,
            f"13V/23E graph should have 12 nodes, "
            f"got {len(self.all_nodes)}: "
            f"{_count_nodes_by_type(self.root)}",
        )

    def test_two_r_five_s_five_p(self) -> None:
        """Test node type counts: 2 R-nodes, 5 S-nodes, 5 P-nodes."""
        r: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.R
        ]
        s: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.S
        ]
        p: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.P
        ]
        self.assertEqual(len(r), 2, f"Expected 2 R-nodes, got {len(r)}")
        self.assertEqual(len(s), 5, f"Expected 5 S-nodes, got {len(s)}")
        self.assertEqual(len(p), 5, f"Expected 5 P-nodes, got {len(p)}")


class TestSPQRPetersenAugmentedTwice(unittest.TestCase):
    """Tests for the SPQR-tree of the doubly-augmented Petersen graph.

    Round 1: for each of the 15 Petersen edges (u,v), add a parallel
    path u-w1-w2-v alongside.  Round 2: for each of the 60 round-1
    edges, add another parallel path alongside.
    Result: 160 vertices, 240 edges, 136 SPQR nodes.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for the doubly-augmented Petersen."""
        self.g: MultiGraph = \
            _make_petersen_augmented_twice()
        """The doubly-augmented Petersen graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for doubly-aug. Petersen."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_136_nodes_total(self) -> None:
        """Test that the doubly-augmented Petersen has 136 nodes.

        Expected: 136 total (60 P + 75 S + 1 R).
        """
        self.assertEqual(
            len(self.all_nodes), 136,
            f"Doubly-augmented Petersen should have 136 nodes, "
            f"got {len(self.all_nodes)}: "
            f"{_count_nodes_by_type(self.root)}",
        )

    def test_one_r_node(self) -> None:
        """Test that there is exactly 1 R-node (Petersen skeleton)."""
        r: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.R
        ]
        self.assertEqual(
            len(r), 1,
            f"Expected 1 R-node, got {len(r)}",
        )

    def test_sixty_p_nodes(self) -> None:
        """Test that there are exactly 60 P-nodes."""
        p: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.P
        ]
        self.assertEqual(
            len(p), 60,
            f"Expected 60 P-nodes, got {len(p)}",
        )

    def test_seventy_five_s_nodes(self) -> None:
        """Test that there are exactly 75 S-nodes."""
        s: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.S
        ]
        self.assertEqual(
            len(s), 75,
            f"Expected 75 S-nodes, got {len(s)}",
        )


class TestSPQRDiamondExact(unittest.TestCase):
    """Exact SPQR-tree node tests for the diamond graph.

    The diamond has separation pair {2,3}.  Expected: 3 nodes total:
    2 S-nodes (the two triangular halves as polygons) and 1 P-node
    (the direct edge (2,3) forming a bond with 1 real + 2 virtual
    edges = 3 edges -> P-node).
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for the diamond graph."""
        self.root: SPQRNode = \
            build_spqr_tree(_make_diamond())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_exactly_three_nodes(self) -> None:
        """Test that the diamond produces exactly 3 SPQR nodes."""
        self.assertEqual(
            len(self.all_nodes),
            3,
            f"Diamond should have 3 SPQR nodes, "
            f"got {len(self.all_nodes)}",
        )

    def test_two_s_nodes_one_p_node(self) -> None:
        """Test that diamond has 2 S-nodes and 1 P-node."""
        p: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.P
        ]
        s: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.S
        ]
        self.assertEqual(len(p), 1, "Expected 1 P-node")
        self.assertEqual(len(s), 2, "Expected 2 S-nodes")


def _make_ladder() -> MultiGraph:
    """Build the 3-rung ladder graph (2x4 grid).

    Vertices 0-7.  Top row: 0-1-2-3, bottom row: 4-5-6-7,
    rungs: (0,4), (1,5), (2,6), (3,7).  Separation pairs
    {1,5} and {2,6} yield 5 SPQR nodes: 3 S-nodes + 2 P-nodes.

    Inspired by the SageMath ``spqr_tree`` test suite.

    :return: A MultiGraph representing the 3-rung ladder graph.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(4, 5)
    g.add_edge(5, 6)
    g.add_edge(6, 7)
    g.add_edge(0, 4)
    g.add_edge(1, 5)
    g.add_edge(2, 6)
    g.add_edge(3, 7)
    return g


class TestSPQRLadder(unittest.TestCase):
    """Tests for the SPQR-tree of the 3-rung ladder graph.

    The ladder (2x4 grid) has separation pairs {1,5} and {2,6}.
    Expected: 5 nodes (3 S-nodes + 2 P-nodes).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for the ladder graph."""
        self.g: MultiGraph = _make_ladder()
        """The ladder graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for the ladder graph."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_five_nodes_total(self) -> None:
        """Test that the ladder graph produces exactly 5 SPQR nodes."""
        self.assertEqual(
            len(self.all_nodes),
            5,
            f"Ladder should have 5 nodes, "
            f"got {len(self.all_nodes)}: "
            f"{_count_nodes_by_type(self.root)}",
        )

    def test_three_s_nodes_two_p_nodes(self) -> None:
        """Test that the ladder has 3 S-nodes and 2 P-nodes."""
        s: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.S
        ]
        p: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.P
        ]
        self.assertEqual(len(s), 3, "Expected 3 S-nodes")
        self.assertEqual(len(p), 2, "Expected 2 P-nodes")

    def test_no_r_or_q_nodes(self) -> None:
        """Test that the ladder has no R-nodes or Q-nodes."""
        r: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.R
        ]
        q: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.Q
        ]
        self.assertEqual(len(r), 0, "Expected no R-nodes")
        self.assertEqual(len(q), 0, "Expected no Q-nodes")


def _make_c7() -> MultiGraph:
    """Build the 7-cycle C7 (vertices 0-6).

    :return: A MultiGraph representing C7.
    """
    g: MultiGraph = MultiGraph()
    for i in range(7):
        g.add_edge(i, (i + 1) % 7)
    return g


class TestSPQRC7(unittest.TestCase):
    """Tests for the SPQR-tree of the 7-cycle C7.

    C7 is a simple cycle with 7 edges.  It yields a single
    S-node (POLYGON).
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for C7."""
        self.g: MultiGraph = _make_c7()
        """The C7 cycle graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for C7."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_single_s_node(self) -> None:
        """Test that C7 produces exactly 1 S-node."""
        self.assertEqual(len(self.all_nodes), 1)
        self.assertEqual(self.root.type, NodeType.S)

    def test_seven_real_edges_in_skeleton(self) -> None:
        """Test that the S-node skeleton has 7 real edges."""
        real: list[Edge] = [
            e for e in self.root.skeleton.edges
            if not e.virtual
        ]
        self.assertEqual(len(real), 7)


def _make_c8() -> MultiGraph:
    """Build the 8-cycle C8 (vertices 0-7).

    :return: A MultiGraph representing C8.
    """
    g: MultiGraph = MultiGraph()
    for i in range(8):
        g.add_edge(i, (i + 1) % 8)
    return g


class TestSPQRC8(unittest.TestCase):
    """Tests for the SPQR-tree of the 8-cycle C8.

    C8 is a simple cycle with 8 edges.  It yields a single
    S-node (POLYGON).
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for C8."""
        self.g: MultiGraph = _make_c8()
        """The C8 cycle graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for C8."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_single_s_node(self) -> None:
        """Test that C8 produces exactly 1 S-node."""
        self.assertEqual(len(self.all_nodes), 1)
        self.assertEqual(self.root.type, NodeType.S)

    def test_eight_real_edges_in_skeleton(self) -> None:
        """Test that the S-node skeleton has 8 real edges."""
        real: list[Edge] = [
            e for e in self.root.skeleton.edges
            if not e.virtual
        ]
        self.assertEqual(len(real), 8)


def _make_k23() -> MultiGraph:
    """Build the complete bipartite graph K_{2,3}.

    Vertices 0,1 (part A) and 2,3,4 (part B).
    Edges: all 6 pairs between parts.  K_{2,3} has vertex
    connectivity 2 (separation pair {0,1}).  Each internal
    vertex x in {2,3,4} creates a 2-edge path 0-x-1,
    yielding 4 SPQR nodes: 3 S-nodes + 1 P-node.

    Inspired by the SageMath ``spqr_tree`` test suite.

    :return: A MultiGraph representing K_{2,3}.
    """
    g: MultiGraph = MultiGraph()
    for a in [0, 1]:
        for b in [2, 3, 4]:
            g.add_edge(a, b)
    return g


class TestSPQRK23(unittest.TestCase):
    """Tests for the SPQR-tree of K_{2,3}.

    K_{2,3} has 5 vertices and 6 edges.  It has vertex
    connectivity 2 (separation pair {0,1}), yielding
    4 SPQR nodes: 3 S-nodes + 1 P-node.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for K_{2,3}."""
        self.g: MultiGraph = _make_k23()
        """The K_{2,3} graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for K_{2,3}."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_four_nodes_total(self) -> None:
        """Test that K_{2,3} produces exactly 4 SPQR nodes."""
        self.assertEqual(
            len(self.all_nodes), 4,
            f"Expected 4 nodes, got {len(self.all_nodes)}: "
            f"{_count_nodes_by_type(self.root)}",
        )

    def test_three_s_one_p(self) -> None:
        """Test that K_{2,3} has 3 S-nodes and 1 P-node."""
        s: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.S
        ]
        p: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.P
        ]
        self.assertEqual(len(s), 3, "Expected 3 S-nodes")
        self.assertEqual(len(p), 1, "Expected 1 P-node")


def _make_w5() -> MultiGraph:
    """Build the wheel graph W5 (hub + 5-cycle, 6 vertices).

    Hub vertex 0 connected to rim vertices 1-5.
    W5 is 3-connected, yielding a single R-node with 10 edges.

    Inspired by the SageMath ``spqr_tree`` test suite.

    :return: A MultiGraph representing W5.
    """
    g: MultiGraph = MultiGraph()
    for i in range(1, 6):
        g.add_edge(i, i % 5 + 1)
    for i in range(1, 6):
        g.add_edge(0, i)
    return g


class TestSPQRW5(unittest.TestCase):
    """Tests for the SPQR-tree of the wheel graph W5.

    W5 has 6 vertices and 10 edges.  It is triconnected,
    yielding a single R-node.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for W5."""
        self.g: MultiGraph = _make_w5()
        """The W5 wheel graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for W5."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_single_r_node(self) -> None:
        """Test that W5 produces exactly 1 R-node."""
        self.assertEqual(len(self.all_nodes), 1)
        self.assertEqual(self.root.type, NodeType.R)

    def test_ten_real_edges_in_skeleton(self) -> None:
        """Test that the R-node skeleton has 10 real edges."""
        real: list[Edge] = [
            e for e in self.root.skeleton.edges
            if not e.virtual
        ]
        self.assertEqual(len(real), 10)


def _make_w6() -> MultiGraph:
    """Build the wheel graph W6 (hub + 6-cycle, 7 vertices).

    Hub vertex 0 connected to rim vertices 1-6.
    W6 is 3-connected, yielding a single R-node with 12 edges.

    Inspired by the SageMath ``spqr_tree`` test suite.

    :return: A MultiGraph representing W6.
    """
    g: MultiGraph = MultiGraph()
    for i in range(1, 7):
        g.add_edge(i, i % 6 + 1)
    for i in range(1, 7):
        g.add_edge(0, i)
    return g


class TestSPQRW6(unittest.TestCase):
    """Tests for the SPQR-tree of the wheel graph W6.

    W6 has 7 vertices and 12 edges.  It is triconnected,
    yielding a single R-node.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for W6."""
        self.g: MultiGraph = _make_w6()
        """The W6 wheel graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for W6."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_single_r_node(self) -> None:
        """Test that W6 produces exactly 1 R-node."""
        self.assertEqual(len(self.all_nodes), 1)
        self.assertEqual(self.root.type, NodeType.R)

    def test_twelve_real_edges_in_skeleton(self) -> None:
        """Test that the R-node skeleton has 12 real edges."""
        real: list[Edge] = [
            e for e in self.root.skeleton.edges
            if not e.virtual
        ]
        self.assertEqual(len(real), 12)


def _make_q3_cube() -> MultiGraph:
    """Build the Q3 cube graph (3-dimensional hypercube).

    8 vertices (0-7), 12 edges.  The cube graph is 3-connected,
    yielding a single R-node.

    Inspired by the SageMath ``spqr_tree`` test suite.

    :return: A MultiGraph representing the Q3 cube.
    """
    g: MultiGraph = MultiGraph()
    # Bottom face: 0-1-2-3
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 0)
    # Top face: 4-5-6-7
    g.add_edge(4, 5)
    g.add_edge(5, 6)
    g.add_edge(6, 7)
    g.add_edge(7, 4)
    # Vertical edges
    g.add_edge(0, 4)
    g.add_edge(1, 5)
    g.add_edge(2, 6)
    g.add_edge(3, 7)
    return g


class TestSPQRQ3Cube(unittest.TestCase):
    """Tests for the SPQR-tree of the Q3 cube graph.

    The cube (Q3) has 8 vertices and 12 edges.  It is
    3-connected, yielding a single R-node.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for the Q3 cube."""
        self.g: MultiGraph = _make_q3_cube()
        """The Q3 cube graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for Q3 cube."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_single_r_node(self) -> None:
        """Test that Q3 cube produces exactly 1 R-node."""
        self.assertEqual(len(self.all_nodes), 1)
        self.assertEqual(self.root.type, NodeType.R)

    def test_twelve_real_edges_in_skeleton(self) -> None:
        """Test that the R-node skeleton has 12 real edges."""
        real: list[Edge] = [
            e for e in self.root.skeleton.edges
            if not e.virtual
        ]
        self.assertEqual(len(real), 12)


def _make_octahedron() -> MultiGraph:
    """Build the octahedron graph (6 vertices, 12 edges).

    The octahedron is the complete tripartite graph K_{2,2,2}.
    It is 4-connected, yielding a single R-node.

    Inspired by the SageMath ``spqr_tree`` test suite.

    :return: A MultiGraph representing the octahedron.
    """
    g: MultiGraph = MultiGraph()
    # All pairs except (0,5), (1,3), (2,4)
    pairs: list[tuple[int, int]] = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 4), (1, 5),
        (2, 3), (2, 5),
        (3, 4), (3, 5),
        (4, 5),
    ]
    for u, v in pairs:
        g.add_edge(u, v)
    return g


class TestSPQROctahedron(unittest.TestCase):
    """Tests for the SPQR-tree of the octahedron graph.

    The octahedron has 6 vertices and 12 edges.  It is
    4-connected, yielding a single R-node.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for the octahedron."""
        self.g: MultiGraph = _make_octahedron()
        """The octahedron graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for octahedron."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_single_r_node(self) -> None:
        """Test that octahedron produces exactly 1 R-node."""
        self.assertEqual(len(self.all_nodes), 1)
        self.assertEqual(self.root.type, NodeType.R)

    def test_twelve_real_edges_in_skeleton(self) -> None:
        """Test that the R-node skeleton has 12 real edges."""
        real: list[Edge] = [
            e for e in self.root.skeleton.edges
            if not e.virtual
        ]
        self.assertEqual(len(real), 12)


def _make_k4_one_doubled() -> MultiGraph:
    """Build K4 with one edge doubled.

    K4 has 6 edges.  One edge (1-2) is doubled, giving 7
    edges total.  The doubled edge creates separation pair
    {1,2}, yielding 2 SPQR nodes: 1 P-node (BOND with the
    doubled edge) and 1 R-node (K4 skeleton).

    Inspired by the SageMath ``spqr_tree`` test suite.

    :return: A MultiGraph representing K4 with one doubled edge.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(1, 4)
    g.add_edge(2, 3)
    g.add_edge(2, 4)
    g.add_edge(3, 4)
    g.add_edge(1, 2)  # duplicate
    return g


class TestSPQRK4OneDoubled(unittest.TestCase):
    """Tests for the SPQR-tree of K4 + 1 doubled edge.

    K4 with one edge doubled has 7 edges.  Expected: 2 SPQR
    nodes: 1 P-node and 1 R-node.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for K4 + 1 doubled edge."""
        self.g: MultiGraph = _make_k4_one_doubled()
        """The K4 with one doubled edge under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for K4+doubled."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_two_nodes_total(self) -> None:
        """Test that K4+doubled produces exactly 2 SPQR nodes."""
        self.assertEqual(
            len(self.all_nodes), 2,
            f"Expected 2 nodes, got {len(self.all_nodes)}: "
            f"{_count_nodes_by_type(self.root)}",
        )

    def test_one_p_one_r(self) -> None:
        """Test that K4+doubled has 1 P-node and 1 R-node."""
        p: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.P
        ]
        r: list[SPQRNode] = [
            n for n in self.all_nodes
            if n.type == NodeType.R
        ]
        self.assertEqual(len(p), 1, "Expected 1 P-node")
        self.assertEqual(len(r), 1, "Expected 1 R-node")


def _make_mobius_kantor() -> MultiGraph:
    """Build the Mobius-Kantor graph GP(8,3) (16 verts, 24 edges).

    The Mobius-Kantor graph is the generalized Petersen graph
    GP(8,3).  It is 3-regular and 3-connected, yielding a
    single R-node with 24 real edges.

    Inspired by the SageMath ``spqr_tree`` test suite.

    :return: A MultiGraph representing the Mobius-Kantor graph.
    """
    g: MultiGraph = MultiGraph()
    # Outer 8-cycle: 0-1-2-3-4-5-6-7-0
    for i in range(8):
        g.add_edge(i, (i + 1) % 8)
    # Spokes: i -> 8+i
    for i in range(8):
        g.add_edge(i, 8 + i)
    # Inner star (step 3): 8+i -> 8+(i+3)%8
    for i in range(8):
        g.add_edge(8 + i, 8 + (i + 3) % 8)
    return g


class TestSPQRMobiusKantor(unittest.TestCase):
    """Tests for the SPQR-tree of the Mobius-Kantor graph.

    The Mobius-Kantor graph has 16 vertices and 24 edges.
    It is 3-connected, yielding a single R-node.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the SPQR-tree for the Mobius-Kantor graph."""
        self.g: MultiGraph = _make_mobius_kantor()
        """The Mobius-Kantor graph under test."""
        self.root: SPQRNode = build_spqr_tree(self.g)
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_all_invariants(self) -> None:
        """Test all SPQR-tree invariants for Mobius-Kantor."""
        _check_spqr_invariants(self, self.g, self.root)

    def test_single_r_node(self) -> None:
        """Test that Mobius-Kantor produces exactly 1 R-node."""
        self.assertEqual(len(self.all_nodes), 1)
        self.assertEqual(self.root.type, NodeType.R)

    def test_twenty_four_real_edges_in_skeleton(self) -> None:
        """Test that the R-node skeleton has 24 real edges."""
        real: list[Edge] = [
            e for e in self.root.skeleton.edges
            if not e.virtual
        ]
        self.assertEqual(len(real), 24)


def _make_large_ladder(n: int) -> MultiGraph:
    """Build a ladder graph with n rungs (2n vertices, 3n-2 edges).

    The ladder graph is biconnected and produces a chain of
    triconnected components, exercising the full SPQR algorithm.

    :param n: The number of rungs (must be >= 2).
    :return: A MultiGraph representing the ladder.
    """
    g: MultiGraph = MultiGraph()
    for i in range(n):
        g.add_edge(2 * i, 2 * i + 1)
    for i in range(n - 1):
        g.add_edge(2 * i, 2 * (i + 1))
        g.add_edge(2 * i + 1, 2 * (i + 1) + 1)
    return g


class LinearTimeComplexityTest(unittest.TestCase):
    """Test that SPQR-tree construction runs in linear time.

    Builds ladder graphs of increasing sizes, measures the time for
    build_spqr_tree, and checks that the growth ratio is consistent
    with O(n) rather than O(n^2).
    """

    def test_linear_time(self) -> None:
        """Test that doubling graph size roughly doubles run time.

        Builds ladder graphs with 2000, 4000, 8000, and 16000
        rungs.  For each size, measures the median time over 3
        runs.  Asserts that every ratio t(2n)/t(n) is below 3.0,
        which rejects O(n^2) behavior (expected ratio ~4) while
        tolerating noise in a linear algorithm (expected ratio ~2).
        """
        sizes: list[int] = [2000, 4000, 8000, 16000]
        times: list[float] = []
        for n in sizes:
            g: MultiGraph = _make_large_ladder(n)
            trial_times: list[float] = []
            for _ in range(3):
                t0: float = time.perf_counter()
                build_spqr_tree(g.copy())
                t1: float = time.perf_counter()
                trial_times.append(t1 - t0)
            trial_times.sort()
            times.append(trial_times[1])  # median
        for i in range(1, len(sizes)):
            ratio: float = times[i] / times[i - 1]
            self.assertLess(
                ratio, 3.0,
                f"Time ratio for n={sizes[i]} vs "
                f"n={sizes[i - 1]} is {ratio:.2f}, "
                f"suggesting super-linear growth "
                f"(times: {times})"
            )


def _make_wikimedia_spqr() -> MultiGraph:
    """Build the Wikimedia Commons SPQR-tree example graph.

    16 vertices (a-p), 26 edges.  From File:SPQR_tree_2.svg.

    :return: A MultiGraph representing the Wikimedia SPQR example.
    """
    g: MultiGraph = MultiGraph()
    edges: list[tuple[str, str]] = [
        ('a', 'b'), ('a', 'c'), ('a', 'g'),
        ('b', 'd'), ('b', 'h'), ('c', 'd'),
        ('c', 'e'), ('d', 'f'), ('e', 'f'),
        ('e', 'g'), ('f', 'h'), ('h', 'i'),
        ('h', 'j'), ('i', 'j'), ('i', 'n'),
        ('j', 'k'), ('k', 'm'), ('k', 'n'),
        ('m', 'n'), ('l', 'm'), ('l', 'o'),
        ('l', 'p'), ('m', 'o'), ('m', 'p'),
        ('o', 'p'), ('g', 'l'),
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g


def _make_rpst_fig1a() -> MultiGraph:
    """Build the RPST paper Figure 1(a) graph.

    15 vertices, 19 edges.  From Polyvyanyy, Vanhatalo &
    Voelzer (2011), Figure 1(a).

    :return: A MultiGraph representing RPST Fig 1(a).
    """
    g: MultiGraph = MultiGraph()
    edges: list[tuple[str, str]] = [
        ('s', 'u'), ('a1', 'u'), ('a4', 'u'),
        ('a1', 'v'), ('a4', 'w'), ('a3', 'v'),
        ('a3', 'w'), ('a2', 'v'), ('a5', 'w'),
        ('a2', 'x'), ('a5', 'x'), ('x', 'y'),
        ('a6', 'y'), ('y', 'z'), ('a7', 'y'),
        ('a6', 'z'), ('a7', 'z'), ('t', 'z'),
        ('s', 't'),
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g


class TestSPQRWikimediaSpqr(unittest.TestCase):
    """Tests for the SPQR-tree of the Wikimedia SPQR example."""

    def setUp(self) -> None:
        """Build the SPQR-tree for the Wikimedia example."""
        self.root: SPQRNode = \
            build_spqr_tree(_make_wikimedia_spqr())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_node_count(self) -> None:
        """Test that the tree has exactly 5 nodes."""
        self.assertEqual(len(self.all_nodes), 5)

    def test_r_node_count(self) -> None:
        """Test that there are exactly 3 R-nodes."""
        n: int = sum(
            1 for nd in self.all_nodes
            if nd.type == NodeType.R
        )
        self.assertEqual(n, 3)

    def test_s_node_count(self) -> None:
        """Test that there is exactly 1 S-node."""
        n: int = sum(
            1 for nd in self.all_nodes
            if nd.type == NodeType.S
        )
        self.assertEqual(n, 1)

    def test_p_node_count(self) -> None:
        """Test that there is exactly 1 P-node."""
        n: int = sum(
            1 for nd in self.all_nodes
            if nd.type == NodeType.P
        )
        self.assertEqual(n, 1)

    def test_no_adjacent_s_nodes(self) -> None:
        """Test that no S-node is adjacent to another S-node."""
        _assert_no_ss_pp(self, self.root, NodeType.S)

    def test_no_adjacent_p_nodes(self) -> None:
        """Test that no P-node is adjacent to another P-node."""
        _assert_no_ss_pp(self, self.root, NodeType.P)


class TestSPQRRpstFig1a(unittest.TestCase):
    """Tests for the SPQR-tree of the RPST Fig 1(a) graph."""

    def setUp(self) -> None:
        """Build the SPQR-tree for RPST Fig 1(a)."""
        self.root: SPQRNode = \
            build_spqr_tree(_make_rpst_fig1a())
        """The root node of the SPQR tree."""
        self.all_nodes: list[SPQRNode] = \
            _collect_all_nodes(self.root)
        """All SPQR tree nodes."""

    def test_node_count(self) -> None:
        """Test that the tree has exactly 10 nodes."""
        self.assertEqual(len(self.all_nodes), 10)

    def test_r_node_count(self) -> None:
        """Test that there is exactly 1 R-node."""
        n: int = sum(
            1 for nd in self.all_nodes
            if nd.type == NodeType.R
        )
        self.assertEqual(n, 1)

    def test_s_node_count(self) -> None:
        """Test that there are exactly 8 S-nodes."""
        n: int = sum(
            1 for nd in self.all_nodes
            if nd.type == NodeType.S
        )
        self.assertEqual(n, 8)

    def test_p_node_count(self) -> None:
        """Test that there is exactly 1 P-node."""
        n: int = sum(
            1 for nd in self.all_nodes
            if nd.type == NodeType.P
        )
        self.assertEqual(n, 1)

    def test_no_adjacent_s_nodes(self) -> None:
        """Test that no S-node is adjacent to another S-node."""
        _assert_no_ss_pp(self, self.root, NodeType.S)

    def test_no_adjacent_p_nodes(self) -> None:
        """Test that no P-node is adjacent to another P-node."""
        _assert_no_ss_pp(self, self.root, NodeType.P)
