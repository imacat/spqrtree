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
"""Tests for the triconnected components algorithm (_triconnected.py).

Tests cover: triangle K3, 4-cycle C4, complete graph K4, two parallel
edges, three parallel edges, real-edge count invariant, and virtual
edge appearance count.
"""
import unittest
from collections.abc import Hashable

from spqrtree import (
    ComponentType,
    Edge,
    MultiGraph,
    TriconnectedComponent,
    find_triconnected_components,
)


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


def _count_real_edges(
    components: list[TriconnectedComponent],
) -> int:
    """Count total real (non-virtual) edges across all components.

    :param components: A list of triconnected components.
    :return: Total count of real edges summed over all components.
    """
    return sum(
        1
        for comp in components
        for e in comp.edges
        if not e.virtual
    )


def _virtual_edge_component_count(
    components: list[TriconnectedComponent],
) -> dict[int, int]:
    """Count how many components each virtual edge appears in.

    :param components: A list of triconnected components.
    :return: A dict mapping virtual edge ID to component count.
    """
    counts: dict[int, int] = {}
    for comp in components:
        for e in comp.edges:
            if e.virtual:
                counts[e.id] = counts.get(e.id, 0) + 1
    return counts


class TestTriconnectedK3(unittest.TestCase):
    """Tests for triconnected decomposition of the triangle K3."""

    def setUp(self) -> None:
        """Build the components for K3."""
        self.g: MultiGraph = _make_k3()
        """The K3 graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_returns_list(self) -> None:
        """Test that find_triconnected_components returns a list."""
        self.assertIsInstance(self.comps, list)

    def test_single_component(self) -> None:
        """Test that K3 produces exactly 1 triconnected component."""
        self.assertEqual(len(self.comps), 1)

    def test_component_is_polygon(self) -> None:
        """Test that K3 yields a POLYGON component."""
        self.assertEqual(self.comps[0].type, ComponentType.POLYGON)

    def test_polygon_has_three_real_edges(self) -> None:
        """Test that the POLYGON from K3 contains 3 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 3)

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )


class TestTriconnectedC4(unittest.TestCase):
    """Tests for triconnected decomposition of the 4-cycle C4."""

    def setUp(self) -> None:
        """Build the components for C4."""
        self.g: MultiGraph = _make_c4()
        """The C4 graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_single_component(self) -> None:
        """Test that C4 produces exactly 1 triconnected component."""
        self.assertEqual(len(self.comps), 1)

    def test_component_is_polygon(self) -> None:
        """Test that C4 yields a POLYGON component."""
        self.assertEqual(self.comps[0].type, ComponentType.POLYGON)

    def test_polygon_has_four_real_edges(self) -> None:
        """Test that the POLYGON from C4 contains 4 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 4)

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )


class TestTriconnectedK4(unittest.TestCase):
    """Tests for triconnected decomposition of the complete graph K4."""

    def setUp(self) -> None:
        """Build the components for K4."""
        self.g: MultiGraph = _make_k4()
        """The K4 graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_single_component(self) -> None:
        """Test that K4 produces exactly 1 triconnected component."""
        self.assertEqual(len(self.comps), 1)

    def test_component_is_triconnected(self) -> None:
        """Test that K4 yields a TRICONNECTED component."""
        self.assertEqual(
            self.comps[0].type, ComponentType.TRICONNECTED
        )

    def test_triconnected_has_six_real_edges(self) -> None:
        """Test that the TRICONNECTED from K4 has 6 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 6)

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )


class TestTriconnectedTwoParallel(unittest.TestCase):
    """Tests for triconnected decomposition of two parallel edges."""

    def setUp(self) -> None:
        """Build the components for 2 parallel edges between 1 and 2."""
        self.g: MultiGraph = _make_two_parallel()
        """The two-parallel-edges graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_single_component(self) -> None:
        """Test that 2 parallel edges produce exactly 1 component."""
        self.assertEqual(len(self.comps), 1)

    def test_component_is_bond(self) -> None:
        """Test that 2 parallel edges yield a BOND component."""
        self.assertEqual(self.comps[0].type, ComponentType.BOND)

    def test_bond_has_two_real_edges(self) -> None:
        """Test that the BOND from 2 parallel edges has 2 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 2)

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )


class TestTriconnectedThreeParallel(unittest.TestCase):
    """Tests for triconnected decomposition of three parallel edges."""

    def setUp(self) -> None:
        """Build the components for 3 parallel edges between 1 and 2."""
        self.g: MultiGraph = _make_three_parallel()
        """The three-parallel-edges graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_single_component(self) -> None:
        """Test that 3 parallel edges produce exactly 1 component."""
        self.assertEqual(len(self.comps), 1)

    def test_component_is_bond(self) -> None:
        """Test that 3 parallel edges yield a BOND component."""
        self.assertEqual(self.comps[0].type, ComponentType.BOND)

    def test_bond_has_three_real_edges(self) -> None:
        """Test that the BOND has 3 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 3)

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )


class TestTriconnectedInvariants(unittest.TestCase):
    """Tests for global invariants across all graphs."""

    def _check_real_edge_count(self, g: MultiGraph) -> None:
        """Check that real edge count is preserved across decomposition.

        :param g: The input graph.
        :return: None
        """
        comps: list[TriconnectedComponent] = \
            find_triconnected_components(g)
        self.assertEqual(_count_real_edges(comps), g.num_edges())

    def _check_virtual_edges_in_two_comps(
        self, comps: list[TriconnectedComponent]
    ) -> None:
        """Check that each virtual edge appears in exactly 2 components.

        :param comps: The list of triconnected components.
        :return: None
        """
        counts: dict[int, int] = \
            _virtual_edge_component_count(comps)
        for eid, cnt in counts.items():
            self.assertEqual(
                cnt,
                2,
                f"Virtual edge {eid} appears in {cnt} components "
                f"(expected 2)",
            )

    def test_k3_real_edge_count(self) -> None:
        """Test real edge count invariant for K3."""
        self._check_real_edge_count(_make_k3())

    def test_c4_real_edge_count(self) -> None:
        """Test real edge count invariant for C4."""
        self._check_real_edge_count(_make_c4())

    def test_k4_real_edge_count(self) -> None:
        """Test real edge count invariant for K4."""
        self._check_real_edge_count(_make_k4())

    def test_two_parallel_real_edge_count(self) -> None:
        """Test real edge count invariant for 2 parallel edges."""
        self._check_real_edge_count(_make_two_parallel())

    def test_three_parallel_real_edge_count(self) -> None:
        """Test real edge count invariant for 3 parallel edges."""
        self._check_real_edge_count(_make_three_parallel())

    def test_k3_virtual_edges_in_two_comps(self) -> None:
        """Test virtual edge invariant for K3."""
        comps: list[TriconnectedComponent] = \
            find_triconnected_components(_make_k3())
        self._check_virtual_edges_in_two_comps(comps)

    def test_c4_virtual_edges_in_two_comps(self) -> None:
        """Test virtual edge invariant for C4."""
        comps: list[TriconnectedComponent] = \
            find_triconnected_components(_make_c4())
        self._check_virtual_edges_in_two_comps(comps)

    def test_k4_virtual_edges_in_two_comps(self) -> None:
        """Test virtual edge invariant for K4."""
        comps: list[TriconnectedComponent] = \
            find_triconnected_components(_make_k4())
        self._check_virtual_edges_in_two_comps(comps)

    def test_two_parallel_virtual_edges_in_two_comps(self) -> None:
        """Test virtual edge invariant for 2 parallel edges."""
        comps: list[TriconnectedComponent] = \
            find_triconnected_components(_make_two_parallel())
        self._check_virtual_edges_in_two_comps(comps)

    def test_three_parallel_virtual_edges_in_two_comps(self) -> None:
        """Test virtual edge invariant for 3 parallel edges."""
        comps: list[TriconnectedComponent] = \
            find_triconnected_components(_make_three_parallel())
        self._check_virtual_edges_in_two_comps(comps)

    def test_component_types_are_valid(self) -> None:
        """Test that all component types are valid ComponentType values."""
        for g in [
            _make_k3(), _make_c4(), _make_k4(),
            _make_two_parallel(), _make_three_parallel(),
        ]:
            comps: list[TriconnectedComponent] = \
                find_triconnected_components(g)
            for comp in comps:
                self.assertIsInstance(comp.type, ComponentType)
                self.assertIn(
                    comp.type,
                    [
                        ComponentType.BOND,
                        ComponentType.POLYGON,
                        ComponentType.TRICONNECTED,
                    ],
                )

    def test_each_component_has_edges(self) -> None:
        """Test that every component has at least 2 edges."""
        for g in [
            _make_k3(), _make_c4(), _make_k4(),
            _make_two_parallel(), _make_three_parallel(),
        ]:
            comps: list[TriconnectedComponent] = \
                find_triconnected_components(g)
            for comp in comps:
                self.assertGreaterEqual(
                    len(comp.edges),
                    2,
                    f"Component {comp.type} has fewer than 2 edges",
                )


def _make_diamond() -> MultiGraph:
    """Build the diamond graph (K4 minus one edge).

    Vertices 1,2,3,4; edges: 1-2, 1-3, 2-3, 2-4, 3-4.
    The pair {2,3} is a separation pair.

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
    The pair {1,2} is a separation pair.

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

    Two triangles connected by 3 edges.
    Vertices 1-6; this graph is 3-connected.

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


class TestTriconnectedDiamond(unittest.TestCase):
    """Tests for triconnected decomposition of the diamond graph."""

    def setUp(self) -> None:
        """Build the components for the diamond graph."""
        self.g: MultiGraph = _make_diamond()
        """The diamond graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_at_least_two_components(self) -> None:
        """Test that the diamond produces at least 2 components."""
        self.assertGreaterEqual(
            len(self.comps),
            2,
            "Diamond has separation pair {2,3}, expect >=2 components",
        )

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count (5)."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )

    def test_virtual_edges_in_exactly_two_comps(self) -> None:
        """Test that each virtual edge appears in exactly 2 components."""
        counts: dict[int, int] = \
            _virtual_edge_component_count(self.comps)
        for eid, cnt in counts.items():
            self.assertEqual(
                cnt,
                2,
                f"Virtual edge {eid} appears in {cnt} components "
                f"(expected 2)",
            )

    def test_no_ss_adjacency(self) -> None:
        """Test that no two S-type components share a virtual edge."""
        _assert_no_same_type_adjacency(
            self, self.comps, ComponentType.POLYGON
        )

    def test_no_pp_adjacency(self) -> None:
        """Test that no two P-type components share a virtual edge."""
        _assert_no_same_type_adjacency(
            self, self.comps, ComponentType.BOND
        )

    def test_each_component_has_at_least_two_edges(self) -> None:
        """Test that every component has at least 2 edges."""
        for comp in self.comps:
            self.assertGreaterEqual(
                len(comp.edges),
                2,
                f"Component {comp.type} has fewer than 2 edges",
            )


class TestTriconnectedTheta(unittest.TestCase):
    """Tests for triconnected decomposition of the theta graph."""

    def setUp(self) -> None:
        """Build the components for the theta graph."""
        self.g: MultiGraph = _make_theta()
        """The theta graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count (6)."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )

    def test_virtual_edges_in_exactly_two_comps(self) -> None:
        """Test that each virtual edge appears in exactly 2 components."""
        counts: dict[int, int] = \
            _virtual_edge_component_count(self.comps)
        for eid, cnt in counts.items():
            self.assertEqual(
                cnt,
                2,
                f"Virtual edge {eid} appears in {cnt} components "
                f"(expected 2)",
            )

    def test_no_ss_adjacency(self) -> None:
        """Test that no two S-type components share a virtual edge."""
        _assert_no_same_type_adjacency(
            self, self.comps, ComponentType.POLYGON
        )

    def test_no_pp_adjacency(self) -> None:
        """Test that no two P-type components share a virtual edge."""
        _assert_no_same_type_adjacency(
            self, self.comps, ComponentType.BOND
        )

    def test_each_component_has_at_least_two_edges(self) -> None:
        """Test that every component has at least 2 edges."""
        for comp in self.comps:
            self.assertGreaterEqual(
                len(comp.edges),
                2,
                f"Component {comp.type} has fewer than 2 edges",
            )


class TestTriconnectedPrism(unittest.TestCase):
    """Tests for triconnected decomposition of the triangular prism."""

    def setUp(self) -> None:
        """Build the components for the triangular prism."""
        self.g: MultiGraph = _make_prism()
        """The triangular prism graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_single_triconnected_component(self) -> None:
        """Test that the prism (3-connected) yields 1 component."""
        self.assertEqual(len(self.comps), 1)

    def test_component_is_triconnected(self) -> None:
        """Test that the single component is TRICONNECTED."""
        self.assertEqual(
            self.comps[0].type, ComponentType.TRICONNECTED
        )

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count (9)."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )

    def test_nine_real_edges(self) -> None:
        """Test that the TRICONNECTED component contains all 9 edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 9)


def _assert_no_same_type_adjacency(
    tc: unittest.TestCase,
    comps: list[TriconnectedComponent],
    ctype: ComponentType,
) -> None:
    """Assert that no two components of *ctype* share a virtual edge.

    Checks the SPQR-tree invariant that adjacent components in the
    decomposition have different types (no S-S or P-P pairs after
    merging).

    :param tc: The TestCase instance for assertions.
    :param comps: The list of triconnected components.
    :param ctype: The component type to check (BOND or POLYGON).
    :return: None
    """
    # Map: virtual edge ID -> list of component indices containing it.
    ve_to_comps: dict[int, list[int]] = {}
    for i, comp in enumerate(comps):
        for e in comp.edges:
            if e.virtual:
                ve_to_comps.setdefault(e.id, []).append(i)

    for eid, idxs in ve_to_comps.items():
        if len(idxs) == 2:
            i, j = idxs
            both_same: bool = (
                comps[i].type == ctype
                and comps[j].type == ctype
            )
            tc.assertFalse(
                both_same,
                f"Virtual edge {eid} shared by two {ctype.name} "
                f"components (S-S or P-P adjacency not allowed)",
            )


class TestTriconnectedNoSSPP(unittest.TestCase):
    """Tests that no S-S or P-P adjacency occurs for any graph."""

    def _check_no_ss(self, g: MultiGraph) -> None:
        """Check no S-S adjacency for graph g.

        :param g: The input multigraph.
        :return: None
        """
        comps: list[TriconnectedComponent] = \
            find_triconnected_components(g)
        _assert_no_same_type_adjacency(
            self, comps, ComponentType.POLYGON
        )

    def _check_no_pp(self, g: MultiGraph) -> None:
        """Check no P-P adjacency for graph g.

        :param g: The input multigraph.
        :return: None
        """
        comps: list[TriconnectedComponent] = \
            find_triconnected_components(g)
        _assert_no_same_type_adjacency(
            self, comps, ComponentType.BOND
        )

    def test_k3_no_ss(self) -> None:
        """Test no S-S adjacency for K3."""
        self._check_no_ss(_make_k3())

    def test_c4_no_ss(self) -> None:
        """Test no S-S adjacency for C4."""
        self._check_no_ss(_make_c4())

    def test_k4_no_ss(self) -> None:
        """Test no S-S adjacency for K4."""
        self._check_no_ss(_make_k4())

    def test_diamond_no_ss(self) -> None:
        """Test no S-S adjacency for the diamond graph."""
        self._check_no_ss(_make_diamond())

    def test_theta_no_ss(self) -> None:
        """Test no S-S adjacency for the theta graph."""
        self._check_no_ss(_make_theta())

    def test_prism_no_ss(self) -> None:
        """Test no S-S adjacency for the triangular prism."""
        self._check_no_ss(_make_prism())

    def test_k3_no_pp(self) -> None:
        """Test no P-P adjacency for K3."""
        self._check_no_pp(_make_k3())

    def test_c4_no_pp(self) -> None:
        """Test no P-P adjacency for C4."""
        self._check_no_pp(_make_c4())

    def test_k4_no_pp(self) -> None:
        """Test no P-P adjacency for K4."""
        self._check_no_pp(_make_k4())

    def test_diamond_no_pp(self) -> None:
        """Test no P-P adjacency for the diamond graph."""
        self._check_no_pp(_make_diamond())

    def test_theta_no_pp(self) -> None:
        """Test no P-P adjacency for the theta graph."""
        self._check_no_pp(_make_theta())

    def test_prism_no_pp(self) -> None:
        """Test no P-P adjacency for the triangular prism."""
        self._check_no_pp(_make_prism())


def _check_all_invariants(
    tc: unittest.TestCase,
    g: MultiGraph,
    comps: list[TriconnectedComponent],
) -> None:
    """Check all decomposition invariants for a given graph.

    Verifies: real edge count preserved, virtual edges in exactly 2
    components, no S-S or P-P adjacency, each component has >= 2 edges,
    and reconstruction (real edges) matches the original graph.

    :param tc: The TestCase instance for assertions.
    :param g: The original input graph.
    :param comps: The triconnected components to check.
    :return: None
    """
    # 1. Real edge count.
    tc.assertEqual(
        _count_real_edges(comps),
        g.num_edges(),
        "Real edge count mismatch",
    )
    # 2. Virtual edges in exactly 2 components.
    counts: dict[int, int] = _virtual_edge_component_count(comps)
    for eid, cnt in counts.items():
        tc.assertEqual(
            cnt, 2,
            f"Virtual edge {eid} in {cnt} components (expected 2)",
        )
    # 3. No S-S adjacency.
    _assert_no_same_type_adjacency(tc, comps, ComponentType.POLYGON)
    # 4. No P-P adjacency.
    _assert_no_same_type_adjacency(tc, comps, ComponentType.BOND)
    # 5. Each component has at least 2 edges.
    for comp in comps:
        tc.assertGreaterEqual(
            len(comp.edges), 2,
            f"Component {comp.type} has fewer than 2 edges",
        )
    # 6. Reconstruction: real edges across all components == original.
    orig_edge_ids: set[int] = {e.id for e in g.edges}
    decomp_real_ids: set[int] = set()
    for comp in comps:
        for e in comp.edges:
            if not e.virtual:
                decomp_real_ids.add(e.id)
    tc.assertEqual(
        decomp_real_ids,
        orig_edge_ids,
        "Reconstructed real-edge set does not match original graph",
    )


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
        (8, 11), (8, 9), (8, 12), (9, 10), (9, 11),
        (9, 12), (10, 12),
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
        (8, 9), (8, 11), (8, 12), (9, 10), (9, 11),
        (9, 12), (10, 11), (10, 12),
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g


def _make_gm_example() -> MultiGraph:
    """Build the Gutwenger-Mutzel (2001) example graph.

    28 edges, 16 vertices. Used in [GM2001] as the running example.

    :return: A MultiGraph representing the GM2001 example.
    """
    g: MultiGraph = MultiGraph()
    edges: list[tuple[int, int]] = [
        (1, 2), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5),
        (4, 5), (4, 6), (5, 7), (5, 8), (5, 14), (6, 8),
        (7, 14), (8, 9), (8, 10), (8, 11), (8, 12),
        (9, 10), (10, 13), (10, 14), (10, 15), (10, 16),
        (11, 12), (11, 13), (12, 13),
        (14, 15), (14, 16), (15, 16),
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g


def _make_multiedge_complex() -> MultiGraph:
    """Build a complex graph with multi-edges embedded in a larger graph.

    5 vertices, 7 edges; two pairs of parallel edges (1-5 and 2-3)
    embedded in a cycle.  Expected: 2 BOND + 1 POLYGON.

    :return: A MultiGraph with embedded parallel edges.
    """
    g: MultiGraph = MultiGraph()
    # Cycle backbone: 1-2, 3-4, 4-5
    g.add_edge(1, 2)
    # Double edge 2-3
    g.add_edge(2, 3)
    g.add_edge(2, 3)
    # Continue cycle: 3-4, 4-5
    g.add_edge(3, 4)
    g.add_edge(4, 5)
    # Double edge 1-5
    g.add_edge(1, 5)
    g.add_edge(1, 5)
    return g


class TestTriconnectedWikipediaExample(unittest.TestCase):
    """Tests for triconnected decomposition of the Wikipedia example."""

    def setUp(self) -> None:
        """Build the components for the Wikipedia SPQR-tree example."""
        self.g: MultiGraph = _make_wikipedia_example()
        """The Wikipedia example graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for the Wikipedia example."""
        _check_all_invariants(self, self.g, self.comps)

    def test_at_least_two_components(self) -> None:
        """Test that the Wikipedia example produces multiple components."""
        self.assertGreaterEqual(
            len(self.comps),
            2,
            "Wikipedia example has separation pairs, expect >=2 comps",
        )


class TestTriconnectedHTExample(unittest.TestCase):
    """Tests for triconnected decomposition of the HT1973 example."""

    def setUp(self) -> None:
        """Build the components for the Hopcroft-Tarjan 1973 example."""
        self.g: MultiGraph = _make_ht_example()
        """The Hopcroft-Tarjan 1973 example graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for the HT1973 example."""
        _check_all_invariants(self, self.g, self.comps)

    def test_at_least_two_components(self) -> None:
        """Test that the HT1973 example produces multiple components."""
        self.assertGreaterEqual(
            len(self.comps),
            2,
            "HT1973 example has separation pairs, expect >=2 comps",
        )


class TestTriconnectedGMExample(unittest.TestCase):
    """Tests for triconnected decomposition of the GM2001 example."""

    def setUp(self) -> None:
        """Build the components for the Gutwenger-Mutzel 2001 example."""
        self.g: MultiGraph = _make_gm_example()
        """The Gutwenger-Mutzel 2001 example graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for the GM2001 example."""
        _check_all_invariants(self, self.g, self.comps)

    def test_at_least_two_components(self) -> None:
        """Test that the GM2001 example produces multiple components."""
        self.assertGreaterEqual(
            len(self.comps),
            2,
            "GM2001 example has separation pairs, expect >=2 comps",
        )


class TestTriconnectedMultiEdgeComplex(unittest.TestCase):
    """Tests for decomposition of a complex multi-edge graph.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for the complex multi-edge graph."""
        self.g: MultiGraph = _make_multiedge_complex()
        """The multi-edge complex graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for the multi-edge graph."""
        _check_all_invariants(self, self.g, self.comps)

    def test_has_bond_components(self) -> None:
        """Test that multi-edges produce BOND components."""
        bond_types: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.BOND
        ]
        self.assertGreaterEqual(
            len(bond_types), 1,
            "Multi-edge graph should have at least one BOND "
            "component",
        )

    def test_has_polygon_component(self) -> None:
        """Test that the backbone cycle produces a POLYGON component."""
        poly_types: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.POLYGON
        ]
        self.assertGreaterEqual(
            len(poly_types), 1,
            "Multi-edge graph should have at least one POLYGON",
        )

    def test_exact_component_structure(self) -> None:
        """Test exact component counts: 2 BONDs and 1 POLYGON.

        The graph (1,2),(1,5)x2,(2,3)x2,(3,4),(4,5) has two
        parallel pairs -> 2 BOND, and a backbone cycle -> 1 POLYGON.
        """
        bond_count: int = sum(
            1 for c in self.comps
            if c.type == ComponentType.BOND
        )
        poly_count: int = sum(
            1 for c in self.comps
            if c.type == ComponentType.POLYGON
        )
        self.assertEqual(
            bond_count, 2,
            f"Expected 2 BOND components, got {bond_count}",
        )
        self.assertEqual(
            poly_count, 1,
            f"Expected 1 POLYGON component, got {poly_count}",
        )


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

    The chord (0,3) creates a separation pair {0,3} splitting the
    graph into two 4-cycles and a degenerate bond.

    :return: A MultiGraph representing C6 plus a chord.
    """
    g: MultiGraph = MultiGraph()
    for i in range(6):
        g.add_edge(i, (i + 1) % 6)
    g.add_edge(0, 3)
    return g


def _make_k5() -> MultiGraph:
    """Build the complete graph K5 (10 edges, vertices 0-4).

    K5 is 4-connected, hence triconnected.

    :return: A MultiGraph representing K5.
    """
    g: MultiGraph = MultiGraph()
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(i, j)
    return g


def _make_petersen() -> MultiGraph:
    """Build the Petersen graph (10 vertices, 15 edges).

    The Petersen graph is 3-connected (triconnected).
    Outer 5-cycle: 0-1-2-3-4-0.
    Spokes: 0-5, 1-6, 2-7, 3-8, 4-9.
    Inner pentagram: 5-7, 7-9, 9-6, 6-8, 8-5.

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


def _make_three_k4_cliques() -> MultiGraph:
    """Build graph: 3 K4 cliques sharing poles {0, 1}.

    Vertices 0-7; poles are 0 and 1; each clique K4(0,1,a,b) adds
    the 6 edges among {0,1,a,b}.  The edge (0,1) appears 3 times.

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
    Expected: 4 components (1 BOND + 3 POLYGON).

    :return: A MultiGraph with three length-3 paths.
    """
    g: MultiGraph = MultiGraph()
    for a, b in [(2, 3), (4, 5), (6, 7)]:
        g.add_edge(0, a)
        g.add_edge(a, b)
        g.add_edge(b, 1)
    return g


class TestTriconnectedC5(unittest.TestCase):
    """Tests for triconnected decomposition of the 5-cycle C5."""

    def setUp(self) -> None:
        """Build the components for C5."""
        self.g: MultiGraph = _make_c5()
        """The C5 graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_single_polygon_component(self) -> None:
        """Test that C5 yields exactly 1 POLYGON component."""
        self.assertEqual(len(self.comps), 1)
        self.assertEqual(self.comps[0].type, ComponentType.POLYGON)

    def test_five_real_edges(self) -> None:
        """Test that the POLYGON from C5 contains 5 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 5)

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )


class TestTriconnectedC6(unittest.TestCase):
    """Tests for triconnected decomposition of the 6-cycle C6.

    Expected: 1 POLYGON component (the entire cycle).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for C6."""
        self.g: MultiGraph = _make_c6()
        """The C6 graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_single_polygon_component(self) -> None:
        """Test that C6 yields exactly 1 POLYGON component."""
        self.assertEqual(len(self.comps), 1)
        self.assertEqual(self.comps[0].type, ComponentType.POLYGON)

    def test_six_real_edges(self) -> None:
        """Test that the POLYGON from C6 contains 6 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 6)

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )


class TestTriconnectedC6Chord(unittest.TestCase):
    """Tests for triconnected decomposition of C6 plus chord (0,3).

    The chord (0,3) creates separation pair {0,3} yielding 3
    components: 2 POLYGON and 1 BOND.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for C6 with chord."""
        self.g: MultiGraph = _make_c6_with_chord()
        """The C6-with-chord graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for C6 plus chord."""
        _check_all_invariants(self, self.g, self.comps)

    def test_three_components(self) -> None:
        """Test that C6 plus chord produces exactly 3 components.

        The chord (0,3) creates separation pair {0,3} yielding
        2 POLYGON components and 1 BOND.
        """
        self.assertEqual(
            len(self.comps),
            3,
            f"C6+chord should have 3 components, "
            f"got {len(self.comps)}",
        )

    def test_two_polygon_components(self) -> None:
        """Test that C6 plus chord has 2 POLYGON components."""
        poly: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.POLYGON
        ]
        self.assertEqual(
            len(poly), 2,
            f"Expected 2 POLYGON components, got {len(poly)}",
        )

    def test_one_bond_component(self) -> None:
        """Test that C6 plus chord has 1 BOND component."""
        bond: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.BOND
        ]
        self.assertEqual(
            len(bond), 1,
            f"Expected 1 BOND component, got {len(bond)}",
        )


class TestTriconnectedK5(unittest.TestCase):
    """Tests for triconnected decomposition of the complete graph K5.

    K5 is 4-connected, so the entire graph is one TRICONNECTED
    component.
    """

    def setUp(self) -> None:
        """Build the components for K5."""
        self.g: MultiGraph = _make_k5()
        """The K5 graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_single_triconnected_component(self) -> None:
        """Test that K5 yields exactly 1 TRICONNECTED component."""
        self.assertEqual(len(self.comps), 1)
        self.assertEqual(
            self.comps[0].type, ComponentType.TRICONNECTED
        )

    def test_ten_real_edges(self) -> None:
        """Test that the TRICONNECTED component has 10 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 10)

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )


class TestTriconnectedPetersen(unittest.TestCase):
    """Tests for triconnected decomposition of the Petersen graph.

    The Petersen graph is 3-connected, so it yields a single
    TRICONNECTED component with all 15 edges.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for the Petersen graph."""
        self.g: MultiGraph = _make_petersen()
        """The Petersen graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_single_triconnected_component(self) -> None:
        """Test that the Petersen graph yields 1 TRICONNECTED."""
        self.assertEqual(len(self.comps), 1)
        self.assertEqual(
            self.comps[0].type, ComponentType.TRICONNECTED
        )

    def test_fifteen_real_edges(self) -> None:
        """Test that the TRICONNECTED component has 15 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 15)

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )


class TestTriconnectedThreeK4Cliques(unittest.TestCase):
    """Tests for decomposition of 3 K4 cliques sharing poles {0,1}.

    Expected: 4 components: 1 BOND (3-way parallel at {0,1}) and
    3 TRICONNECTED components (one per K4 clique).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for the three-K4-cliques graph."""
        self.g: MultiGraph = _make_three_k4_cliques()
        """The three-K4-cliques graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for three K4 cliques."""
        _check_all_invariants(self, self.g, self.comps)

    def test_four_components_total(self) -> None:
        """Test that the graph produces exactly 4 components.

        Expected: 1 BOND + 3 TRICONNECTED = 4 total.
        """
        self.assertEqual(
            len(self.comps),
            4,
            f"Expected 4 components, got {len(self.comps)}",
        )

    def test_three_triconnected_components(self) -> None:
        """Test that there are exactly 3 TRICONNECTED components."""
        tc: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.TRICONNECTED
        ]
        self.assertEqual(
            len(tc), 3,
            f"Expected 3 TRICONNECTED components, "
            f"got {len(tc)}",
        )

    def test_one_bond_component(self) -> None:
        """Test that there is exactly 1 BOND component."""
        bond: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.BOND
        ]
        self.assertEqual(
            len(bond), 1,
            f"Expected 1 BOND component, got {len(bond)}",
        )


class TestTriconnectedThreeLongPaths(unittest.TestCase):
    """Tests for decomposition of 3 length-3 paths between 0 and 1.

    Expected: 4 components: 1 BOND (3-way parallel at {0,1}) and
    3 POLYGON components (one per length-3 path).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for the three-long-paths graph."""
        self.g: MultiGraph = _make_three_long_paths()
        """The three-long-paths graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for three long paths."""
        _check_all_invariants(self, self.g, self.comps)

    def test_four_components_total(self) -> None:
        """Test that the graph produces exactly 4 components.

        Expected: 1 BOND + 3 POLYGON = 4 total.
        """
        self.assertEqual(
            len(self.comps),
            4,
            f"Expected 4 components, got {len(self.comps)}",
        )

    def test_three_polygon_components(self) -> None:
        """Test that there are exactly 3 POLYGON components."""
        poly: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.POLYGON
        ]
        self.assertEqual(
            len(poly), 3,
            f"Expected 3 POLYGON components, "
            f"got {len(poly)}",
        )

    def test_one_bond_component(self) -> None:
        """Test that there is exactly 1 BOND component."""
        bond: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.BOND
        ]
        self.assertEqual(
            len(bond), 1,
            f"Expected 1 BOND component, got {len(bond)}",
        )


def _make_k33() -> MultiGraph:
    """Build the complete bipartite graph K_{3,3} (9 edges).

    Vertices 0,1,2 on one side and 3,4,5 on the other.  K_{3,3}
    is 3-connected (triconnected) and non-planar.

    :return: A MultiGraph representing K_{3,3}.
    """
    g: MultiGraph = MultiGraph()
    for i in range(3):
        for j in range(3, 6):
            g.add_edge(i, j)
    return g


def _make_w4() -> MultiGraph:
    """Build the wheel graph W4: hub vertex 0, rim vertices 1-4.

    Edges: hub spokes (0,1),(0,2),(0,3),(0,4) and rim cycle
    (1,2),(2,3),(3,4),(4,1).  W4 is 3-connected.

    :return: A MultiGraph representing W4.
    """
    g: MultiGraph = MultiGraph()
    for i in range(1, 5):
        g.add_edge(0, i)
        g.add_edge(i, i % 4 + 1)
    return g


def _make_k3_doubled() -> MultiGraph:
    """Build K3 with each edge doubled (6 edges total).

    Each of the 3 triangle edges appears twice.  Expected: 4
    components (1 POLYGON + 3 BOND).

    :return: A MultiGraph representing K3 with doubled edges.
    """
    g: MultiGraph = MultiGraph()
    for u, v in [(1, 2), (2, 3), (1, 3)]:
        g.add_edge(u, v)
        g.add_edge(u, v)
    return g


def _make_four_parallel() -> MultiGraph:
    """Build a graph with 4 parallel edges between vertices 1 and 2.

    :return: A MultiGraph with four parallel edges.
    """
    g: MultiGraph = MultiGraph()
    for _ in range(4):
        g.add_edge(1, 2)
    return g


def _make_five_parallel() -> MultiGraph:
    """Build a graph with 5 parallel edges between vertices 1 and 2.

    :return: A MultiGraph with five parallel edges.
    """
    g: MultiGraph = MultiGraph()
    for _ in range(5):
        g.add_edge(1, 2)
    return g


def _make_three_long_paths_doubled() -> MultiGraph:
    """Build 3 length-3 paths between 0 and 1 with each edge doubled.

    All 9 edges of _make_three_long_paths() appear twice.  Expected:
    13 components (3 POLYGON + 10 BOND).

    :return: A MultiGraph with doubled length-3 paths.
    """
    g: MultiGraph = MultiGraph()
    for a, b in [(2, 3), (4, 5), (6, 7)]:
        for u, v in [(0, a), (a, b), (b, 1)]:
            g.add_edge(u, v)
            g.add_edge(u, v)
    return g


def _make_graph6_sage_docstring() -> MultiGraph:
    """Build the 13-vertex, 23-edge graph from graph6 'LlCG{O@?GBoMw?'.

    This biconnected graph has separation pairs yielding 12 split
    components (5 BOND + 2 TRICONNECTED + 5 POLYGON).

    :return: A MultiGraph with 13 vertices and 23 edges.
    """
    g: MultiGraph = MultiGraph()
    edges: list[tuple[int, int]] = [
        (0, 1), (1, 2), (0, 3), (2, 3), (3, 4), (4, 5),
        (3, 6), (4, 6), (5, 6), (0, 7), (4, 7), (7, 8),
        (8, 9), (7, 10), (8, 10), (9, 10), (0, 11),
        (7, 11), (8, 11), (9, 11), (0, 12), (1, 12),
        (2, 12),
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g


def _make_petersen_augmented_twice() -> MultiGraph:
    """Build Petersen graph with two rounds of path augmentation.

    Round 1: for each of the 15 Petersen edges (u,v), add path
    u-w1-w2-v alongside.  Round 2: for each of the 60 edges from
    round 1, add path alongside.  Result: 160 vertices, 240 edges.
    Expected: 136 components (60 BOND + 75 POLYGON + 1 TRICONNECTED).

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
    # Round 2: augment the 60 edges from round 1.
    round1_edges: list[tuple[Hashable, Hashable]] = [
        (e.u, e.v) for e in g.edges
    ]
    for u, v in round1_edges:
        g.add_edge(u, next_v)
        g.add_edge(next_v, next_v + 1)
        g.add_edge(next_v + 1, v)
        next_v += 2
    return g


class TestTriconnectedK33(unittest.TestCase):
    """Tests for triconnected decomposition of K_{3,3}.

    K_{3,3} is 3-connected, so it yields a single TRICONNECTED
    component.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for K_{3,3}."""
        self.g: MultiGraph = _make_k33()
        """The K_{3,3} graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for K_{3,3}."""
        _check_all_invariants(self, self.g, self.comps)

    def test_single_triconnected_component(self) -> None:
        """Test that K_{3,3} yields exactly 1 TRICONNECTED."""
        self.assertEqual(len(self.comps), 1)
        self.assertEqual(
            self.comps[0].type, ComponentType.TRICONNECTED
        )

    def test_nine_real_edges(self) -> None:
        """Test that the TRICONNECTED component has 9 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 9)

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )


class TestTriconnectedW4(unittest.TestCase):
    """Tests for triconnected decomposition of the wheel graph W4.

    W4 (hub + 4-rim) is 3-connected, yielding 1 TRICONNECTED.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for W4."""
        self.g: MultiGraph = _make_w4()
        """The W4 wheel graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for W4."""
        _check_all_invariants(self, self.g, self.comps)

    def test_single_triconnected_component(self) -> None:
        """Test that W4 yields exactly 1 TRICONNECTED component."""
        self.assertEqual(len(self.comps), 1)
        self.assertEqual(
            self.comps[0].type, ComponentType.TRICONNECTED
        )

    def test_eight_real_edges(self) -> None:
        """Test that the TRICONNECTED component has 8 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 8)

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )


class TestTriconnectedK3Doubled(unittest.TestCase):
    """Tests for triconnected decomposition of K3 with doubled edges.

    Each K3 edge appears twice.  Expected: 1 POLYGON (backbone) and
    3 BOND components (one per parallel pair).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for K3 with doubled edges."""
        self.g: MultiGraph = _make_k3_doubled()
        """The K3-doubled graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for K3 doubled."""
        _check_all_invariants(self, self.g, self.comps)

    def test_four_components(self) -> None:
        """Test exactly 4 components: 1 POLYGON + 3 BOND."""
        self.assertEqual(
            len(self.comps), 4,
            f"Expected 4 components, got {len(self.comps)}",
        )

    def test_one_polygon_three_bonds(self) -> None:
        """Test 1 POLYGON and 3 BOND components."""
        poly: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.POLYGON
        ]
        bond: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.BOND
        ]
        self.assertEqual(len(poly), 1, "Expected 1 POLYGON")
        self.assertEqual(len(bond), 3, "Expected 3 BOND")


class TestTriconnectedFourParallel(unittest.TestCase):
    """Tests for triconnected decomposition of 4 parallel edges."""

    def setUp(self) -> None:
        """Build the components for 4 parallel edges."""
        self.g: MultiGraph = _make_four_parallel()
        """The four-parallel-edges graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_single_bond_component(self) -> None:
        """Test that 4 parallel edges yield exactly 1 BOND."""
        self.assertEqual(len(self.comps), 1)
        self.assertEqual(
            self.comps[0].type, ComponentType.BOND
        )

    def test_four_real_edges(self) -> None:
        """Test that the BOND has 4 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 4)

    def test_total_real_edges(self) -> None:
        """Test that total real edge count equals input edge count."""
        self.assertEqual(
            _count_real_edges(self.comps), self.g.num_edges()
        )


class TestTriconnectedFiveParallel(unittest.TestCase):
    """Tests for triconnected decomposition of 5 parallel edges."""

    def setUp(self) -> None:
        """Build the components for 5 parallel edges."""
        self.g: MultiGraph = _make_five_parallel()
        """The five-parallel-edges graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_single_bond_component(self) -> None:
        """Test that 5 parallel edges yield exactly 1 BOND."""
        self.assertEqual(len(self.comps), 1)
        self.assertEqual(
            self.comps[0].type, ComponentType.BOND
        )

    def test_five_real_edges(self) -> None:
        """Test that the BOND has 5 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 5)


class TestTriconnectedThreeLongPathsDoubled(unittest.TestCase):
    """Tests for 3 length-3 paths with all edges doubled.

    Each of the 9 edges appears twice, yielding 13 components
    (3 POLYGON + 10 BOND).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for the doubled 3-paths graph."""
        self.g: MultiGraph = \
            _make_three_long_paths_doubled()
        """The doubled three-long-paths graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for doubled 3-paths."""
        _check_all_invariants(self, self.g, self.comps)

    def test_at_least_four_components(self) -> None:
        """Test that doubled 3-paths produces many components."""
        self.assertGreaterEqual(
            len(self.comps), 4,
            "Doubled 3-paths should have many components",
        )

    def test_has_polygon_and_bond(self) -> None:
        """Test that both POLYGON and BOND components exist."""
        poly: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.POLYGON
        ]
        bond: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.BOND
        ]
        self.assertGreaterEqual(
            len(poly), 1, "Need >= 1 POLYGON"
        )
        self.assertGreaterEqual(
            len(bond), 1, "Need >= 1 BOND"
        )


class TestTriconnectedSageDocstringGraph(unittest.TestCase):
    """Tests for the 13-vertex, 23-edge graph (graph6 'LlCG{O@?GBoMw?').

    This biconnected graph has multiple separation pairs and yields
    12 split components (5 BOND + 2 TRICONNECTED + 5 POLYGON).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for the 13-vertex docstring graph."""
        self.g: MultiGraph = _make_graph6_sage_docstring()
        """The SageMath docstring graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for the 13V/23E graph."""
        _check_all_invariants(self, self.g, self.comps)

    def test_at_least_two_components(self) -> None:
        """Test that the 13V/23E graph produces multiple components."""
        self.assertGreaterEqual(
            len(self.comps),
            2,
            "Graph has separation pairs and must have > 1 component",
        )


class TestTriconnectedPetersenAugmentedTwice(unittest.TestCase):
    """Tests for the doubly-augmented Petersen graph.

    Round 1: for each of the 15 Petersen edges (u,v), a parallel path
    u-w1-w2-v is added alongside.  Round 2: for each of the 60 edges
    from round 1, another parallel path is added.
    Result: 160 vertices, 240 edges, 136 components.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for the doubly-augmented Petersen."""
        self.g: MultiGraph = \
            _make_petersen_augmented_twice()
        """The doubly-augmented Petersen graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for doubly-aug. Petersen."""
        _check_all_invariants(self, self.g, self.comps)

    def test_136_total_components(self) -> None:
        """Test that the doubly-augmented Petersen yields 136 comps.

        Expected: 60 BOND + 75 POLYGON + 1 TRICONNECTED = 136 total.
        """
        self.assertEqual(
            len(self.comps),
            136,
            f"Doubly-augmented Petersen should have 136 components, "
            f"got {len(self.comps)}",
        )

    def test_one_triconnected(self) -> None:
        """Test that there is exactly 1 TRICONNECTED component."""
        tc: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.TRICONNECTED
        ]
        self.assertEqual(
            len(tc), 1,
            f"Expected 1 TRICONNECTED, got {len(tc)}",
        )

    def test_sixty_bonds(self) -> None:
        """Test that there are exactly 60 BOND components."""
        bonds: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.BOND
        ]
        self.assertEqual(
            len(bonds), 60,
            f"Expected 60 BOND, got {len(bonds)}",
        )

    def test_seventy_five_polygons(self) -> None:
        """Test that there are exactly 75 POLYGON components."""
        polys: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.POLYGON
        ]
        self.assertEqual(
            len(polys), 75,
            f"Expected 75 POLYGON, got {len(polys)}",
        )


class TestTriconnectedDiamondExact(unittest.TestCase):
    """Exact component-type tests for the diamond graph.

    The diamond has separation pair {2,3}.  Expected: exactly 3
    components: 2 POLYGON (the two triangular halves) and 1 BOND
    (the direct edge (2,3) with 2 virtual edges = P-node).
    """

    def setUp(self) -> None:
        """Build the components for the diamond graph."""
        self.g: MultiGraph = _make_diamond()
        """The diamond graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_exactly_three_components(self) -> None:
        """Test that the diamond produces exactly 3 components."""
        self.assertEqual(
            len(self.comps),
            3,
            f"Diamond should have 3 components, "
            f"got {len(self.comps)}",
        )

    def test_two_polygons_one_bond(self) -> None:
        """Test that diamond has 2 POLYGON and 1 BOND components."""
        poly: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.POLYGON
        ]
        bond: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.BOND
        ]
        self.assertEqual(len(poly), 2, "Expected 2 POLYGON")
        self.assertEqual(len(bond), 1, "Expected 1 BOND")


def _make_ladder() -> MultiGraph:
    """Build the 3-rung ladder graph (2x4 grid).

    Vertices 0-7.  Top row: 0-1-2-3, bottom row: 4-5-6-7,
    rungs: (0,4), (1,5), (2,6), (3,7).  Separation pairs
    {1,5} and {2,6} yield 5 components: 3 POLYGON + 2 BOND.

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


class TestTriconnectedLadder(unittest.TestCase):
    """Tests for triconnected decomposition of the 3-rung ladder graph.

    The ladder (2x4 grid) has separation pairs {1,5} and {2,6}.
    Expected: 5 components (3 POLYGON + 2 BOND).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for the ladder graph."""
        self.g: MultiGraph = _make_ladder()
        """The 3-rung ladder graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for the ladder graph."""
        _check_all_invariants(self, self.g, self.comps)

    def test_five_components(self) -> None:
        """Test that the ladder graph yields exactly 5 components."""
        self.assertEqual(
            len(self.comps),
            5,
            f"Expected 5 components, got {len(self.comps)}",
        )

    def test_three_polygons_two_bonds(self) -> None:
        """Test that the ladder has 3 POLYGON and 2 BOND components."""
        poly: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.POLYGON
        ]
        bond: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.BOND
        ]
        self.assertEqual(len(poly), 3, "Expected 3 POLYGON")
        self.assertEqual(len(bond), 2, "Expected 2 BOND")

    def test_real_edge_count(self) -> None:
        """Test that total real edge count equals 10."""
        self.assertEqual(
            _count_real_edges(self.comps),
            self.g.num_edges(),
        )


def _make_c7() -> MultiGraph:
    """Build the 7-cycle C7 (vertices 0-6).

    :return: A MultiGraph representing C7.
    """
    g: MultiGraph = MultiGraph()
    for i in range(7):
        g.add_edge(i, (i + 1) % 7)
    return g


class TestTriconnectedC7(unittest.TestCase):
    """Tests for triconnected decomposition of the 7-cycle C7.

    C7 is biconnected but not triconnected.  It yields a single
    POLYGON component with 7 real edges.
    """

    def setUp(self) -> None:
        """Build the components for C7."""
        self.g: MultiGraph = _make_c7()
        """The C7 graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for C7."""
        _check_all_invariants(self, self.g, self.comps)

    def test_single_component(self) -> None:
        """Test that C7 produces exactly 1 component."""
        self.assertEqual(len(self.comps), 1)

    def test_component_is_polygon(self) -> None:
        """Test that C7 yields a POLYGON component."""
        self.assertEqual(self.comps[0].type, ComponentType.POLYGON)

    def test_seven_real_edges(self) -> None:
        """Test that the POLYGON has 7 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
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


class TestTriconnectedC8(unittest.TestCase):
    """Tests for triconnected decomposition of the 8-cycle C8.

    C8 is biconnected but not triconnected.  It yields a single
    POLYGON component with 8 real edges.
    """

    def setUp(self) -> None:
        """Build the components for C8."""
        self.g: MultiGraph = _make_c8()
        """The C8 graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for C8."""
        _check_all_invariants(self, self.g, self.comps)

    def test_single_component(self) -> None:
        """Test that C8 produces exactly 1 component."""
        self.assertEqual(len(self.comps), 1)

    def test_component_is_polygon(self) -> None:
        """Test that C8 yields a POLYGON component."""
        self.assertEqual(self.comps[0].type, ComponentType.POLYGON)

    def test_eight_real_edges(self) -> None:
        """Test that the POLYGON has 8 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 8)


def _make_k23() -> MultiGraph:
    """Build the complete bipartite graph K_{2,3}.

    Vertices 0,1 (part A) and 2,3,4 (part B).
    Edges: all 6 pairs between parts.  K_{2,3} has vertex
    connectivity 2: removing {0,1} disconnects the graph.
    Each internal vertex x in {2,3,4} creates a path 0-x-1,
    yielding 4 components: 3 POLYGON + 1 BOND.

    Inspired by the SageMath ``spqr_tree`` test suite.

    :return: A MultiGraph representing K_{2,3}.
    """
    g: MultiGraph = MultiGraph()
    for a in [0, 1]:
        for b in [2, 3, 4]:
            g.add_edge(a, b)
    return g


class TestTriconnectedK23(unittest.TestCase):
    """Tests for triconnected decomposition of K_{2,3}.

    K_{2,3} has 5 vertices and 6 edges.  It has vertex
    connectivity 2 (separation pair {0,1}), yielding
    4 components: 3 POLYGON + 1 BOND.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for K_{2,3}."""
        self.g: MultiGraph = _make_k23()
        """The K_{2,3} graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for K_{2,3}."""
        _check_all_invariants(self, self.g, self.comps)

    def test_four_components(self) -> None:
        """Test that K_{2,3} produces exactly 4 components."""
        self.assertEqual(
            len(self.comps), 4,
            f"Expected 4 components, got {len(self.comps)}",
        )

    def test_three_polygons_one_bond(self) -> None:
        """Test that K_{2,3} has 3 POLYGON and 1 BOND."""
        poly: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.POLYGON
        ]
        bond: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.BOND
        ]
        self.assertEqual(len(poly), 3, "Expected 3 POLYGON")
        self.assertEqual(len(bond), 1, "Expected 1 BOND")

    def test_real_edge_count(self) -> None:
        """Test that total real edge count equals 6."""
        self.assertEqual(
            _count_real_edges(self.comps),
            self.g.num_edges(),
        )


def _make_w5() -> MultiGraph:
    """Build the wheel graph W5 (hub + 5-cycle, 6 vertices).

    Hub vertex 0 connected to rim vertices 1-5.
    W5 is 3-connected, yielding 1 TRICONNECTED with 10 edges.

    Inspired by the SageMath ``spqr_tree`` test suite.

    :return: A MultiGraph representing W5.
    """
    g: MultiGraph = MultiGraph()
    for i in range(1, 6):
        g.add_edge(i, i % 5 + 1)
    for i in range(1, 6):
        g.add_edge(0, i)
    return g


class TestTriconnectedW5(unittest.TestCase):
    """Tests for triconnected decomposition of the wheel W5.

    W5 has 6 vertices and 10 edges.  It is triconnected,
    so it yields a single TRICONNECTED component.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for W5."""
        self.g: MultiGraph = _make_w5()
        """The W5 wheel graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for W5."""
        _check_all_invariants(self, self.g, self.comps)

    def test_single_component(self) -> None:
        """Test that W5 produces exactly 1 component."""
        self.assertEqual(len(self.comps), 1)

    def test_component_is_triconnected(self) -> None:
        """Test that W5 yields a TRICONNECTED component."""
        self.assertEqual(
            self.comps[0].type, ComponentType.TRICONNECTED,
        )

    def test_ten_real_edges(self) -> None:
        """Test that the TRICONNECTED has 10 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 10)


def _make_w6() -> MultiGraph:
    """Build the wheel graph W6 (hub + 6-cycle, 7 vertices).

    Hub vertex 0 connected to rim vertices 1-6.
    W6 is 3-connected, yielding 1 TRICONNECTED with 12 edges.

    Inspired by the SageMath ``spqr_tree`` test suite.

    :return: A MultiGraph representing W6.
    """
    g: MultiGraph = MultiGraph()
    for i in range(1, 7):
        g.add_edge(i, i % 6 + 1)
    for i in range(1, 7):
        g.add_edge(0, i)
    return g


class TestTriconnectedW6(unittest.TestCase):
    """Tests for triconnected decomposition of the wheel W6.

    W6 has 7 vertices and 12 edges.  It is triconnected,
    so it yields a single TRICONNECTED component.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for W6."""
        self.g: MultiGraph = _make_w6()
        """The W6 wheel graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for W6."""
        _check_all_invariants(self, self.g, self.comps)

    def test_single_component(self) -> None:
        """Test that W6 produces exactly 1 component."""
        self.assertEqual(len(self.comps), 1)

    def test_component_is_triconnected(self) -> None:
        """Test that W6 yields a TRICONNECTED component."""
        self.assertEqual(
            self.comps[0].type, ComponentType.TRICONNECTED,
        )

    def test_twelve_real_edges(self) -> None:
        """Test that the TRICONNECTED has 12 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 12)


def _make_q3_cube() -> MultiGraph:
    """Build the Q3 cube graph (3-dimensional hypercube).

    8 vertices (0-7), 12 edges.  The cube graph is 3-connected,
    yielding 1 TRICONNECTED component with 12 edges.

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


class TestTriconnectedQ3Cube(unittest.TestCase):
    """Tests for triconnected decomposition of the Q3 cube graph.

    The cube (Q3) has 8 vertices and 12 edges.  It is
    3-connected, so it yields a single TRICONNECTED component.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for the Q3 cube."""
        self.g: MultiGraph = _make_q3_cube()
        """The Q3 cube graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for Q3 cube."""
        _check_all_invariants(self, self.g, self.comps)

    def test_single_component(self) -> None:
        """Test that Q3 cube produces exactly 1 component."""
        self.assertEqual(len(self.comps), 1)

    def test_component_is_triconnected(self) -> None:
        """Test that Q3 cube yields a TRICONNECTED component."""
        self.assertEqual(
            self.comps[0].type, ComponentType.TRICONNECTED,
        )

    def test_twelve_real_edges(self) -> None:
        """Test that the TRICONNECTED has 12 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 12)


def _make_octahedron() -> MultiGraph:
    """Build the octahedron graph (6 vertices, 12 edges).

    The octahedron is the complete tripartite graph K_{2,2,2}.
    It is 4-connected, yielding 1 TRICONNECTED component.

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


class TestTriconnectedOctahedron(unittest.TestCase):
    """Tests for triconnected decomposition of the octahedron.

    The octahedron has 6 vertices and 12 edges.  It is 4-connected,
    so it yields a single TRICONNECTED component.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for the octahedron."""
        self.g: MultiGraph = _make_octahedron()
        """The octahedron graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for octahedron."""
        _check_all_invariants(self, self.g, self.comps)

    def test_single_component(self) -> None:
        """Test that octahedron produces exactly 1 component."""
        self.assertEqual(len(self.comps), 1)

    def test_component_is_triconnected(self) -> None:
        """Test that octahedron yields a TRICONNECTED component."""
        self.assertEqual(
            self.comps[0].type, ComponentType.TRICONNECTED,
        )

    def test_twelve_real_edges(self) -> None:
        """Test that the TRICONNECTED has 12 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 12)


def _make_k4_one_doubled() -> MultiGraph:
    """Build K4 with one edge doubled.

    K4 has 6 edges.  One edge (say 1-2) is doubled, giving 7
    edges total.  The doubled edge creates a separation pair
    {1,2}, yielding 2 components: 1 BOND (the doubled edge) and
    1 TRICONNECTED (the K4 skeleton with a virtual edge).

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


class TestTriconnectedK4OneDoubled(unittest.TestCase):
    """Tests for triconnected decomposition of K4 + 1 doubled edge.

    K4 with one edge doubled has 7 edges.  The doubled edge
    creates separation pair {1,2}, yielding 2 components:
    1 BOND and 1 TRICONNECTED.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for K4 + 1 doubled edge."""
        self.g: MultiGraph = _make_k4_one_doubled()
        """The K4-one-doubled graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for K4+doubled."""
        _check_all_invariants(self, self.g, self.comps)

    def test_two_components(self) -> None:
        """Test that K4+doubled produces exactly 2 components."""
        self.assertEqual(
            len(self.comps), 2,
            f"Expected 2 components, got {len(self.comps)}",
        )

    def test_one_bond_one_triconnected(self) -> None:
        """Test that K4+doubled has 1 BOND and 1 TRICONNECTED."""
        bond: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.BOND
        ]
        tri: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.TRICONNECTED
        ]
        self.assertEqual(len(bond), 1, "Expected 1 BOND")
        self.assertEqual(
            len(tri), 1, "Expected 1 TRICONNECTED",
        )

    def test_real_edge_count(self) -> None:
        """Test that total real edge count equals 7."""
        self.assertEqual(
            _count_real_edges(self.comps),
            self.g.num_edges(),
        )


def _make_mobius_kantor() -> MultiGraph:
    """Build the Mobius-Kantor graph GP(8,3) (16 verts, 24 edges).

    The Mobius-Kantor graph is the generalized Petersen graph
    GP(8,3).  It is 3-regular and 3-connected, yielding 1
    TRICONNECTED component with 24 real edges.

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


class TestTriconnectedMobiusKantor(unittest.TestCase):
    """Tests for triconnected decomposition of Mobius-Kantor graph.

    The Mobius-Kantor graph has 16 vertices and 24 edges.  It is
    3-connected, so it yields a single TRICONNECTED component.

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for the Mobius-Kantor graph."""
        self.g: MultiGraph = _make_mobius_kantor()
        """The Mobius-Kantor graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for Mobius-Kantor."""
        _check_all_invariants(self, self.g, self.comps)

    def test_single_component(self) -> None:
        """Test that Mobius-Kantor produces exactly 1 component."""
        self.assertEqual(len(self.comps), 1)

    def test_component_is_triconnected(self) -> None:
        """Test that Mobius-Kantor yields TRICONNECTED."""
        self.assertEqual(
            self.comps[0].type, ComponentType.TRICONNECTED,
        )

    def test_twenty_four_real_edges(self) -> None:
        """Test that the TRICONNECTED has 24 real edges."""
        real: list[Edge] = [
            e for e in self.comps[0].edges if not e.virtual
        ]
        self.assertEqual(len(real), 24)


def _make_petersen_augmented() -> MultiGraph:
    """Build the Petersen graph with each edge augmented by a path.

    For each of the 15 Petersen edges (u,v), two intermediate
    vertices w1 and w2 are added and a path u-w1-w2-v is inserted
    alongside the original edge.  Result: 40 vertices, 60 edges.
    Expected: 31 components (15 BOND + 15 POLYGON + 1 TRICONNECTED).

    Inspired by the SageMath ``spqr_tree`` test suite.

    :return: The augmented Petersen multigraph.
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
        w1: int = next_v
        w2: int = next_v + 1
        next_v += 2
        g.add_edge(u, w1)
        g.add_edge(w1, w2)
        g.add_edge(w2, v)
    return g


class TestTriconnectedPetersenAugmented(unittest.TestCase):
    """Tests for triconnected decomposition of augmented Petersen.

    Each Petersen edge (u,v) gets a parallel path u-w1-w2-v.
    Expected: 31 components (15 BOND + 15 POLYGON + 1 TRICONN).

    Inspired by the SageMath ``spqr_tree`` test suite.
    """

    def setUp(self) -> None:
        """Build the components for the augmented Petersen."""
        self.g: MultiGraph = _make_petersen_augmented()
        """The augmented Petersen graph under test."""
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)
        """The triconnected split components."""

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants for aug. Petersen."""
        _check_all_invariants(self, self.g, self.comps)

    def test_thirty_one_components(self) -> None:
        """Test that augmented Petersen has 31 components."""
        self.assertEqual(
            len(self.comps), 31,
            f"Expected 31 components, got {len(self.comps)}",
        )

    def test_one_triconnected(self) -> None:
        """Test that there is exactly 1 TRICONNECTED component."""
        tri: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.TRICONNECTED
        ]
        self.assertEqual(
            len(tri), 1,
            f"Expected 1 TRICONNECTED, got {len(tri)}",
        )

    def test_fifteen_bonds(self) -> None:
        """Test that there are exactly 15 BOND components."""
        bonds: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.BOND
        ]
        self.assertEqual(
            len(bonds), 15,
            f"Expected 15 BOND, got {len(bonds)}",
        )

    def test_fifteen_polygons(self) -> None:
        """Test that there are exactly 15 POLYGON components."""
        polys: list[TriconnectedComponent] = [
            c for c in self.comps
            if c.type == ComponentType.POLYGON
        ]
        self.assertEqual(
            len(polys), 15,
            f"Expected 15 POLYGON, got {len(polys)}",
        )

    def test_real_edge_count(self) -> None:
        """Test that total real edge count equals 60."""
        self.assertEqual(
            _count_real_edges(self.comps),
            self.g.num_edges(),
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


class TestTriconnectedWikimediaSpqr(unittest.TestCase):
    """Tests for triconnected decomposition of the Wikimedia
    SPQR example."""

    def setUp(self) -> None:
        """Set up graph and triconnected components."""
        self.g: MultiGraph = _make_wikimedia_spqr()
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)

    def test_component_count(self) -> None:
        """Test that there are exactly 5 components."""
        self.assertEqual(len(self.comps), 5)

    def test_triconnected_count(self) -> None:
        """Test that there are exactly 3 TRICONNECTED."""
        n: int = sum(
            1 for c in self.comps
            if c.type == ComponentType.TRICONNECTED
        )
        self.assertEqual(n, 3)

    def test_polygon_count(self) -> None:
        """Test that there is exactly 1 POLYGON."""
        n: int = sum(
            1 for c in self.comps
            if c.type == ComponentType.POLYGON
        )
        self.assertEqual(n, 1)

    def test_bond_count(self) -> None:
        """Test that there is exactly 1 BOND."""
        n: int = sum(
            1 for c in self.comps
            if c.type == ComponentType.BOND
        )
        self.assertEqual(n, 1)

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants."""
        _check_all_invariants(self, self.g, self.comps)


class TestTriconnectedRpstFig1a(unittest.TestCase):
    """Tests for triconnected decomposition of RPST Fig 1(a)."""

    def setUp(self) -> None:
        """Set up graph and triconnected components."""
        self.g: MultiGraph = _make_rpst_fig1a()
        self.comps: list[TriconnectedComponent] = \
            find_triconnected_components(self.g)

    def test_component_count(self) -> None:
        """Test that there are exactly 10 components."""
        self.assertEqual(len(self.comps), 10)

    def test_triconnected_count(self) -> None:
        """Test that there is exactly 1 TRICONNECTED."""
        n: int = sum(
            1 for c in self.comps
            if c.type == ComponentType.TRICONNECTED
        )
        self.assertEqual(n, 1)

    def test_polygon_count(self) -> None:
        """Test that there are exactly 8 POLYGON."""
        n: int = sum(
            1 for c in self.comps
            if c.type == ComponentType.POLYGON
        )
        self.assertEqual(n, 8)

    def test_bond_count(self) -> None:
        """Test that there is exactly 1 BOND."""
        n: int = sum(
            1 for c in self.comps
            if c.type == ComponentType.BOND
        )
        self.assertEqual(n, 1)

    def test_all_invariants(self) -> None:
        """Test all decomposition invariants."""
        _check_all_invariants(self, self.g, self.comps)


def _make_cut_vertex_graph() -> MultiGraph:
    """Build a graph with a cut vertex.

    Graph: 1-2-3 triangle connected to 3-4-5 triangle via
    shared vertex 3 (the cut vertex).

    :return: A MultiGraph with a cut vertex at vertex 3.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(1, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 5)
    g.add_edge(3, 5)
    return g


def _make_disconnected_graph() -> MultiGraph:
    """Build a disconnected graph with two components.

    Component 1: edge 1-2.
    Component 2: edge 3-4.

    :return: A disconnected MultiGraph.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(1, 2)
    g.add_edge(3, 4)
    return g


def _make_path_graph() -> MultiGraph:
    """Build a path graph 1-2-3 (not biconnected).

    Vertex 2 is a cut vertex (removing it disconnects 1 and 3).

    :return: A MultiGraph representing a path.
    """
    g: MultiGraph = MultiGraph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    return g


class TestBiconnectivityCheck(unittest.TestCase):
    """Tests that non-biconnected graphs raise ValueError."""

    def test_cut_vertex_raises(self) -> None:
        """Test that a graph with a cut vertex raises ValueError."""
        g: MultiGraph = _make_cut_vertex_graph()
        with self.assertRaises(ValueError) as ctx:
            find_triconnected_components(g)
        self.assertIn("cut vertex", str(ctx.exception))

    def test_disconnected_raises(self) -> None:
        """Test that a disconnected graph raises ValueError."""
        g: MultiGraph = _make_disconnected_graph()
        with self.assertRaises(ValueError) as ctx:
            find_triconnected_components(g)
        self.assertIn("not connected", str(ctx.exception))

    def test_path_raises(self) -> None:
        """Test that a path graph (has cut vertex) raises ValueError."""
        g: MultiGraph = _make_path_graph()
        with self.assertRaises(ValueError) as ctx:
            find_triconnected_components(g)
        self.assertIn("cut vertex", str(ctx.exception))

    def test_single_vertex_no_edges(self) -> None:
        """Test that a single vertex with no edges returns empty."""
        g: MultiGraph = MultiGraph()
        g.add_vertex(1)
        comps: list[TriconnectedComponent] = \
            find_triconnected_components(g)
        self.assertEqual(comps, [])

    def test_biconnected_graph_ok(self) -> None:
        """Test that a biconnected graph does not raise."""
        g: MultiGraph = _make_k3()
        comps: list[TriconnectedComponent] = \
            find_triconnected_components(g)
        self.assertEqual(len(comps), 1)
