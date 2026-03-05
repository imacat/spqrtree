Introduction
============

What is an SPQR-Tree?
---------------------

An SPQR-tree is a tree data structure that represents the decomposition
of a biconnected graph into its triconnected components.  Each node of
the tree corresponds to one of four types:

- **S-node** (series): a simple cycle.
- **P-node** (parallel): a bundle of parallel edges between two poles.
- **Q-node**: a single real edge (degenerate case).
- **R-node** (rigid): a 3-connected subgraph that cannot be further
  decomposed.

SPQR-trees are widely used in graph drawing, planarity testing,
and network reliability analysis.


Features
--------

- Pure Python --- no compiled extensions or external dependencies.
- Handles multigraphs (parallel edges between the same vertex pair).
- Implements the Gutwenger & Mutzel (2001) algorithm with corrections
  to Hopcroft & Tarjan (1973).
- Simple ``dict``-based input for quick prototyping.
- Typed package with :pep:`561` support.


Installation
------------

Install from `PyPI <https://pypi.org/project/spqrtree/>`_ with pip:

.. code-block:: bash

   pip install spqrtree

Or install the latest development version from GitHub:

.. code-block:: bash

   pip install git+https://github.com/imacat/spqrtree.git

Requires Python 3.10 or later.


Quick Start
-----------

Build an SPQR-tree from an adjacency-list dictionary:

.. code-block:: python

   from spqrtree import SPQRTree

   # A simple diamond graph (K4 minus one edge)
   graph = {
       1: [2, 3, 4],
       2: [1, 3, 4],
       3: [1, 2, 4],
       4: [1, 2, 3],
   }
   tree = SPQRTree(graph)
   print(tree.root.type)    # NodeType.R
   print(len(tree.nodes()))  # number of SPQR-tree nodes

The input dictionary maps each vertex to its list of neighbors.  For
each pair ``(u, v)`` where ``u < v``, one edge is added.


Usage Guide
-----------

Inspecting the Tree
~~~~~~~~~~~~~~~~~~~

The :class:`~spqrtree.SPQRTree` object exposes two main attributes:

- :attr:`~spqrtree.SPQRTree.root` --- the root
  :class:`~spqrtree.SPQRNode` of the tree.
- :meth:`~spqrtree.SPQRTree.nodes` --- all nodes in BFS order.

.. code-block:: python

   from spqrtree import SPQRTree, NodeType

   graph = {
       0: [1, 2],
       1: [0, 2, 3],
       2: [0, 1, 3],
       3: [1, 2],
   }
   tree = SPQRTree(graph)

   for node in tree.nodes():
       print(node.type, node.poles)

Each :class:`~spqrtree.SPQRNode` has the following attributes:

- ``type`` --- a :class:`~spqrtree.NodeType` enum (``S``, ``P``,
  ``Q``, or ``R``).
- ``skeleton`` --- the skeleton graph containing the real and virtual
  edges of the component.
- ``poles`` --- the two vertices shared with the parent component.
- ``parent`` --- the parent node (``None`` for the root).
- ``children`` --- the list of child nodes.


Understanding Node Types
~~~~~~~~~~~~~~~~~~~~~~~~

**S-node (series):**
Represents a cycle.  The skeleton is a simple polygon whose edges
alternate between real edges and virtual edges leading to children.

**P-node (parallel):**
Represents parallel edges between two pole vertices.  The skeleton
contains three or more edges (real and/or virtual) between the same
pair of poles.

**R-node (rigid):**
Represents a 3-connected component that cannot be further decomposed
by any separation pair.  The skeleton is a 3-connected graph.

**Q-node:**
Represents a single real edge.  Q-nodes appear as leaves of the tree.


Using a MultiGraph
~~~~~~~~~~~~~~~~~~

For more control, build a
:class:`~spqrtree._graph.MultiGraph` directly:

.. code-block:: python

   from spqrtree._graph import MultiGraph
   from spqrtree import SPQRTree

   g = MultiGraph()
   g.add_edge(0, 1)
   g.add_edge(1, 2)
   g.add_edge(2, 0)
   g.add_edge(1, 3)
   g.add_edge(3, 2)

   tree = SPQRTree(g)


References
----------

The implementation is based on the following papers:

- J. Hopcroft and R. Tarjan, "Dividing a graph into triconnected
  components," *SIAM Journal on Computing*, vol. 2, no. 3,
  pp. 135--158, 1973.
  `doi:10.1137/0202012 <https://doi.org/10.1137/0202012>`_

- G. Di Battista and R. Tamassia, "On-line planarity testing,"
  *SIAM Journal on Computing*, vol. 25, no. 5, pp. 956--997, 1996.
  `doi:10.1137/S0097539794280736
  <https://doi.org/10.1137/S0097539794280736>`_

- C. Gutwenger and P. Mutzel, "A linear time implementation of
  SPQR-trees," *Proc. 8th International Symposium on Graph Drawing
  (GD 2000)*, LNCS 1984, pp. 77--90, Springer, 2001.
  `doi:10.1007/3-540-44541-2_8
  <https://doi.org/10.1007/3-540-44541-2_8>`_


Acknowledgments
---------------

The test suite was validated against the SPQR-tree implementation in
`SageMath <https://www.sagemath.org/>`_, which served as the reference
for verifying correctness of the decomposition results.
