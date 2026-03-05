========
spqrtree
========


Description
===========

**spqrtree** is a pure Python implementation of the SPQR-tree data
structure for biconnected graphs.  It decomposes a biconnected graph
into its triconnected components (S, P, Q, and R nodes) and organizes
them as a tree.

SPQR-trees are a classical tool in graph theory, widely used for
planarity testing, graph drawing, and network analysis.

The implementation is based on the Gutwenger & Mutzel (2001) linear-time
algorithm, with corrections to Hopcroft & Tarjan (1973), and follows the
SPQR-tree data structure defined by Di Battista & Tamassia (1996).

Features:

- Pure Python --- no compiled extensions or external dependencies.
- Handles multigraphs (parallel edges between the same vertex pair).
- Simple ``dict``-based input for quick prototyping.
- Typed package with PEP 561 support.
- Requires Python 3.10 or later.


Installation
============

You can install spqrtree with ``pip``:

::

    pip install spqrtree

You may also install the latest source from the
`spqrtree GitHub repository`_.

::

    pip install git+https://github.com/imacat/spqrtree.git


Quick Start
===========

.. code-block:: python

    from spqrtree import SPQRTree, NodeType

    # K4 complete graph
    graph = {
        1: [2, 3, 4],
        2: [1, 3, 4],
        3: [1, 2, 4],
        4: [1, 2, 3],
    }
    tree = SPQRTree(graph)
    print(tree.root.type)     # NodeType.R
    print(len(tree.nodes()))  # number of SPQR-tree nodes


Documentation
=============

Refer to the `documentation on Read the Docs`_.


Change Log
==========

Refer to the `change log`_.


References
==========

- C. Gutwenger and P. Mutzel, "A Linear Time Implementation of
  SPQR-Trees," *Graph Drawing (GD 2000)*, LNCS 1984, pp. 77--90,
  2001. `doi:10.1007/3-540-44541-2_8`_

- J. E. Hopcroft and R. E. Tarjan, "Dividing a Graph into
  Triconnected Components," *SIAM Journal on Computing*, 2(3),
  pp. 135--158, 1973. `doi:10.1137/0202012`_

- G. Di Battista and R. Tamassia, "On-Line Planarity Testing,"
  *SIAM Journal on Computing*, 25(5), pp. 956--997, 1996.
  `doi:10.1137/S0097539794280736`_


Acknowledgments
===============

This project was written from scratch in pure Python but drew
inspiration from the `SageMath`_ project.  The SPQR-tree
implementation in SageMath served as a valuable reference for
both its implementation approach and its comprehensive test
cases.

Development was assisted by `Claude Code`_ (Anthropic).


Copyright
=========

 Copyright (c) 2026 imacat.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.


Authors
=======

| imacat
| imacat@mail.imacat.idv.tw
| 2026/3/4

.. _spqrtree GitHub repository: https://github.com/imacat/spqrtree
.. _documentation on Read the Docs: https://spqrtree.readthedocs.io
.. _change log: https://spqrtree.readthedocs.io/en/latest/changelog.html
.. _doi\:10.1007/3-540-44541-2_8: https://doi.org/10.1007/3-540-44541-2_8
.. _doi\:10.1137/0202012: https://doi.org/10.1137/0202012
.. _doi\:10.1137/S0097539794280736: https://doi.org/10.1137/S0097539794280736
.. _SageMath: https://www.sagemath.org/
.. _Claude Code: https://claude.com/claude-code
