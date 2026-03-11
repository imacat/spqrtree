Change Log
==========


Version 0.1.1
-------------

Released 2026/3/11

- ``find_triconnected_components()`` now raises ``ValueError`` when
  the input graph is not biconnected (disconnected or has a cut
  vertex).  Previously, non-biconnected input was silently accepted
  and produced incorrect results.
- Use public API imports in documentation examples.


Version 0.1.0
-------------

Released 2026/3/5

Initial release.
