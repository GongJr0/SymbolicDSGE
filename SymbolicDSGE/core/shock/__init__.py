"""Shock specification, resolution, and drawing.

A spec is authored against :class:`Shock` (``generators``), resolved against a
compiled model into a :class:`ShockPlan` (``spec``), and drawn from that plan
(``plan``). The split is what lets one resolution serve many draws: everything
the model and the spec fix between them is paid once, and only the seed varies.

Only :class:`Shock` is reachable from here; it is the one name a user authors
against. Everything else is internal to the library and is imported from the
module that defines it.
"""

from .generators import Shock

__all__ = ["Shock"]
