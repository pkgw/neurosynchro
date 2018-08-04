.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

:py:mod:`neurosynchro.impl`: the neural-network implementation
==============================================================

.. automodule:: neurosynchro.impl
   :synopsis: load and use *neurosynchro*’s neural networks

.. currentmodule:: neurosynchro.impl

This module contains :class:`PhysicalApproximator`, the class that uses
prebuild :mod:`keras` neural networks to rapidly approximate radiative
transfer coefficients.


.. autoclass:: PhysicalApproximator

   .. automethod:: compute_all_nontrivial
