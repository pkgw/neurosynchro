.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

:py:mod:`neurosynchro`: the main module
=======================================

.. automodule:: neurosynchro
   :synopsis: train and use neural networks to approximate polarized synchrotron
              radiative transfer coefficients

.. currentmodule:: neurosynchro

The top-level module of *neurosynchro* does not actually include any
neural-network functionality: such functionality requires loading the
:mod:`keras` module, which is very slow to initialize. Instead, this module
contains generic code for dealing with training set data and radiative
transfer coefficients:

* :ref:`loading-training-set-data`
* :ref:`working-with-rt-coefficients`
* :ref:`mappings`


.. _loading-training-set-data:

Loading Training Set Data
-------------------------

.. autofunction:: basic_load


.. _working-with-rt-coefficients:

Working With Radiative Transfer Coefficients
--------------------------------------------

.. autofunction:: detrivialize_stokes_basis


.. _mappings:

Mappings
--------

.. autoclass:: Mapping

   .. automethod:: from_info_and_samples
   .. automethod:: from_dict
   .. automethod:: phys_to_norm
   .. automethod:: norm_to_phys
   .. automethod:: to_dict


.. autofunction:: mapping_from_info_and_samples
.. autofunction:: mapping_from_dict
