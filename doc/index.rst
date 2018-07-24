.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

neurosynchro
============

*Neurosynchro* is a small Python package for creating and using neural
networks to quickly approximate the coefficients needed for fully-polarized
synchrotron radiative transfer. It builds on the `Keras <https://keras.io/>`_
deep learning library.

This documentation refers to version |version| of *neurosynchro*.


Who It’s For
============

*Neurosynchro* solves a very specific problem.

*You* are a person that wants to perform some `numerical radiative transfer
<https://en.wikipedia.org/wiki/Radiative_transfer>`_ of `synchrotron radiation
<https://en.wikipedia.org/wiki/Synchrotron_radiation>`_ in the `Stokes basis
<https://en.wikipedia.org/wiki/Stokes_parameters>`_. In this problem, for each
point in space you need to calculate eight matrix coefficients: three numbers
describing emission (total intensity, linear polarization, and circular
polarization), three describing absorption, and two “Faraday terms” (rotation
of linear polarization, conversion of circular and linear).

You have a code, such as `Rimphony <https://github.com/pkgw/rimphony/>`_ or
`Symphony <https://github.com/AFD-Illinois/symphony>`_, that can calculate
these numbers for you as a function of some input physical parameters such as
an electron temperature or an energy power law index.

Your code is very precise but it’s just *too slow* for your needs: you need to
do lots of calculations for lots of input parameters, and things are taking
too long.

Enter *neurosynchro*. This package provides a toolkit for training neural
networks to quickly approximate the coefficients you need. Once you’ve trained
your network, you can get very good approximations of the necessary
coefficients really quickly — something like 10,000 times faster than the
detailed calculations.


How to Use It
=============

If *neurosynchro* is the tool you need, here are the steps to using it:

.. toctree::
   :maxdepth: 2

   make-training-set.rst
   train-networks.rst
   use-networks.rst


Python API Reference
====================


.. toctree::
   :maxdepth: 2

   api/index.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
