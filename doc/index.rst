.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

neurosynchro
============

*Neurosynchro* is a small Python package for creating and using neural
networks to quickly approximate the coefficients needed for fully-polarized
synchrotron radiative transfer. It builds on the `Keras <https://keras.io/>`_
deep learning library and is licensed under the `MIT License
<https://github.com/pkgw/neurosynchro/blob/master/LICENSE>`_.

This documentation refers to version |version| of *neurosynchro*.

- :ref:`Take me to the tutorial where I get to train some neural networks!
  <download-training-set>` (The training can take a few hours if you’re on a
  laptop, though.)

- :ref:`Take me to the tutorial where I just calculate some coefficients!
  <download-trained-networks>` (This tutorial is much quicker … maybe too
  quick.)


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
an electron temperature or an energy power law index. You also have a code
such as `grtrans <https://github.com/jadexter/grtrans>`_ that can perform your
radiative transfer integration in the Stokes basis.

The problem is that your synchrotron calculator code is very precise but it’s
just *too slow* for your needs: you need to do lots of integrations for
arbitrary input parameters, and things are taking too long.

Enter *neurosynchro*. This package provides a toolkit for training neural
networks to quickly approximate the coefficients you need. Once you’ve trained
your network, you can get very good approximations of the necessary
coefficients really quickly — something like 10,000 times faster than the
detailed calculations.


Installation
============

.. toctree::
   :maxdepth: 1

   installation.rst


.. _how-to-use:

Tutorial, AKA, How to Use It
============================

There are three main ways that you might use *neurosynchro*:

1. You might have the data files for a useful, trained neural network on hand.
   In this case, all you need to do is :ref:`load it up and use it
   <download-trained-networks>`. We have provided sample trained neural networks for
   download — the above page walks you through downloading them. So, this is
   the quickest way to get started.

2. You might have a training set on hand, but not have created your neural
   networks. In this case, you need to go through the process of
   :ref:`training a new set of neural networks <download-training-set>`.
   Once again, an example training set is available for download. The training
   takes a few dozen core-hours of CPU time, so it can take a little while.

3. Finally, if you are investigating a new or unusual region of parameter
   space, you might not even have an existing training set to work with. In
   this case you will need to start at the very beginning by :ref:`making your
   own training set <make-training-set>`. How you do so is beyond the purview
   of *neurosynchro*, but in the above pages we document the format that the
   training data must obey.

Or, to break down the various steps in roughly sequential order:

.. toctree::
   :maxdepth: 2

   make-training-set.rst
   download-training-set.rst
   transform-training-set.rst
   specify-parameter-transformations.rst
   precalculate-domain-and-range.rst
   train-networks.rst
   download-trained-networks.rst
   test-problems.rst
   use-networks.rst


Python API Reference
====================

In most cases, the only part of the Python API that you will care about is the
:class:`neurosynchro.impl.PhysicalApproximator` class — this is the class that
loads up a neural network and computes coefficients for you. The rest of
*neurosynchro*’s functionality is best accessed through the command-line
interface ``neurosynchro``, rather than directly using the Python code. As
such, many internal parts of the code are not documented.

.. toctree::
   :maxdepth: 2

   api/index.rst
   api/impl.rst
   api/grtrans.rst


Contributing to the Project
===========================

Contributions are welcome! See the `CONTRIBUTING.md
<https://github.com/pkgw/neurosynchro/blob/master/CONTRIBUTING.md>`_ file
attached to `the project’s GitHub repository
<https://github.com/pkgw/neurosynchro>`_. The goal is to run the project with
standard open-source best practices. We do wish to call your attention to the
`code of conduct
<https://github.com/pkgw/neurosynchro/blob/master/CODE_OF_CONDUCT.md>`_,
however. Participation in the project is contingent on abiding by the code in
both letter and spirit.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
