.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

.. _download-trained-networks:

Download Some Pre-Trained Neural Network Data
=============================================

One nice feature of the *neurosynchro* model is that the data files
representing a trained neural network are quite compact. Once a network has
been trained for your particular problem space, it’s easy and quick to use it.

The tutorial sequence starting with :ref:`download-training-set` describes how
to obtain a training set and train the neural nets on your own machine. This
step is essential in the (quite common) circumstance that you need a network
whose parameter ranges are tuned to your particular application. But here,
we’ll just download pre-trained neural network files.

All you need to do is download and unpack the bzipped *tar* archive
`rimphony_powerlaw_s5-5e7_p1.5-7_nndata.tar.bz2
<https://zenodo.org/record/1341364/files/rimphony_powerlaw_s5-5e7_p1.5-7_nndata.tar.bz2>`_,
which is archived on `Zenodo <https://zenodo.org/>`_ under DOI
`10.5281/zenodo.1341364 <https://doi.org/10.5281/zenodo.1341364>`_. It’s about
160 kiB. This archive contains the data needed to do synchrotron radiative
transfer if your particle distribution is a power-law in energy, isotropic in
pitch angle, the power-law indices range between 1.5 and 7, and the relevant
harmonic numbers are between 5 and 50,000,000.

Unpacking this archive will create a directory named
``rimphony_powerlaw_s5-5e7_p1.5-7``. Remaining steps working from the terminal
will assume that you’re working from inside this directory, e.g.::

  $ cd rimphony_powerlaw_s5-5e7_p1.5-7

**Next**: :ref:`run some test problems! <run-test-problems>`
