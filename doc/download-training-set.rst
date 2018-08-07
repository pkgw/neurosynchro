.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

.. _download-training-set:

Download a Sample Training Set
==============================

In order to train a neural network, you need something to train it on!

In *neurosynchro*, the training data are a bunch of samples of some detailed
synchrotron calculation. That is: given a choice of some input parameters
(e.g., a harmonic number), the detailed calculation produces eight output
parameters (the Stokes radiative transfer coefficients). The exact number of
input parameters can vary depending on what particle distribution is being
modeled. *Neurosynchro* needs a lot of samples of this function to develop a
good approximation to it.

The page :ref:`make-training-set` describes how such a training set should be
parametrized and formatted. Here, we’ll just download a training set and start
using it without worrying about the details.

.. note:: The training set is about a 300 MB download. It expands to a 700 MB
          directory, which we will soon reprocess into another directory
          that’s also about 700 MB.

First, download the example training data set. It is archived on `Zenodo
<https://zenodo.org/>`_ under DOI `10.5281/zenodo.1341154
<https://doi.org/10.5281/zenodo.1341154>`_. The main data file is a bzipped
*tar* archive named `rimphony_powerlaw_s5-5e7_p1.5-7.tar.bz2
<https://zenodo.org/record/1341154/files/rimphony_powerlaw_s5-5e7_p1.5-7.tar.bz2>`_.
The following shell command should download it::

  $ curl -fsSL https://zenodo.org/record/1341154/files/rimphony_powerlaw_s5-5e7_p1.5-7.tar.bz2 >trainingset.tar.bz2

.. note:: This portion of the tutorial is driven through the command line, so
          you should open up a terminal even if you download and unpack the
          data file through your web browser or some other means.

And this command should unpack it, creating a directory named
``rimphony_powerlaw_s5-5e7_p1.5-7``::

  $ tar xjf trainingset.tar.bz2

If you poke around, you will see that the new directory contains about 500
text files. These are the raw training data, working out to about 22 million
synchrotron coefficient calculations. They are performed for an isotropic
power-law model with power-law indices ranging between 1.5 and 7.

It is easiest to do the training if we rearrange the directory structure a
little bit. The following commands will set things up in a way that we have found
to be convenient::

  $ mv rimphony_powerlaw_s5-5e7_p1.5-7 rawsamples
  $ neurosynchro init-nndir rimphony_powerlaw_s5-5e7_p1.5-7
  $ mv rawsamples rimphony_powerlaw_s5-5e7_p1.5-7/
  $ cd rimphony_powerlaw_s5-5e7_p1.5-7/

The second command above creates a new directory called
``rimphony_powerlaw_s5-5e7_p1.5-7`` (the same name as was initially created,
until we renamed it) and populates it with a default configuration file,
``nn_config.toml``, that we will edit later. The final command changes your
shell to work from this new directory, which will now also contain a
subdirectory named ``rawsamples``. Subsequent steps in the tutorial will
assume that you are still operating from this directory. Make sure to return
to it if you close and re-open your shell.

**Next**: :ref:`transform your training set <transform-training-set>`.
