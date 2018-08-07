.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

.. _transform-training-set:

Transform Your Training Set
===========================

The :ref:`standard parameters <standard-output-parameters>` of radiative
transfer in the Stokes basis are the easiest to work with in a scientific
context, but they can be tricky to work with when doing numerical
approximations. This is because they must obey invariants such as

.. math::

   |j_I| > |j_Q|,

which can be broken in the face of approximation noise. Therefore,
*neurosynchro* internally uses a transformed set of parameters that don’t have
to obey such invariants.

Once you have generated training set data, then, you must transform them into
this internal representation. This is done with the ``transform`` subcommand
of the ``neurosynchro`` command that gets installed along with the
*neurosynchro* Python package. It’s very straightforward to use. Assuming that
you are using the standard directory structure, just run::

  $ mkdir -p transformed
  $ neurosynchro transform rawsamples >transformed/all.txt

Here, ``rawsamples`` is the name of the directory containing the training
data. The ``transform`` subcommand prints the transformed database to standard
output, so in the example above, shell redirection is used to save the results
to the file ``all.txt`` in a new subdirectory. With the example training set
data, the resulting text file will be hundreds of megabytes in size.

The ``transform`` subcommand will also filter your data, discarding any rows
containing non-finite values or with outputs that do not obey the necessary
invariants to begin with. It prints a statistical summary of this filtering to
the standard error stream.

The ``summarize`` subcommand will print out some summary information about
your training set::

  $ neurosynchro summarize transformed

Here you could also give an argument of ``rawsamples`` to analyze the
pre-transformed inputs.

**Next**: :ref:`specify your parameter transformations <specify-parameter-transformations>`.
