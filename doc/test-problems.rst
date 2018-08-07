.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

.. _run-test-problems:

Run Some Test Problems
======================

There are many ways to test the performance of neural networks. Sadly,
*neurosynchro* does not currently provide much infrastructure for running such
tests — this is an area for improvement. At the moment, the most prominent
tools are the reports of mean squared error (MSE) and residual plots that you
get from the ``neurosynchro train`` command.

However *neurosynchro* does include a tool that will use `grtrans
<https://github.com/jadexter/grtrans>`_ to do full radiative transfer
integrations with both precisely-calculated and approximated coefficients, so
that you can check its performance in an end-to-end example.

.. warning:: Unfortunately, in its current state, ``grtrans`` is really
             difficult to install in a useful way … and due to this situation,
             we don’t distribute any canned test problems. We can’t recommend
             this testing strategy for any but the hardest-core users. You
             might want to skip ahead to learn about how to :ref:`use your
             networks in your application <use-in-application>`.

To run these tests, the ``grtrans`` Python module ``radtrans_integrate`` must
be importable in your Python setup. Unfortunately, ``grtrans`` does not
provide a standard installation path, so it can be a bit of a hassle to make
this module available. You will know that things are working when the
following command runs without printing any error messages::

  $ python -c 'import radtrans_integrate'

Detailed instructions on how to install ``grtrans`` properly are beyond the
scope of this manual.

Once ``grtrans`` has been installed, you need to create a test data set, as
described below. Then a command like this will run the comparison::

  $ neurosynchro test-grtrans . testproblem.txt

Here is example output for a test problem prepared for the tutorial neural
networks, trained on a power-law electron distribution::

  Using Theano backend.
  Precise computation: I=2.1134e-08  Q=-2.6238e-10  U=-1.6681e-09  V=4.4779e-09  calc_time=103937 ms
  WARNING: some of the approximations were out-of-sample
  Approx. computation: I=2.1151e-08  Q=-3.0201e-10  U=-1.6711e-09  V=4.4855e-09  calc_time=234 ms
  Accuracy: I=0.001  Q=0.151  U=0.002  V=0.002
  Speedup: 444.5


Generating data for a test problem
----------------------------------

The format of the test data file is an expanded version of the one used for
the training data. :ref:`As with the training data <make-training-set>`, the
data should come in a line-oriented text file with several tab-separated
columns and a header line, with input, output, and metadata parameters
indicated as with the training data.

Given such a file, the test tool runs a radiative transfer integration along a
single ray, where each row in the test data file gives a sample of these
coefficients as they change along the ray.

To compute the “right answer”, the testing tool performs a radiative transfer
integration using the coefficients exactly as they are given in your file.
Therefore the test file must contain columns for the standard suite of eight
output parameters: ``(j_I, alpha_I, j_Q, alpha_Q, j_V, alpha_V, rho_Q,
rho_V)``, computed at the reference frequency and density :ref:`as described
for the training data <make-training-set>`.

Four extra columns are also needed to put everything on the physical footing
used by ``grtrans``:

*d(meta)*
  This is the distance along the ray path of each sample point, measured in cm.
  These values should start at zero and increase monotonically along each
  row of the file.
*psi(meta)*
  This is the projected angle between some invariant Stokes U axis and the
  magnetic field, measured in radians. These values can just be zero if you
  don’t care about having a magnetic field whose direction rotates on the sky.
  This parameter is used to map from the local linear polarization frame (in
  which the Stokes U coefficients are always aligned with the magnetic field
  and hence are always zero) to the observer frame.
*n_e(meta)*
  This is the local density of synchrotron-emitting electrons, in cm
  :superscript:`-3`.
*time_ms(meta)*
  This is how much time it took to do the precise calculation of the RT
  coefficients for this row, in milliseconds. This column is only used to
  determine how much faster *neurosynchro* was compared to the detailed
  calculation. You can set it to zero if you don't collect this information.

It is not necessary to choose realistic values for these parameters — the
point of the testing is to just compare the precise and approximated results.
But if you provide realistic values, the integration should give realistic
results.

To compute the approximated result, the test tool will load up the specified
neural network and make predictions based on the input parameters listed in
your file. You must have parameters named *s*, for the local harmonic number,
and *theta*, for the angle between the line-of-sight and the local magnetic
field (measured in radians). The RT integration is performed at a fixed
observing frequency, which means that the local magnetic field strength *B* is
defined implicitly by the ratio of *s* and that frequency.

Note that the goal here is to assess the real-world impact of approximation
errors — not necessarily to test the ``grtrans`` integrator in challenging
circumstances. When generating your test data sets, there’s no particular need
to explore unusual corners of parameter space unless you expect them to arise
in your science application.

**Next**: :ref:`use your networks in your application! <use-in-application>`
