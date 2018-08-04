.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

.. _make-training-set:
   
Make Your Training Set
======================

In order to train a neural network, you need something to train it on!

The training data are a bunch of samples of your detailed calculation. That
is: given a choice of some input parameters (e.g., a harmonic number), your
detailed calculation will produce eight output parameters (the Stokes
radiative transfer coefficients). The exact number of input parameters can
vary depending on what particle distribution you’re modeling. *Neurosynchro*
needs a lot of samples of this function to develop a good approximation to it.

From the standpoint of *neurosynchro*, the tool that you use to **generate**
those data doesn’t matter. What matters is the **format** in which the
training data are stored. However, it is true that *neurosynchro* has been
designed to work with `rimphony <https://github.com/pkgw/rimphony/>`_, which
has `sample programs
<https://github.com/pkgw/rimphony/blob/master/examples/crank-out-pitchypl.rs>`_
that will generate training data sets in the format described below.

.. attention:: Read this section carefully! *Neurosynchro* bakes in some
               assumptions that might surprise you.

The training data fed to *neurosynchro* must be saved as a set of plain
textual tables stored in a single directory. The file names must end in
``.txt``. Each table file is line-oriented. The first line is a header, and
all subsequent lines give samples of the exact calculation. For example::

   s(log)	theta(lin)	p(lin)	k(lin)	time_ms(meta)	j_I(res)	alpha_I(res)	j_Q(res)	alpha_Q(res)	j_V(res)	alpha_V(res)	rho_Q(res)	rho_V(res)
   1.95393e2	8.8966e-1	3.49e0	1.41e0	2.21270e3	8.42819e-35	2.26887e-8	-6.439070e-35	-1.80416e-8	1.17279e-35	3.56901e-9	3.2947e-7	3.8318e-5
   6.51244e2	8.0044e-1	3.28e0	1.94e0	3.30821e3	6.88161e-36	9.03608e-10	-5.226766e-36	-7.17748e-10	6.41000e-37	9.59868e-11	2.4798e-8	1.2309e-5

The header lines gives the names of each column. Each column name includes a
suffix indicating its type:

``lin``
   An input parameter that is sampled linearly in some range. At the moment,
   the way in which the parameter was sampled isn’t actually used anywhere
   in the code. But it can be a helpful piece of information to have handy
   when specifying how the neural nets will be trained.
``log``
   An input parameter that is sampled logarithmically in some range.
``res``
   A output result from the computation.
``meta``
   A metadata value that records some extra piece of information about
   the sample in question. Above, the ``time_ms`` metadata item records
   how many milliseconds it took for *Rimphony* to calculate the sample.
   This is useful for identifying regions of parameter space where the
   code runs into numerical problems.

So, in the example above, there are four input parameters. The detailed
calculation shows that when the harmonic number *s* ≃ 195, observing angle
*theta* ≃ 0.9 radians, energy power-law index *p* ≃ 3.5, and pitch-angle
distribution index *k* ≃ 1.4, the emission coefficient *j_I* ≃ 8 × 10
:superscript:`-35` erg s :superscript:`-1` cm :superscript:`-2` Hz
:superscript:`-1` sr :superscript:`-1`. The *rimphony* calculation of that
result took about 2.2 seconds.

Something like 100,000 rows is enough to train some good neural networks. It
doesn't matter how many different files those rows are split into.

.. tip:: *Neurosynchro* takes a directory of files as an input, rather than
         one specific file, since the former is easier to create on a big HPC
         cluster where you can launch 1,000 jobs to compute coefficients for
         you in parallel.

.. tip:: Each individual input file can be easily loaded into a `Pandas
         <https://pandas.pydata.org/>`_ data frame with the function call
         ``pandas.read_table()``.


Important assumptions
---------------------

In the example above, there are just four input parameters: *s*, *theta*, *p*,
and *k*. These are likely not the usual parameters that you see when thinking
about synchrotron radiation. There’s an important reason for this!

*Neurosynchro* bakes in three key assumptions about how synchrotron radiation
works:

1. You must compute all of your coefficients **at an observing frequency of 1
   Hz**! This is because synchrotron coefficients scale simply with frequency:
   emission coefficients linearly with ν, absorption coefficients as 1/ν. So
   the observing frequency doesn’t actually need to be part of the neural
   network regression.
2. You must compute all of your coefficients **at an energetic particle
   density of 1 per cubic centimeter**! Here too, all the synchrotron
   coefficients scale simply with the energetic particle density (namely, they
   all scale linearly). Once again this means that the energetic particle
   density doesn´t actually need to be part of the regression.
3. You only need to do computations where the angle between the line of sight
   and the magnetic field, θ, is less than 90°. *Neurosynchro* assumes that
   all parameters are symmetric with regards to θ = 90° except the Stokes V
   components, which negate.

Given those assumptions, almost every part of *neurosynchro* expects that the
following input parameters will exist:

*s*
   The harmonic number of interest, such that:

   .. math::

      \nu_\text{obs} = s \nu_\text{cyc} = s \frac{e B}{2 \pi m_e c}

*theta*
   The angle between the direction of radiation propagation and the local
   magnetic field, *in radians*.

Given the known ways in which synchrotron coefficients scale, the standard
quartet of input parameters ``nu``, ``theta``, ``n_e``, and ``B`` can be
reduced to these two parameters, plus scalings that are known *a priori*. In
the example above, the two remaining parameters *p* and *k* relate to the
shape of the particle distribution function.

.. _standard-output-parameters:

On the output side, *neurosynchro* applies some more assumptions to ensure
that it always produces physically sensible output (i.e., that the polarized
Stokes emission parameters are never bigger than the total-intensity Stokes
emission parameter). It also uses the standard linear polarization basis in
which Stokes Q is aligned with the magnetic field, which means that the
Stokes U parameters are zero by construction (see, for example, Shcherbakov &
Huang (2011), `DOI 10.1111/j.1365-2966.2010.17502.x
<https://doi.org/10.1111/j.1365-2966.2010.17502.x>`_, equation 17). So unless
you are doing something very unusual, your tables should always contain eight
output results:

*j_I*
   The calculated Stokes I emission coefficient, in erg s :superscript:`-1`
   cm :superscript:`-2` Hz :superscript:`-1` sr :superscript:`-1`.
*j_Q*
   The Stokes Q emission coefficient, in the same units.
*j_V*
   The Stokes V emission coefficient, in the same units.
*alpha_I*
   The calculated Stokes I absorption coefficient, in cm :superscript:`-1`.
*alpha_Q*
   The Stokes Q absorption coefficient, in the same units.
*alpha_V*
   The Stokes V absorption coefficient, in the same units.
*rho_Q*
   The Faraday conversion coefficient (mixing Stokes U and Stokes V), in
   the same units.
*rho_V*
   The Faraday rotation coefficient (mixing Stokes Q and Stokes U), in
   the same units.

**Next**: :ref:`transform your training set <transform-training-set>`.
