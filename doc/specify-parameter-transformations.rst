.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

.. _specify-parameter-transformations:

Specify Your Parameter Transformations
======================================

The neural networks used by *neurosynchro* perform best when their input and
output parameters are normalized. In order for *neurosynchro* to do this
normalization, you must give it some hints about your input and output
parameters.

These hints are stored in a configuration file that will go along with your
compiled neural network data. If you’re following along with the tutorial,
this file should already exist in the directory that your terminal is working
in; it was created by the ``neurosynchro init-nndir`` command.

Before training the neural nets, however, we need to :ref:`finalize the
configuration file <finalizing-config-file>` as described below.

.. attention:: If you’re following along with the tutorial, the final section
               gives important instructions you need to follow. Most of the
               rest of the material can be skimmed for now, though. Maybe read
               it while your nets are training?


.. _transformations:

How *Neurosynchro* transforms parameters
----------------------------------------

Before we go into the details of configuration, we need to describe the
transformations that *neurosynchro* performs for its computations.

In :ref:`the previous section <transform-training-set>`, you transformed the
output parameters of your training set. That first stage of transformation
makes it so that the numerics don’t need to worry about obeying various
physical invariants that apply to the Stokes-basis radiative transfer
coefficients.

The numbers in the resulting output file (called ``transformed/all.txt`` in
the example command) are referred to as *physical* values in the context of
*neurosynchro*. These are the numbers that you will eventually get back out of
the neural network.

Inside *neurosynchro*, physical values are transformed into what are called,
well, *transformed* values that are easier to work with. For instance, the
total emission coefficient ``j_I`` should generally undergo a log transform
because its physical values span a huge dynamic range. At this stage, there is
also a question of how to deal physical inputs that are out of bounds: if you
trained your neural network on power-law indices between 1 and 3, and your
code asks *neurosynchro* to compute coefficients with a power-law index of
3.5, what should *neurosynchro* do? The “correct” answer to this question can
vary, so *neurosynchro* lets you configure its behavior.

Finally, for the innermost calculations the transformed values are converted
to *normalized* values with a simple linear mapping:

.. math::

   n = m (t - t_0).

Typically the mean is subtracted off and the standard deviation is divided
out, so that the distribution of the normalized parameters should be
approximately normal (i.e., Gaussian). In special circumstances, however,
other transforms can be applied.


The ``nn_config.toml`` configuration file
-----------------------------------------

The ``neurosynchro init-nndir`` initialization command will create a neural
network data directory and put one file in it: a configuration file named
``nn_config.toml``. This file specifies how the input and output parameters
should be handled. The file is in `TOML format
<https://github.com/toml-lang/toml#readme>`_, which is a simple scheme for
storing structured data. (TOML is like `JSON <https://www.json.org/>`_ but, in
the opinion of this writer, better.)

The default file consists of a series of stanzas each denoted by a pair of
double square brackets. By default, the first stanza looks like this::

  [[params]]
  name = "s"
  maptype = "log"

The delimiter ``[[params]]`` indicates that this stanza defines an input
parameter. The following lines assign various settings as described below.

There are also stanzas marked with the delimiter ``[[results]]``, which give
configuration settings for output results. The two kinds of stanzas accept
almost identical kinds of configuration values; exceptions will be noted
below.

Allowed configuration values are as follows:

*name*
  This isn’t really a configuration setting. It gives the name of the input or
  output parameter being configured. This should be one of the items that
  appears in the column headers of your training sample. Names corresponding
  to nonexistent columns will be ignored.
*maptype*
  This setting specifies how the physical values given in your training set
  are transformed. The most useful options are ``direct``, which performs no
  transformation, and ``log``, which takes the logarithm of the physical
  values. The full set of possible values is :ref:`given below <map-types>`.
*phys_bounds_mode*
  As mentioned above, *neurosynchro* has to decide what to do if you ask it to
  compute coefficients outside of the bounds spanned by its training set. This
  parameter configures how those bounds are actually determined. There are
  just two options: ``empirical`` and ``theta``. For ``empirical``, the bounds
  are determined by the empirical minimum and maximum physical parameter
  values seen in the training set. For ``theta``, the bounds are fixed to be
  the range between 0 and π/2. The default is ``empirical``, which is what
  makes sense in almost all cases. The ``theta`` setting is recommended for
  the ``theta`` input parameter, so that calculations where θ got *really*
  close to π/2 don’t get rejected even if your training set didn’t get *quite*
  as close.
*normalization_mode*
  This setting specifies how to determine the parameters of the linear
  transform used to obtain the final normalized values. If it is ``gaussian``,
  the default, the mean and standard deviation of the transformed values
  (note: not the physical values) are used to yield an approximately normal
  distribution. If ``unit_interval``, the min and max of the transformed
  values will be used in such a way that the normalized values span the unit
  interval ``[0, 1]``.
*out_of_sample*
  This setting specifies what to do if a physical value lies beyond the range
  of the training set. This can happen on *either* the input *or* the output
  side. Your simulation might require input parameters that are beyond the
  ones you trained on, but also the neural network approximator might yield
  results that end up lying outside of training-set range. Possible values are
  ``ignore`` (the default), ``clip``, and ``nan``. With ``ignore``, the sample
  limits are ignored and the calculation plunges ahead recklessly. With
  ``clip``, the input or output physical parameters are clipped to stay within
  the sampled physical range — note that means that you can get back results
  that just plain *do not correspond* to the parameters that you thought you
  were using! The *neurosynchro* driver code collects flags so that you can
  tell when this happens. Finally, ``nan`` flags the affected calculations and
  causes the driver to return `Not-a-Number
  <https://en.wikipedia.org/wiki/NaN>`_ values unconditionally.
*trainer*
  This setting only applies to output parameters. It specifies which scheme
  will be used to train the neural network to compute this output. There is a
  ``generic`` trainer that generally does well; the list of all possibilities
  is :ref:`given in the next section <trainer-types>`.

.. _map-types:

Map Types
~~~~~~~~~

*Neurosynchro* supports the following transformations between “physical”
parameter values and internal “transformed” values:

*abs_log*
  The transformed value is the logarithm of the absolute value of the
  physical value. This transform is not reversible on its own. It is used
  by the driver code for the *rho_Q* parameter, which both spans a large
  dynamic range and takes on both positive and negative values. The driver
  deals with this by splitting it into two components: an overall amplitude
  (using this mapping) and a sign term.
*direct*
  The transformed value is the physical value. This is useful for parameters
  like power-law indices that do not span a large dynamic range.
*log*
  The transformed value is the base-10 logarithm of the physical value. This
  is useful for parameters that span large dynamic ranges and are always positive.
*logit*
  The transformed value is the logit of the physical value:

  .. math::
     t = \log(\frac{p}{1 - p})

  This maps a value in the range ``(0, 1)`` to the range ``(-∞, +∞)``, so to
  use this the physical value must be constrained to lie in the unit interval.
  This is the case for the “polarization share” parameters used in the
  transformed output parameters.
*neg_log*
  The transformed value is the base-10 logarithm of the negation of the
  physical value. This is useful for parameters that span large dynamic ranges
  and are always negative.
*ninth_root*
  The transformed value is the ninth root of the physical value, preserving
  sign. This is adequate for parameters that span large-ish dynamic ranges and
  take on both positive and negative values. This mapping is no longer used in
  the recommended configuration.
*sign*
  The transformed value is the signum of the physical value: either -1, 0, or
  1 depending on physical value’s sign. This transform is not reversible on
  its own, but is used for the *rho_Q* parameter as described above.


.. _finalizing-config-file:

Finalizing the Configuration File
---------------------------------

The configuration file generated by the ``neurosynchro init-nndir`` command
contains suggested defaults for the *s* and *theta* input parameters and the
suite of output parameters generated by the ``neurosynchro transform`` step.

However, the command doesn’t (and can’t) know what other input parameters your
model uses, so you must edit the ``nn_config.toml`` file to define them. Add
stanzas analogous to the example one used for the *s* input parameter. The
defaults are often useful, so you probably only need to ask yourself:

* Should this parameter have a ``maptype`` of ``direct`` or ``log``?
* What do I want its ``out_of_sample`` behavior to be?

You may want to revisit this file to, for example, try a different neural
network training scheme to improve *neurosynchro*’s performance for a certain
model output parameter.


Finalizing Configuration for the Tutorial
-----------------------------------------

For the purposes of the tutorial, here are the specific adjustments you should
make to the default configuration file:

1. Add a stanza defining the power-law index parameter, after the stanza for
   the ``theta`` parameter::

     [[params]]
     name = "p"
     maptype = "direct"

.. rho-q: twolayer??

**Next**: :ref:`precalculate the domain and range of your training set <precalculate-domain-range>`.
