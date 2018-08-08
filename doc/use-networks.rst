.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

.. _use-in-application:

Use Your Networks for Your Application
======================================

You’ve generated a training set, trained your networks, and tested their
performance in some sample problems. Time to use them for real!

As we have written, actually performing a full radiative transfer integration
is not within *neurosynchro*’s purview. As :ref:`suggested in the previous
section <run-test-problems>`, the tool `grtrans
<https://github.com/jadexter/grtrans>`_ can be a good choice, but there are
many options out there.

.. keep the code below narrow so the reader can see it all

Instead, *neurosynchro*’s job is just to compute radiative transfer
coefficients quickly, given some input parameters. This is as straightforward
as::

  # Python 2/3 compatibility:

  from __future__ import absolute_import, division, print_function

  # Imports:

  import neurosynchro.impl
  import numpy as np

  # Configuration: all you need is the path to the directory with some
  # trained neural nets. If you're following the tutorial, the
  # relevant directory is named this -- but you should use '.' if
  # you started Python in the directory containing `nn_config.toml`.

  nn_dir = 'rimphony_powerlaw_s5-5e7_p1.5-7'

  # Create an object that will do your calculations. Initializing the
  # neural networks typically takes ~10 seconds.

  approx = neurosynchro.impl.PhysicalApproximator(nn_dir)

  # Create some arrays of physical parameters of interest. The
  # following parameters will occur in all computations:

  N = 256  # how many points to sample?
  nu = 1e10  # observing frequency in Hz
  B = np.linspace(100, 10, N)  # samples of magnetic field strengths,
                               # in Gauss
  n_e = np.logspace(6, 4, N)  # electron densities, in cm^-3
  theta = 0.7  # angle(s) between line-of-sight and the local B
               # field, in radians
  psi = np.linspace(0.1, 0.3, N)  # angles between local B field and
                                  # observer's Stokes U axis, in
                                  # radians

  # But you will almost always need additional arrays of data
  # specifying the parameters of the electron distribution
  # function. Exactly which parameters are relevant parameters depends
  # on which distribution function is being used. For an isotropic
  # power law distribution, there's just:

  p = np.linspace(2, 4, N)  # electron energy power law index

  # Compute coefficients in basis where Stokes U tracks the local
  # field. "oos" stands for "out of sample", and gives information
  # about if any of the calculations ran outside of the domain of the
  # training set. Additional parameters describing the electron
  # distribution function would be passed as additional keywords.

  coeffs, oos = approx.compute_all_nontrivial(nu, B, n_e, theta, p=p)

  # Transform to a basis in which Stokes U is invariant from the
  # observer's perspective:

  coeffs = neurosynchro.detrivialize_stokes_basis(coeffs, psi)

  # The coefficients are packed as follows:

  names = 'jI aI jQ aQ jU aU jV aV rQ rU rV'.split()

  for index, name in enumerate(names):
      print('Mean {} coefficient: {:e}'
            .format(name, np.mean(coeffs[...,index])))

That’s about all there is to it, in the end — what you do with the numbers you
get from *neurosynchro* is not its business. But if you’d like a few more
details, you might want to start with the documentation of Python API of the
:class:`neurosynchro.impl.PhysicalApproximator` class.
