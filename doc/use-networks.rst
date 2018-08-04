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

Instead, *neurosynchro*’s job is just to compute radiative transfer
coefficients quickly, given some input parameters. This is as straightforward as::

  import neurosynchro.impl
  import numpy as np

  nn_dir = 'nndir' # path to the directory with your trained neural nets
  approx = neurosynchro.impl.PhysicalApproximator(nn_dir)

  nu = 1e10 # observing frequency in Hz
  B = {some array of sampled magnetic field strengths, in Gauss}
  n_e = {array of electron densities, in cm^-3}
  theta = {array of angles between line-of-sight and the local B field, in radians}
  psi = {array of angles between local B field and observer's Stokes U axis, in radians}

  # Additional arrays correspond to parameters of the electron energy distribution
  # function; the relevant parameters depend on which distribution function you
  # have been simulating. For an isotropic power law distribution, there is usually
  # just one:
  p = {array of electron energy power law indices}

  # Compute coefficients in basis where Stokes U tracks the local field:
  coeffs = approx.compute_all_nontrivial(nu, B, n_e, theta, p=p)

  # Transform to a basis in which Stokes U is invariant from the observer's perspective:
  coeffs = neurosynchro.detrivialize_stokes_basis(coeffs, psi)

  # The coefficients are packed as follows:
  names = 'jI aI jQ aQ jU aU jV aV rQ rU rV'.split()

  for index, name in enumerate(names):
      print('Mean {} coefficient: {:e}'.format(name, np.mean(coeffs[...,index])))

Tadaa!
