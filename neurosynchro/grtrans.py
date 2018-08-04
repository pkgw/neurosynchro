# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""This module provides helpers for doing radiative transfer integrations with
`grtrans <https://github.com/jadexter/grtrans>`_, including a simple
framework for running end-to-end tests.

In order to use this functionality, the Python module ``radtrans_integrate``
must be importable. Sadly `grtrans <https://github.com/jadexter/grtrans>`_
doesn't install itself like a regular Python package, so getting this working
can be a pain. Documenting the installation procedure is beyond the scope of
this project.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd


# grtrans' integration methods:
METHOD_LSODA_YES_LINEAR_STOKES = 0 # LSODA with IS_LINEAR_STOKES=1
METHOD_DELO = 1 # DELO method from Rees+ (1989ApJ...339.1093R)
METHOD_FORMAL = 2 # "formal" method = "matricant (O-matrix) method from Landi Degl'Innocenti"
METHOD_LSODA_NO_LINEAR_STOKES = 3 # LSODA with IS_LINEAR_STOKES=0 -- this is "under development" spherical stokes


def integrate_ray_formal(x, j, K):
    """Use `grtrans <https://github.com/jadexter/grtrans>`_ to integrate one ray
    using its "formal" (matricant / O-matrix) method.

    **Call signature**

    *x*
      1D array, shape (n,). Path length along ray, starting from zero, in cm.
    *j*
      Array, shape (n, 4). Emission coefficients: ``j_{IQUV}``, in that order.
    *K*
      Array, shape (n, 7). Absorption coefficients and Faraday mixing coefficients:
      ``alpha_{IQUV}, rho_{QUV}``.
    Return value
      Array of shape (4, m): Stokes intensities ``IQUV`` along parts of the
      ray with non-zero total emissivities; m <= n.

    """
    # A correct version of the fully-specified absorption matrix is in Leung+
    # 2011 (10.1088/0004-637X/737/1/21). From looking at the function
    # `radtrans_jac_form` of grtrans' `radtrans_integrate.f90` file, one can
    # see that the K vector is indeed packed in the way described above.
    from radtrans_integrate import radtrans_integrate

    n = x.size

    # The formal method doesn't do the same kind of clipping as LSODA *inside*
    # grtrans, but here we do the same clipping for consistency, and since it
    # is genuinely true that the clipped samples are superfluous.

    if np.all(j[:,0] == 0.):
        return np.zeros((4, n))

    i0 = 0
    i1 = n - 1

    while j[i0,0] == 0.:
        i0 += 1
    while j[i1,0] == 0.:
        i1 -= 1

    n = i1 + 1 - i0
    x = x[i0:i1+1]
    j = j[i0:i1+1]
    K = K[i0:i1+1]

    # OK we can go.

    radtrans_integrate.init_radtrans_integrate_data(
        METHOD_FORMAL, # method selector
        4, # number of equations
        n, # number of input data points
        n, # number of output data points
        10., # maximum optical depth; not used by "formal"
        1., # maximum absolute step size; not used by "formal"
        0.1, # absolute tolerance; not used by "formal"
        0.1, # relative tolerance; not used by "formal"
        1e-2, # "thin" parameter for DELO method; not used by "formal"
        1, # maximum number of steps; not used by "formal"
    )

    try:
        tau = x # not used by "formal"
        radtrans_integrate.integrate(x[::-1], j[::-1], K[::-1], tau[::-1], 4)
        i = radtrans_integrate.intensity.copy()
    finally:
        # If we exit without calling this, the next init call causes an abort
        radtrans_integrate.del_radtrans_integrate_data()
    return i


def integrate_ray_lsoda(x, j, K, atol=1e-8, rtol=1e-6, max_step_size=None,
                        frac_max_step_size=1e-3, max_steps=100000):
    """Use `grtrans <https://github.com/jadexter/grtrans>`_ to integrate one ray
    using its LSODA method.

    **Call signature**

    *x*
      1D array, shape (n,). Path length along ray, starting from zero, in cm.
    *j*
      Array, shape (n, 4). Emission coefficients: ``j_{IQUV}``, in that order.
    *K*
      Array, shape (n, 7). Absorption coefficients and Faraday mixing coefficients:
      ``alpha_{IQUV}, rho_{QUV}``.
    *atol*
      Some kind of tolerance parameter.
    *rtol*
      Some kind of tolerance parameter.
    *max_step_size*
      The maximum absolute step size. Overrides *frac_max_step_size*.
    *frac_max_step_size*
      If *max_step_size* is not specified, the maximum step size passed to the
      integrator is ``x.max()`` multiplied by this parameter. Experience shows
      that (for LSODA at least) this parameter must be pretty small to get
      good convergence!
    *max_steps*
      The maximum number of steps to take.
    Return value
      Array of shape (4, m): Stokes intensities IQUV along parts of the ray with
      non-zero total emissivities; m <= n.

    """
    n = x.size

    if max_step_size is None:
        max_step_size = frac_max_step_size * x.max()

    # the LSODA method clips its input arrays based on "tau" and zero emission
    # coefficients. It's hard for us to find out how it clipped, though, so we
    # reproduce its logic. LSODA doesn't use "tau" for anything besides this
    # clipping, so we pass it all zeros.

    if np.all(j[:,0] == 0.):
        return np.zeros((4, n))

    i0 = 0
    i1 = n - 1

    while j[i0,0] == 0.:
        i0 += 1
    while j[i1,0] == 0.:
        i1 -= 1

    n = i1 + 1 - i0
    x = x[i0:i1+1]
    j = j[i0:i1+1]
    K = K[i0:i1+1]

    # OK we can go.

    radtrans_integrate.init_radtrans_integrate_data(
        METHOD_LSODA_YES_LINEAR_STOKES, # method selector
        4, # number of equations
        n, # number of input data points
        n, # number of output data points
        10., # maximum optical depth; defused here (see comment above)
        max_step_size, # maximum absolute step size
        atol, # absolute tolerance
        rtol, # relative tolerance
        1e-2, # "thin" parameter for DELO method ... to be researched
        max_steps, # maximum number of steps
    )

    try:
        tau = np.zeros(n)
        radtrans_integrate.integrate(x[::-1], j, K, tau, 4)
        i = radtrans_integrate.intensity.copy()
    finally:
        # If we exit without calling this, the next init call causes an abort
        radtrans_integrate.del_radtrans_integrate_data()
    return i


def integrate(d, coeffs, psi):
    """Integrate a ray with `grtrans <https://github.com/jadexter/grtrans>`_,
    using reasonable defaults.

    **Call signature**

    *d*
      An array giving the location of each sample along the ray, starting from zero, in cm.
    *coeffs*
      An array of shape (N, 8) of RT coefficients in the basis where the
      Stokes U coefficients are always zero. Such arrays are returned by
      :meth:`neurosynchro.impl.PhysicalApproximator.compute_all_nontrivial`.
    *psi*
      An array of angles between the local magnetic field and the observer’s Stokes U
      axis, in radians.
    Return value
      An array of shape (4,), giving the Stokes IQUV at the end of the ray.

    This function is mainly intended to test what happens if the passed-in
    coefficients are slightly different due to the neural network
    approximation. So we don't provide many knobs or diagnostics here.

    """
    from . import detrivialize_stokes_basis

    xformed = detrivialize_stokes_basis(coeffs, psi)
    j = np.empty(coeffs.shape[:-1] + (4,))
    j[...,0] = xformed[...,0]
    j[...,1] = xformed[...,2]
    j[...,2] = xformed[...,4]
    j[...,3] = xformed[...,6]
    K = np.empty(coeffs.shape[:-1] + (7,))
    K[...,0] = xformed[...,1]
    K[...,1] = xformed[...,3]
    K[...,2] = xformed[...,5]
    K[...,3:] = xformed[...,7:]

    iquv = integrate_ray_formal(d, j, K)
    return iquv[:,-1]


def make_parser(ap=None):
    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument('--frequency', '-f', type=float, metavar='<ghz>',
                    default=10.0, help='The observing frequency to simulate')
    ap.add_argument('nndir', type=str, metavar='<nndir>',
                    help='The path to the neural-net directory.')
    ap.add_argument('testdata', type=str, metavar='<testdata>',
                    help='A file containing test data')
    return ap


def grtrans_cli(settings):
    from pwkit import cgs
    from pwkit.cli import die
    from time import time
    from .impl import PhysicalApproximator, hardcoded_nu_ref, hardcoded_ne_ref

    # Read and validate the test dataset.

    testdata = pd.read_table(settings.testdata)

    psi = testdata.get('psi(meta)')
    if psi is None:
        die('the test dataset must contain a column of field-to-Stokes-U angles \"psi(meta)\"')

    d = testdata.get('d(meta)')
    if d is None:
        die('the test dataset must contain a column of integration path lengths \"d(meta)\"')

    n_e = testdata.get('n_e(meta)')
    if n_e is None:
        die('the test dataset must contain a column of particle densities \"n_e(meta)\"')

    time_ms = testdata.get('time_ms(meta)')
    if time_ms is None:
        die('the test dataset must contain a column of computation times \"time_ms(meta)\"')

    s = None
    theta = None
    others = {}

    for col in testdata.columns:
        if col.startswith('s('):
            s = testdata[col]
        elif col.startswith('theta('):
            theta = testdata[col]
        elif col.endswith('(lin)') or col.endswith('(log)'):
            others[col.split('(')[0]] = testdata[col]

    if s is None:
        die('the test dataset must have an input parameter of the harmonic number \"s\"')

    if theta is None:
        die('the test dataset must have an input parameter of the field-to-LOS angle \"theta\"')

    # Get the coefficients into physical units, packed in our standard format.

    nu_hz = settings.frequency * 1e9
    freq_scale = nu_hz / hardcoded_nu_ref
    n_e_scale = n_e / hardcoded_ne_ref

    coeffs = np.empty((psi.size, 8))
    coeffs[...,0] = testdata['j_I(res)'] * freq_scale
    coeffs[...,1] = testdata['alpha_I(res)'] / freq_scale
    coeffs[...,2] = testdata['j_Q(res)'] * freq_scale
    coeffs[...,3] = testdata['alpha_Q(res)'] / freq_scale
    coeffs[...,4] = testdata['j_V(res)'] * freq_scale
    coeffs[...,5] = testdata['alpha_V(res)'] / freq_scale
    coeffs[...,6] = testdata['rho_Q(res)'] / freq_scale
    coeffs[...,7] = testdata['rho_V(res)'] / freq_scale
    coeffs *= n_e_scale.values.reshape((-1, 1))

    # Ground truth:

    iquv_precise = integrate(d, coeffs, psi)
    ctime_precise = time_ms.sum()
    print('Precise computation: I={:.4e}  Q={:.4e}  U={:.4e}  V={:.4e}  calc_time={:.0f} ms'.format(
          iquv_precise[0], iquv_precise[1], iquv_precise[2], iquv_precise[3], ctime_precise
    ))

    # Now set up the approximator and do the same thing. (Note that often the
    # timing seems backwards, because the time spent doing the precise
    # calculation has already been spent, whereas we have a lot of overhead to
    # set up the neural networks.)

    B = 2 * np.pi * cgs.me * cgs.c * nu_hz / (s * cgs.e)
    approx = PhysicalApproximator(settings.nndir)
    t0 = time()
    coeffs, oos = approx.compute_all_nontrivial(nu_hz, B, n_e, theta, **others)
    ctime_approx = 1000 * (time() - t0)

    if np.any(oos != 0):
        print('WARNING: some of the approximations were out-of-sample')

    iquv_approx = integrate(d, coeffs, psi)
    print('Approx. computation: I={:.4e}  Q={:.4e}  U={:.4e}  V={:.4e}  calc_time={:.0f} ms'.format(
          iquv_approx[0], iquv_approx[1], iquv_approx[2], iquv_approx[3], ctime_approx
    ))

    acc = np.abs((iquv_approx - iquv_precise) / iquv_precise)
    print('Accuracy: I={:.3f}  Q={:.3f}  U={:.3f}  V={:.3f}'.format(acc[0], acc[1], acc[2], acc[3]))

    print('Speedup: {:.1f}'.format(ctime_precise / ctime_approx))
