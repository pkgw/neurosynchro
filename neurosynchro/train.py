# -*- mode: python; coding: utf-8 -*-
# Copyright 2017-2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Train one of the neural networks.

Meant to be run as a program in production, but you can import it to
experiment with training regimens.

"""
from __future__ import absolute_import, division, print_function

import argparse, sys, time
from pwkit.cli import die
from pwkit.io import Path

from . import DomainRange


def generic_trainer(m):
    from keras.layers import Dense

    m.add(Dense(
        units = 300,
        input_dim = m.domain_range.n_params,
        activation = 'relu',
        kernel_initializer = 'normal',
    ))
    m.add(Dense(
        units = 1,
        activation = 'linear',
        kernel_initializer = 'normal',
    ))
    m.compile('adam', 'mse')
    hist = m.ns_fit(
        epochs = 30,
        batch_size = 2000,
        verbose = 0,
    )
    print('Intermediate MSE:', hist.history['loss'][-1])
    m.ns_sigma_clip(7)
    hist = m.ns_fit(
        epochs = 30,
        batch_size = 2000,
        verbose = 0,
    )
    m.final_mse = hist.history['loss'][-1]
    return m


def twolayer_trainer(m):
    from keras.layers import Dense

    m.add(Dense(
        units = 120,
        input_dim = m.domain_range.n_params,
        activation = 'relu',
        kernel_initializer = 'normal',
    ))
    m.add(Dense(
        units = 60,
        activation = 'relu',
        kernel_initializer = 'normal',
    ))
    m.add(Dense(
        units = 1,
        activation = 'linear',
        kernel_initializer = 'normal',
    ))
    m.compile('adam', 'mse')
    hist = m.ns_fit(
        epochs = 30,
        batch_size = 2000,
        verbose = 0,
    )
    print('Intermediate MSE:', hist.history['loss'][-1])
    m.ns_sigma_clip(7)
    hist = m.ns_fit(
        epochs = 30,
        batch_size = 2000,
        verbose = 0,
    )
    m.final_mse = hist.history['loss'][-1]
    return m


def binary_trainer(m):
    from keras.layers import Dense

    m.add(Dense(
        units = 120,
        input_dim = m.domain_range.n_params,
        activation = 'relu',
        kernel_initializer = 'normal',
    ))
    m.add(Dense(
        units = 60,
        activation = 'relu',
        kernel_initializer = 'normal',
    ))
    m.add(Dense(
        units = 1,
        activation = 'sigmoid',
        kernel_initializer = 'normal',
    ))
    m.compile(optimizer='adam', loss='binary_crossentropy')
    hist = m.ns_fit(
        epochs = 30,
        batch_size = 2000,
        verbose = 0,
    )
    print('Intermediate MSE:', hist.history['loss'][-1])
    # Note: no sigma-clipping
    hist = m.ns_fit(
        epochs = 30,
        batch_size = 2000,
        verbose = 0,
    )
    m.final_mse = hist.history['loss'][-1]
    return m


def load_data_and_train(datadir, nndir, result_name):
    from .impl import NSModel

    cfg_path = Path(nndir) / 'nn_config.toml'
    dr, rinfo = DomainRange.from_serialized(cfg_path, result_to_extract=result_name)

    if rinfo is None:
        die('no known result named %r', result_name)

    sd = dr.load_and_normalize(datadir)

    trainer_name = rinfo['trainer']
    trainer_func = globals().get(trainer_name + '_trainer')
    if trainer_func is None:
        die('unknown trainer function %r', trainer_name)

    print('Training with scheme \"%s\"' % trainer_name)
    m = NSModel()
    m.ns_setup(rinfo['_index'], sd)
    t0 = time.time()
    trainer_func(m)
    m.training_wall_clock = time.time() - t0
    return m


def page_results(m, residuals=False, thin=500):
    import omega as om

    pg = om.makeDisplayPager()
    for i in range(m.domain_range.n_params):
        pg.send(m.ns_plot(i, plot_err=residuals, thin=thin))

    pg.done()


def make_parser(ap=None):
    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument('-p', '--plot', action='store_true',
                    help='Compare the NN and Symphony after training.')
    ap.add_argument('-r', '--residuals', action='store_true',
                    help='If plotting, plot residuals rather than absolute values (requires `omegaplot` package).')
    ap.add_argument('datadir', type=str, metavar='<datadir>',
                    help='The path to the sample data directory.')
    ap.add_argument('nndir', type=str, metavar='<nndir>',
                    help='The path to the neural-net directory.')
    ap.add_argument('result_name', type=str, metavar='<result-name>',
                    help='The name of the simulation result to train on.')
    return ap


def train_cli(settings):
    m = load_data_and_train(settings.datadir, settings.nndir, settings.result_name)
    print('Achieved MSE of %g in %.1f seconds for %s.' %
          (m.final_mse, m.training_wall_clock, settings.result_name))

    if settings.plot:
        page_results(m, residuals=settings.residuals)

    outpath = str(Path(settings.nndir) / ('%s.h5' % settings.result_name))
    m.save(outpath, overwrite=True)
