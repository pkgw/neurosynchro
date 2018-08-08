#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators
# Licensed under the MIT License

from __future__ import absolute_import, division, print_function

from setuptools import setup, Extension

# Keep synchronized with doc/conf.py:
version = '0.1.4'

setup(
    name = 'neurosynchro',
    version = version,
    author = 'Peter Williams',
    author_email = 'peter@newton.cx',
    url = 'https://github.com/pkgw/neurosynchro/',
    license = 'MIT',
    description = ('Use neural networks to approximate polarized synchrotron '
                   'radiative transfer coefficients'),

    long_description = '''\
*Neurosynchro* is a small Python package for creating and using neural
networks to quickly approximate the coefficients needed for fully-polarized
synchrotron radiative transfer. It builds on the `Keras <https://keras.io/>`_
deep learning library. Documentation may be found `on ReadTheDocs
<https://neurosynchro.readthedocs.io/en/stable/>`_.

Say that you have a code — such as `Rimphony
<https://github.com/pkgw/rimphony/>`_ or `Symphony
<https://github.com/AFD-Illinois/symphony>`_ — that calculates synchrotron
radiative transfer coefficients as a function of some input model parameters
(electron temperature, particle energy index, etc.). These calculations are
often accurate but slow. With *neurosynchro*, you can train a neural network
that will quickly approximate these calculations with good accuracy. The
achievable level of accuracy will depend on the particulars of your target
distribution function, range of input parameters, and so on.

This code is specific to synchrotron radiation because it makes certain
assumptions about how the coefficients scale with input parameters such as the
observing frequency.''',

    # Synchronize with README.md and doc/requirements.txt:
    install_requires = [
        'keras >=2.1',
        'numpy >=1.10',
        'pandas >=0.23.0',
        'pwkit >=0.8.19',
        'pytoml >=0.1.0',
        'six >=1.10',
    ],

    packages = [
        'neurosynchro',
    ],

    entry_points = {
        'console_scripts': [
            'neurosynchro = neurosynchro.cli:main',
        ],
    },

    include_package_data = True,

    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],

    keywords = 'neural-networks radiative-transfer science',
)
