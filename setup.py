#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators
# Licensed under the MIT License

from __future__ import absolute_import, division, print_function

from setuptools import setup, Extension

setup(
    name = 'neurosynchro',
    author = 'Peter Williams <peter@newton.cx>',
    version = '0.1.0',
    url = 'https://github.com/pkgw/neurosynchro/',
    license = 'MIT',
    description = ('Use neural networks to approximate polarized synchrotron '
                   'radiative transfer coefficients'),

    # Synchronize with README.md:
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
