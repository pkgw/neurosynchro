# neurosynchro

*Neurosynchro* is a small Python package for creating and using neural networks
to quickly approximate the coefficients needed for fully-polarized synchrotron
radiative transfer. It builds on the [Keras](https://keras.io/) deep learning
library.

Say that you have a code — such as
[Rimphony](https://github.com/pkgw/rimphony/) or
[Symphony](https://github.com/AFD-Illinois/symphony) — that calculates
synchrotron radiative transfer coefficients as a function of some input model
parameters (electron temperature, particle energy index, etc.). These
calculations are often accurate but slow. With *neurosynchro*, you can train a
neural network that will quickly approximate these calculations with good
accuracy. The achievable level of accuracy will depend on the particulars of
your target distribution function, range of input parameters, and so on.

This code is specific to synchrotron radiation because it makes certain
assumptions about how the coefficients scale with input parameters such as the
observing frequency.

Neurosynchro is written by Peter K. G. Williams (<pwilliams@cfa.harvard.edu>).

## Documentation

*Neurosynchro’s* documentation
 [is on ReadTheDocs](https://neurosynchro.readthedocs.io/en/stable/).

## Requirements

<!-- Keep synchronized with setup.py and doc/requirements.txt -->

- [keras](https://keras.io/) version 2.1 or greater.
- [numpy](https://www.numpy.org/) version 1.10 or greater.
- [pandas](https://pandas.pydata.org/) version 0.23.0 or greater.
- [pwkit](https://github.com/pkgw/pwkit/) version 0.8.19 or greater.
- [pytoml](https://github.com/avakar/pytoml) version 0.1.0 or greater.
- [six](https://six.readthedocs.io/) version 1.10 or greater.

## Recent Changes

See [the changelog](CHANGELOG.md).

## Copyright and License

This code is copyright Peter K. G. Williams and collaborators. It is licensed
under the [MIT License](https://opensource.org/licenses/MIT).
