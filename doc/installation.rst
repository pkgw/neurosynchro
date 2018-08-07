.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

How to Install *neurosynchro*
=============================

*Neurosynchro* is a pure-Python package, so it is not difficult to install by
itself. However, it requires heavyweight Python libraries like `Keras
<https://keras.io/>`_, which contain a lot of compiled code, so the
recommended means of installation is though an `Anaconda Python
<https://conda.io/docs/user-guide/install/index.html>`_ system.

Pre-built packages of *neurosynchro* are available through `conda-forge
<https://conda-forge.org/>`_, a community-based project that uses the `conda
<https://conda.io/docs/>`_ package manager. If you are using a ``conda``-based
Python installation, the quickest path to getting *neurosynchro* going is to
run the following commands in your terminal::

  $ conda config --add channels conda-forge
  $ conda install neurosynchro

Note, however, that ``conda-forge`` provides rebuilds of virtually every
package in the stock Anaconda system, and this installation method will
configure your system to prefer them. You may suddenly see lots of package
version changes related to this. You can install *neurosynchro* without
committing to using ``conda-forge`` for everything by skipping the ``conda
config`` command and instead running::

  $ conda install -c conda-forge neurosynchro

In the author’s experience, however, it is better to go the first route. This
latter route won’t get package updates from ``conda-forge`` and can lead to
vexing dependency mismatches further down the line.

Finally, it is also possible to try to install *neurosynchro* through `pip
<https://pip.pypa.io/en/stable/>`_ in the standard fashion, with ``pip install
neurosynchro``. However, as mentioned above, this can easily get hairy if one
of the big binary dependencies isn’t available. See `neurosynchro’s
requirements list <https://github.com/pkgw/neurosynchro/#requirements>`_ for
a list of what you’ll need.
