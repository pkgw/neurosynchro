.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

.. _precalculate-domain-range:

Precalculate the Domain and Range of Your Training Set
======================================================

There’s one more piece of preparation to do before actually training the
neural networks. For efficiency, quantities like the bounds on the physical
input and output parameters are gathered from the training set just once, then
saved in ``nn_config.toml``. You can gather this information with the
following command::

  $ neurosynchro lock-domain-range transformed .

Here ``transformed`` is the path to *transformed* training set and ``.`` is
the path to the directory containing your ``nn_config.toml`` file — which, in
the standard tutorial layout, is the current directory. This command will
rewrite this file, embedding the needed quantities.

**Next**: :ref:`train your neural networks! <train-your-networks>`
