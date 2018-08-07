.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

.. _train-your-networks:

Train Your Networks
===================

We’re finally ready to train your networks! After all of our preparatory work,
the training is pretty straightforward. For each output parameter, one runs a
command like::

  $ neurosynchro train transformed . j_I

Here, ``transformed`` is the path to the directory with the transformed
training set, ``.`` is the result directory containing the ``nn_config.toml``
file (the current directory, in the standard tutorial layout), and ``j_I`` is
the name of the component to train for. After training is complete, a file
named ``j_I.h5`` will be saved in the result directory. The program will print
out the mean squared error (MSE) characterizing the neural network’s
performance against the training set.

When training against the training set used in the tutorial, it takes about 20
minutes to train on each parameter when using an 8-core laptop CPU. Because
there are 9 parameters to train, this means that the training might take
something like 3 hours in total. (No substantial effort has gone into
optimizing the training process!) If you’re ready to commit to that, you can
train all of the parameters in sequence with::

  $ all="j_I alpha_I rho_Q rho_V j_frac_pol alpha_frac_pol j_V_share alpha_V_share rho_Q_sign"
  $ for p in $all ; do neurosynchro train transformed . $p ; done

If you pass the argument ``-p`` to the ``train`` subcommand, diagnostic plots
will be shown after training is complete. The plots will be made with the
obscure `omegaplot <https://github.com/pkgw/omegaplot>`_ package, so make sure
to install it before trying this option.


.. _trainer-types:

Trainer Types
~~~~~~~~~~~~~

*Neurosynchro* supports the following neural network training schemes. For
each output parameter, you can specify which scheme to use by editing its
``trainer`` keyword in the ``nn_config.toml`` file.

*generic*
  This neural network has the following characteristics:

  * Dense, single-layer architecture
  * 300 neurons
  * `RELU activation <https://keras.io/activations/#relu>`_.
  * Keras’s ``normal`` kernel initializer
  * Trained with the `adam optimizer <https://keras.io/optimizers/#adam>`_.
  * Optimized against the `mean-squared-error (MSE)
    <https://keras.io/losses/#mean_squared_error>`_ loss function.

  The network is trained in two passes. First, 30 epochs of training are run.
  Then the training set is sigma-clipped with ±7σ tolerance — the intention is
  to remove any cases where the detailed calculation has mistakenly delivered
  totally bogus results. Then 30 more epochs of training are run.

  This network has been observed to perform well in a variety of real-world
  situations.

*twolayer*
  This neural network has the following characteristics:

  * Dense, two-layer architecture
  * 120 neurons in first layer, 60 in second
  * `RELU activation <https://keras.io/activations/#relu>`_ in both layers.
  * Keras’s ``normal`` kernel initializer
  * Trained with the `adam optimizer <https://keras.io/optimizers/#adam>`_.
  * Optimized against the `mean-squared-error (MSE)
    <https://keras.io/losses/#mean_squared_error>`_ loss function.

  The training is run in the same way as in the ``generic`` setup. This
  network has been observed to perform a little bit better than the generic
  network when predicting the *rho_Q* output parameter. This doesn’t always
  hold, though; if you wish to investigate, try both and see which gives a
  better MSE.

*binary*
  This neural network has the following characteristics:

  * Dense, two-layer architecture
  * 120 neurons in first layer, 60 in second
  * `RELU activation <https://keras.io/activations/#relu>`_ in both layers.
  * `Sigmoid activation <https://keras.io/activations/#sigmoid>`_ in the output layer.
  * Keras’s ``normal`` kernel initializer
  * Trained with the `adam optimizer <https://keras.io/optimizers/#adam>`_.
  * Optimized against the `binary cross-entropy
    <https://keras.io/losses/#binary_crossentropy>`_ loss function.

  The training is run in almost the same way as in the ``generic`` setup, but
  no sigma-clipping is performed. This setup is intended for the *rho_Q_sign*
  output parameter, which predicts the sign of the *rho_Q* coefficient.
  However, sometimes the ``generic`` scheme actually performs better in
  practice. Once again, investigate by trying both and seeing which gives a
  better MSE.

This menu of options is, obviously, quite limited. For novel applications, you
may have to edit the code to add new training schemes. `Pull requests
<https://github.com/pkgw/neurosynchro/pulls>`_ contributing new ones are more
than welcome!

**Next**: :ref:`run some test problems! <run-test-problems>`
