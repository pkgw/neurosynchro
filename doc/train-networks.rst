.. Copyright 2018 Peter K. G. Williams and collaborators. Licensed under the
   Creative Commons Attribution-ShareAlike 4.0 International License.

.. _train-your-networks:

Train Your Networks
===================

We’re finally ready to train your networks! After all of our preparatory work,
the training is pretty straightforward. For each output parameter, run a
command like::

  $ neurosynchro train xformdir nndir j_I

Here, ``xformdir`` is the path to the directory with the transformed training
set, ``nndir`` is the path to the result directory, and ``j_I`` is the name of
the component to train for. After training is complete, a file named
``j_I.h5`` will be saved in the result directory. The program will print out
the mean squared error (MSE) characterizing the neural network’s performance
against the training set.

In the author’s workflow, training typically takes about 20 minutes for each
parameter. No substantial effort has gone into optimizing the training
process.

If you pass the argument ``-p`` to the ``train`` subcommand, diagnostic plots
will be shown after training is complete. The plots will be made with the
obscure `omegaplot <https://github.com/pkgw/omegaplot>`_ package, so make sure
to install it before using this option.


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
  no sigma-clipping is performed. This setup is optimize for the *rho_Q_sign*
  output parameter, which predicts the sign of the *rho_Q* coefficient.

This menu of options is, obviously, quite limited. For novel applications, you
may have to edit the code to add new training schemes. `Pull requests
<https://github.com/pkgw/neurosynchro/pulls>`_ contributing new ones are more
than welcome!

**Next**: :ref:`run some test problems! <run-test-problems>`
