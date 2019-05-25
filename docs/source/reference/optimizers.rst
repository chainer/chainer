Optimizers
==========

.. module:: chainer.optimizers
.. currentmodule:: chainer
.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.optimizers.AdaDelta
   chainer.optimizers.AdaGrad
   chainer.optimizers.Adam
   chainer.optimizers.AdamW
   chainer.optimizers.AMSGrad
   chainer.optimizers.AdaBound
   chainer.optimizers.AMSBound
   chainer.optimizers.CorrectedMomentumSGD
   chainer.optimizers.MomentumSGD
   chainer.optimizers.NesterovAG
   chainer.optimizers.MSVAG
   chainer.optimizers.RMSprop
   chainer.optimizers.RMSpropGraves
   chainer.optimizers.SGD
   chainer.optimizers.SMORMS3

Optimizer base classes
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.Optimizer
   chainer.UpdateRule
   chainer.optimizer.Hyperparameter
   chainer.GradientMethod

Hook functions
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.optimizer_hooks.WeightDecay
   chainer.optimizer_hooks.Lasso
   chainer.optimizer_hooks.GradientClipping
   chainer.optimizer_hooks.GradientHardClipping
   chainer.optimizer_hooks.GradientNoise
   chainer.optimizer_hooks.GradientLARS
