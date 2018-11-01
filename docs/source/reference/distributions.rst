Probability Distributions
=========================

.. module:: chainer.distributions

Chainer provides many :class:`~chainer.Distribution` implementations in the
:mod:`chainer.distributions` package.


Distributions
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   
   chainer.distributions.Bernoulli
   chainer.distributions.Beta
   chainer.distributions.Categorical
   chainer.distributions.Dirichlet
   chainer.distributions.Exponential
   chainer.distributions.Gamma
   chainer.distributions.Laplace
   chainer.distributions.LogNormal
   chainer.distributions.MultivariateNormal
   chainer.distributions.Normal
   chainer.distributions.OneHotCategorical
   chainer.distributions.Pareto
   chainer.distributions.Poisson
   chainer.distributions.Uniform


Functionals of distribution
---------------------------

.. currentmodule:: chainer

.. autosummary::
  :toctree: generated/
  :nosignatures:
  
  chainer.cross_entropy
  chainer.kl_divergence
  chainer.register_kl


Base classes
------------

.. autosummary::
  :toctree: generated/
  :nosignatures:

  chainer.Distribution
