Assertion and Testing
=====================

Chainer provides some facilities to make debugging easy.

.. _type-check-utils:

Type checking utilities
-----------------------
:class:`~chainer.FunctionNode` uses a systematic type checking of the :mod:`chainer.utils.type_check` module.
It enables users to easily find bugs of forward and backward implementations.
You can find examples of type checking in some function implementations.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.utils.type_check.Expr
   chainer.utils.type_check.expect
   chainer.utils.type_check.TypeInfo
   chainer.utils.type_check.TypeInfoTuple

Gradient checking utilities
---------------------------
Most function implementations are numerically tested by *gradient checking*.
This method computes numerical gradients of forward routines and compares their results with the corresponding backward routines.
It enables us to make the source of issues clear when we hit an error of gradient computations.
The :mod:`chainer.gradient_check` module makes it easy to implement the gradient checking.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.gradient_check.check_backward
   chainer.gradient_check.numerical_grad

Standard Assertions
-------------------

The assertions have same names as NumPy's ones.
The difference from NumPy is that they can accept both :class:`numpy.ndarray`
and :class:`cupy.ndarray`.

.. module:: chainer.testing
.. currentmodule:: chainer
.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.testing.assert_allclose
   chainer.testing.assert_warns

Function testing utilities
--------------------------

Utilities for testing functions.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.testing.FunctionTestCase
   chainer.testing.unary_math_function_unittest

Serialization testing utilities
-------------------------------

Utilities for testing serializable objects.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.testing.save_and_load
   chainer.testing.save_and_load_hdf5
   chainer.testing.save_and_load_npz
   chainer.testing.get_trainer_with_mock_updater

Trainer Extension Testing Utilities
-----------------------------------

Utilities for testing :ref:`trainer extensions <extensions>`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.testing.get_trainer_with_mock_updater

Repeat decorators
-----------------

These decorators have a decorated test run multiple times
in a single invocation. Criteria of passing / failing
of the test changes according to the type of decorators.
See the documentation of each decorator for details.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.testing.condition.repeat_with_success_at_least
   chainer.testing.condition.repeat
   chainer.testing.condition.retry


Unit test annotation
--------------------

Decorators for annotating unit tests.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.testing.attr.gpu
   chainer.testing.attr.multi_gpu
   chainer.testing.with_requires
   chainer.testing.fix_random


Parameterized test
------------------

Decorators for making a unit test parameterized.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.testing.parameterize
   chainer.testing.product
   chainer.testing.product_dict
   chainer.testing.inject_backend_tests
   
