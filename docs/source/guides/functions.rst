Define your own function
========================

.. currentmodule:: chainer

In this section, you will learn about the following things:

* How to define a function on variables
* Useful tools to write a function using a GPU
* How to test the function definition

After reading this section, you will be able to:

* Write your own functions
* Define simple kernels in the function definition

.. include:: ../imports.rst

Differentiable Functions
------------------------

Chainer provides a collection of functions in the :mod:`chainer.functions` module.
It covers typical use cases in deep learning, so many existing works can be implemented with them.
On the other hand, deep learning is evolving rapidly and we cannot cover all possible functions to define unseen architectures.
So it is important to learn how to define your own functions.

.. _new-vs-old-style-functions:

New-Style v.s. Old-Style Functions
----------------------------------

In Chainer, you can define a function in two ways: new-style and old-style.

* New-style functions inherit from :class:`chainer.FunctionNode` class (introduced in Chainer v3).
  Forward computation can be implemented using NumPy/CuPy.
  Backward computation needs to be implemented by using (possibly a composition of) other new-style functions.
* Old-style functions inherit from :class:`chainer.Function` class.
  Forward and backward computation can be implemented using NumPy/CuPy.

The primary advantage of using new-style functions is that they support computation of higher-order gradients (a.k.a. higher-order derivative or double backpropagation).
Higher-order gradients are used in some models e.g., recently-proposed GAN architectures.
New-style functions are also better in terms of performance of backward, as the interface allows an implementation to skip the computation of unneeded input gradients.

Currently, most of :doc:`built-in functions <../reference/functions>` are implemented in new-style (with a few exceptions listed in `#4449 <https://github.com/chainer/chainer/issues/4449>`__).
Basically, we recommend you use new-style when implementing new functions.
However, you can still continue to use existing old-style functions for the foreseeable future.

In the following sections, we describe steps to implenent user-defiend functions in new-style.
You can also refer to :ref:`implement-old-style-functions` and :ref:`migrate-from-old-style` if you have interest.

.. _implement-new-style-functions:

Implementing New-Style Functions
--------------------------------

First, suppose we want to define an elementwise function :math:`f(x, y, z) = x * y + z`.
While it is possible to implement this equation using a combination of the ``*`` and ``+`` functions,
defining it as a single function may reduce memory consumption, so it is *not* only a toy example.
Here we call this function *MulAdd*.

Let's start with defining MulAdd working on the CPU.
New-style functions must inherit the :class:`chainer.FunctionNode` class.
The skeleton of a function looks like:

.. testcode::

   class MulAdd(FunctionNode):
       def forward_cpu(self, inputs):
           # do forward computation on CPU
           return some_tuple

       def backward(self, target_input_indexes, grad_outputs):
           # do backward computation
           return some_tuple

We must implement :meth:`~FunctionNode.forward_cpu` and :meth:`~FunctionNode.backward` methods.

* In :meth:`~FunctionNode.forward_cpu` function, ``inputs`` is a tuple of array(s).
  You need to return a tuple of array(s), which is a result of forward computation.
* In :meth:`~FunctionNode.backward` function, ``grad_outputs`` is a tuple of :class:`~chainer.Variable`\ (s) which are gradients with regard to each output(s), i.e., the length of ``grad_outputs`` tuple equals to the number of outputs returned by ``forward_cpu``).
  You need to return a tuple of :class:`~chainer.Variable`\ (s) which are gradients with regard to each input(s), i.e., the length of returned tuple equals to the number of inputs to ``forward_cpu``.
  You can optionally use ``target_input_indexes`` (a tuple of indices required to compute gradients) to omit computing unnecessary gradients.
  We will show you the usage of ``target_input_indexes`` later.

.. warning::

   Be careful to return a tuple even if you have just one array or Variable to return.

.. note::

   Unlike old-style functions, inputs and outputs of backward method in new-style functions are :class:`~chainer.Variable`\s.
   In other words, the backward method is device agnostic; there are no ``backward_cpu`` or ``backward_gpu`` in :class:`~FunctionNode`.

MulAdd is simple and can be implemented as follows:

.. testcode::

   class MulAdd(FunctionNode):
       def forward_cpu(self, inputs):
           # Unpack input arrays (``numpy.ndarray``).
           x, y, z = inputs

           # Mark inputs (``x`` and ``y``) as retained so that it can be
           # accessed during the backward process.
           self.retain_inputs((0, 1))

           # Compute results.
           w = x * y + z

           # Return the result as a tuple.
           return w,

       def backward(self, target_input_indexes, grad_outputs):
           # Unpack inputs retained in the forward process (``Variable``).
           x, y = self.get_retained_inputs()

           # Get gradients w.r.t. the output (Variable).
           gw, = grad_outputs

           # Compute gradients w.r.t the inputs.
           gx = y * gw
           gy = x * gw
           gz = gw

           # Return the result as a tuple.
           return gx, gy, gz

.. testcode::
   :hide:

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   w, = MulAdd().apply((x, y, z))
   w.grad = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
   w.backward()

As per the warning above, the ``forward_cpu`` method returns a tuple of single element.
Note that all arrays appearing in ``forward_cpu`` method are :class:`numpy.ndarray`.
The forward function is straightforward; it unpacks the input tuple, computes the output, and packs it into a tuple.
The backward function is a bit more complicated.
Recall the rule of differentiation of multiplication.
This example just implements the rule.
Look at the return values, the function just packs the gradient of each input in the same order and returns them.

By just defining the core computation of forward and backward,
:class:`~chainer.FunctionNode` class provides a chaining logic on it (i.e., storing the
history of computation, etc.).

.. note::
   Assuming we implement a (forward) function :math:`y=f(x)` which takes as input the
   vector :math:`x \in \mathbb{R}^n` and produces as output a vector
   :math:`y \in \mathbb{R}^m`. Then the ``backward`` method has to compute

   .. math::
      \lambda_i = \sum_{j=1}^m \frac{\partial y_j}{\partial x_i} \,
      \gamma_j \,\, \text{for}\, i = 1 \dots n

   where :math:`\gamma` is the ``grad_outputs``. Note, that the
   resulting vector :math:`\lambda` must have the same shape as the arguments of the ``forward`` method.

Now let's define the corresponding GPU method.
You can easily predict that the method we have to write is named :meth:`~FunctionNode.forward_gpu`:

.. testcode::

   class MulAdd(FunctionNode):
       def forward_cpu(self, inputs):
           ...

       def forward_gpu(self, inputs):
           # Unpack input arrays (``cupy.ndarray``).
           x, y, z = inputs

           # Mark inputs (``x`` and ``y``) as retained so that it can be
           # accessed during the backward process.
           self.retain_inputs((0, 1))

           # Compute results.
           w = x * y + z

           # Return the result as a tuple.
           return w,

       def backward(self, target_input_indexes, grad_outputs):
           ...

In ``forward_gpu`` method, arrays are of type :class:`cupy.ndarray`.
We use arithmetic operators defined for this class.
These operators implement the basic elementwise arithmetics.

You may find that the definitions of ``forward_gpu`` is exactly same as ``forward_cpu``.
In that case, we can reduce them io :meth:`~FunctionNode.forward`.

.. testcode::

   class MulAdd(FunctionNode):
       def forward(self, inputs):
           # Unpack input arrays (``numpy.ndarray`` or ``cupy.ndarray``).
           x, y, z = inputs

           # Mark inputs (``x`` and ``y``) as retained so that it can be
           # accessed during the backward process.
           self.retain_inputs((0, 1))

           # Compute results.
           w = x * y + z

           # Return the result as a tuple.
           return w,

       def backward(self, inputs, grad_outputs):
           x, y, z = inputs
           gw, = grad_outputs

           gx = y * gw
           gy = x * gw
           gz = gw
           return gx, gy, gz

.. testcode::
   :hide:

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   w, = MulAdd().apply((x, y, z))
   w.grad = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
   w.backward()

Since the :class:`cupy.ndarray` class implements many methods of :class:`numpy.ndarray`, we can write these unified methods in most cases.

The MulAdd function can be used as follows:

.. testcode::

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   w, = MulAdd().apply((x, y, z))

It looks a bit ugly: we have to explicitly instantiate MulAdd before applying it to variables.
We also have to be careful that one instance of MulAdd must not be used multiple times, since it acts as a node in the computational graph.
In Chainer, we often define a thin wrapper Python function that hide the instantiation:

.. testcode::

   def muladd(x, y, z):
       return MulAdd().apply((x, y, z))

   w = muladd(x, y, z)

All functions under :mod:`chainer.functions` are implemented as wrapper functions like this.

Unified forward/backward methods with NumPy/CuPy functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy implements many functions that are compatible to those of NumPy.
We can write unified forward/backward methods with them.
Consider that we want to write a backprop-able function :math:`f(x, y) = \exp(x) + \exp(y)`.
We name it *ExpAdd* here.
It can be written straight-forward as follows:

.. testcode::

   from chainer.backends import cuda

   class ExpAdd(FunctionNode):
       def forward_cpu(self, inputs):
           self.retain_inputs((0, 1))
           x, y = inputs
           z = np.exp(x) + np.exp(y)
           return z,

       def forward_gpu(self, inputs):
           self.retain_inputs((0, 1))
           cupy = cuda.cupy
           x, y = inputs
           z = cupy.exp(x) + cupy.exp(y)
           return z,

       def backward(self, target_input_indexes, grad_outputs):
           x, y = self.get_retained_inputs()
           gz, = grad_outputs

           gx = gz * F.exp(x)
           gy = gz * F.exp(y)
           return gx, gy

   def expadd(x, y):
       z, = ExpAdd().apply((x, y))
       return z

.. testcode::
   :hide:

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = expadd(x, y)
   z.grad = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
   z.backward()

.. note::
   Here we used :mod:`chainer.backends.cuda.cupy` instead of directly accessing :mod:`cupy`.
   This is because the :mod:`cupy` module cannot be imported if the CUDA is not installed.
   In order to keep the implementation valid in non-CUDA environment, we have to defer the access to the ``cupy`` module.
   Note that the :mod:`chainer.backends.cuda` module can be imported even if the CUDA is not installed.
   Of course, the module in such environment is almost useless, but if the interpreter does not run through the code accessing CUDA-dedicated functions, the code is still valid.

The CPU and GPU implementations are almost same, except that :mod:`numpy` is replaced by :mod:`cupy` in ``forward_gpu``.
We can unify these functions using the :func:`chainer.backend.get_array_module` function.
This function accepts arbitrary number of arrays, and returns an appropriate module for them.
See the following code:

.. testcode::

   class ExpAdd(FunctionNode):
       def forward(self, inputs):
           self.retain_inputs((0, 1))
           xp = backend.get_array_module(*inputs)
           x, y = inputs
           z = xp.exp(x) + xp.exp(y)
           return z,

       def backward(self, target_input_indexes, grad_outputs):
           x, y = self.get_retained_inputs()
           gz, = grad_outputs

           gx = gz * F.exp(x)
           gy = gz * F.exp(y)
           return gx, gy

   def expadd(x, y):
       z, = ExpAdd().apply((x, y))
       return z

.. testcode::
   :hide:

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = expadd(x, y)
   z.grad = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
   z.backward()

Note that this code works correctly even if CUDA is not installed in the environment.
If CUDA is not found, :func:`~chainer.backend.get_array_module` function always returns :mod:`numpy`.
We often use the name ``xp`` for the variadic module name, which is analogous to the abbreviation ``np`` for NumPy and ``cp`` for CuPy.


Write an Elementwise Kernel Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's turn back to the MulAdd example.

The GPU implementation of MulAdd as shown above is already fast and parallelized on GPU cores.
However, it invokes two kernels during each of forward (``w = x * y + z``) and backward (``gx = y * gw`` and ``gy = x * gw``) computations.
It might hurt performance, since the intermediate temporary arrays are read and written by possibly different GPU cores, which consumes much bandwidth.
We can reduce the number of invocations by defining our own kernel.
It also reduce the memory consumption.

Most functions only require elementwise operations like MulAdd.
CuPy provides a useful tool to define elementwise kernels, the :class:`cupy.ElementwiseKernel` class, and Chainer wraps it by :func:`chainer.backends.cuda.elementwise` function.
Our MulAdd implementation can be improved as follows:

.. testcode::

   class MulAdd(FunctionNode):
       def forward_cpu(self, inputs):
           self.retain_inputs((0, 1))
           x, y, z = inputs
           w = x * y + z
           return w,

       def forward_gpu(self, inputs):
           self.retain_inputs((0, 1))
           x, y, z = inputs
           w = cuda.cupy.elementwise(
               'float32 x, float32 y, float32 z',
               'float32 w',
               'w = x * y + z',
               'muladd_fwd')(x, y, z)
           return w,

       def backward(self, target_input_indexes, grad_outputs):
           x, y, z = self.get_retained_inputs()
           gw, = grad_outputs
           return MulAddGrad().apply((x, y, z, gw))

   class MulAddGrad(FunctionNode):
       def forward_cpu(self, inputs):
           x, y, z, gw = inputs
           gx = y * gw
           gy = x * gw
           gz = gw
           return gx, gy, gz

       def forward_gpu(self, inputs):
           x, y, z, gw = inputs
           gx, gy = cuda.elementwise(
               'float32 x, float32 y, float32 gw',
               'float32 gx, float32 gy',
               '''
                  gx = y * gw;
                  gy = x * gw;
               ''',
               'muladd_bwd')(x, y, gw)

           gz = gw
           return gx, gy, gz

       def backward(self, target_input_indexes, grad_outputs):
           # You can leave this unimplemented unless you need to compute
           # higher-order derivative using this function.
           raise NotImplementedError()

:func:`chainer.backends.cuda.elementwise` function accepts the essential implementation of the kernel function, and returns a kernel invocation function (actually, it returns :class:`~cupy.ElementwiseKernel` object, which is callable).
In typical usage, we pass four arguments to this function as follows:

1. Input argument list. This is a comma-separated string each entry of which consists of a type specification and an argument name.
2. Output argument list in the same format as the input argument list.
3. Body of *parallel loop*. We can use the input/output argument names as an element of these arrays.
4. Name of the kernel function, which is shown in debuggers and profilers.

Above code is not compiled on every forward/backward computation thanks to two caching mechanisms provided by :func:`chainer.backends.cuda.elementwise`.

The first one is *binary caching*:
:func:`chainer.backends.cuda.elementwise` function caches the compiled binary in the ``$(HOME)/.cupy/kernel_cache`` directory with a hash value of the CUDA code, and reuses it if the given code matches the hash value.
This caching mechanism is actually implemented in CuPy.

The second one is *upload caching*:
Given a compiled binary code, we have to upload it to the current GPU in order to execute it.
:func:`chainer.backends.cuda.elementwise` function memoizes the arguments and the current device, and if it is called with the same arguments for the same device, it reuses the previously uploaded kernel code.

The above MulAdd code only works for float32 arrays.
The :class:`~cupy.ElementwiseKernel` also supports the type-variadic kernel definition.
In order to define variadic kernel functions, you can use *type placeholder* by placing a single character as type specifier:

.. testcode::

  class MulAdd(Function):
      def forward_cpu(self, inputs):
          ...

      def backward_cpu(self, inputs, grad_outputs):
          ...

      def forward_gpu(self, inputs):
          cupy = cuda.cupy
          x, y, z = inputs
          w = cuda.elementwise(
              'T x, T y, T z',
              'T w',
              'w = x * y + z',
              'muladd_fwd')(x, y, z)
          return w,

      def backward_gpu(self, inputs, grad_outputs):
          x, y, z = inputs
          gw, = grad_outputs

          gx, gy = cuda.elementwise(
              'T x, T y, T gw',
              'T gx, T gy',
              '''
                 gx = y * gw;
                 gy = x * gw;
              ''',
              'muladd_bwd')(x, y, gw)

          gz = gw
          return gx, gy, gz

The type placeholder ``T`` indicates an arbitrary data type that CuPy supports.

There are more functionalities on user-defined kernels in CuPy.
:ref:`See the CuPy documentation on user-defined kernels for more details. <udkernel>`

Advanced Topics
---------------

Write a function with training/test mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We sometimes want to make a function behave differently in training and test modes.
The training/test mode in Chainer is configured by :data:`chainer.config`.
This is a thread-local configuration object, and users can substitute True or False to its ``train`` attribute.
You can refer to :ref:`configuration` to see how to configure this flag as well as other configuration items.

Here, we just show how to use this flag to make a function support training/test mode.
You will need to check the value of the boolean flag ``chainer.config.train`` and branch appropriately.

For example, consider the following simple dropout function::

  def dropout(x):
      xp = backend.get_array_module(x.array)
      mask = 2 * (xp.random.rand(*x.shape) > 0.5).astype(x.dtype)
      return x * mask

This function applies dropout to each element and doubles survived elements to preserve the scale.
The above implementation applies dropout even in test mode, but it is not a desired behavior.
We can fix it as follows::

  def dropout(x):
      if not chainer.config.train:
          return x

      xp = backend.get_array_module(x.array)
      mask = 2 * (xp.random.rand(*x.shape) > 0.5).astype(x.dtype)
      return x * mask

The function now supports test mode.
Note that you usually do not have to implement your own dropout function because :func:`~chainer.functions.dropout` is officially provided.

Testing Functions
~~~~~~~~~~~~~~~~~

In order to isolate the cause of learning failure from implementation bugs, it is important to test function implementations.
Chainer provides simple utilities to help writing unit tests.
They are defined in the :mod:`~chainer.gradient_check` module.

The most important test utility is the :func:`~gradient_check.numerical_grad` function.
This function computes the numerical gradient of given function using finite differences.
It can be used as follows:

.. testcode::

   x  = np.random.randn(4, 3).astype(np.float32)
   gy = np.ones((4, 3), dtype=np.float32)
   f  = lambda: (x * x,)
   gx = gradient_check.numerical_grad(f, (x,), (gy,))

``f`` is a closure that returns a tuple of array(s) computed from input arrays.
The second and third arguments of :func:`~gradient_check.numerical_grad` are tuples of input arrays and output gradient arrays, respectively.
The code above computes the numerical gradients of ``sum(f(x))``, where ``sum`` indicates the summation over all elements.
The summation can be weighted by changing ``gy``.
:func:`~gradient_check.numerical_grad` function also accepts additional ``eps`` argument, which indicates the quantization width of finite differences.

.. note::

   :func:`~gradient_check.numerical_grad` function accepts both CPU and GPU arrays.
   Note that we cannot mix CPU and GPU arrays.

Another utility is :func:`chainer.testing.assert_allclose` function.
This is similar to :func:`numpy.testing.assert_allclose` function.
The difference is that Chainer's version accepts CPU and GPU arrays as inputs.
We can mix them in one invocation of :func:`chainer.testing.assert_allclose`.
The default values of optional arguments are also different.

Here is a typical usage of gradient checking utilities.
This is a test example of :func:`functions.relu` function:

.. testcode::

   import unittest

   from chainer import testing

   class TestReLU(unittest.TestCase):
       def test_backward_cpu(self):
           x = Variable(np.random.randn(3, 2).astype(np.float32))
           y = F.relu(x)
           y.grad = np.random.randn(3, 2).astype(np.float32)
           y.backward(retain_grad=True)

           def f():
               return F.relu(x).array,

           gx, = gradient_check.numerical_grad(f, (x.array,), (y.grad,))
           testing.assert_allclose(gx, x.grad)


.. testcode::
   :hide:

   suite = unittest.TestLoader().loadTestsFromTestCase(TestReLU)
   unittest.TextTestRunner().run(suite)


The first four lines of the test code are simple forward and backward computation of ReLU function.
The next two lines compute numerical gradient using the same forward function without backward routine.
And at last, we compare these two results elementwise.
Note that the above test code can be easily modified to test GPU version just by replacing CPU arrays to GPU arrays.

In most cases, we do not write the code like the above explicitly because Chainer
offers a utility function :func:`chainer.gradient_check.check_backward` that follows this procedure.

.. testcode::

   import unittest

   from chainer import gradient_check

   class TestReLU(unittest.TestCase):
       def test_backward_cpu(self):

           def f(x):
               return F.relu(x)

           x = np.random.randn(3, 2).astype(np.float32)
           y_grad = np.random.randn(3, 2).astype(np.float32)

           gradient_check.check_backward(f, x, y_grad, atol=1e-4, rtol=1e-4)

.. testcode::
   :hide:

   suite = unittest.TestLoader().loadTestsFromTestCase(TestReLU)
   unittest.TextTestRunner().run(suite)


You can find many examples of function tests under :tree:`tests/chainer_tests/functions_tests` directory.

You can use :func:`chainer.gradient_check.check_double_backward` to run gradient check for the second order gradient computed by new-style functions.
This function runs two backwpropagations; first to compute the gradient ``gx`` of ``y`` w.r.t. ``x``, and second to compute the gradient of ``gx`` w.r.t. ``x``.
It can be used like :func:`~chainer.gradient_check.check_backward`, but :func:`~chainer.gradient_check.check_double_backward` expects an additional argument ``x_grad_grad``, which is an array or a tuple of arrays used for initializing the gradient array of each gradient w.r.t. an input.
In other words, this argument is used to initialize ``gx.grad`` for the second backprop.


Implementing User-Defined Links
-------------------------------

Some functions are meant to be combined with parameters.
In such case, it is useful to write a small **link** that wraps the function.
We have already seen how to define a chain that wraps other links (by inheriting :class:`Chain` class) in :doc:`models`.
Here we study how to define a link that does not hold any other links.

As the first example, suppose that we want to implement elementwise product function between the input array and the parameter array.
It can be defined as follows:

.. testcode::

   class EltwiseParamProduct(Link):
       def __init__(self, shape):
           super(EltwiseParamProduct, self).__init__()
           with self.init_scope():
               self.W = chainer.Parameter(initializers.Normal(scale=1.), shape)

       def __call__(self, x):
           return self.W * x

For another example, assume we want to define a simple linear layer.
It is already defined as :class:`chainer.links.Linear`, so this is an educational example.
The linear layer is divided into two parts: a function and its wrapper link.
First, we have to define a function on variables:

.. testcode::

   class LinearFunction(FunctionNode):
       def forward(self, inputs):
           x, W, b = inputs
           return x.dot(W.T) + b,

       def backward(self, inputs, grad_outputs):
           x, W, b = inputs
           gy, = grad_outputs

           gx = gy.dot(W)
           gW = gy.T.dot(x)
           gb = gy.sum(axis=0)
           return gx, gW, gb

   def linear(x, W, b):
       return LinearFunction()(x, W, b)

This function takes three arguments: input, weight, and bias.
It can be used as a part of model definition, though is inconvenient since the user have to manage the weight and bias parameters directly.
In order to make a convenient module, let's wrap it into a link:

.. testcode::

   class Linear(Link):
       def __init__(self, in_size, out_size):
           super(Linear, self).__init__()
           with self.init_scope():
               self.W = chainer.Parameter(
                   initializers.Normal(1. / math.sqrt(in_size)),
                   (out_size, in_size))
               self.b = chainer.Parameter(0, (out_size,))

       def __call__(self, x):
           return linear(x, self.W, self.b)

This link hides the parameters of the linear layer.

.. note::

   An advanced tip to implement functions: if you want to preserve some information between forward and backward computations (e.g. to cache some arrays), you can store it as attributes.
   Be careful that it might increase the memory consumption during the whole forward-backward computation.
   If you want to train very large networks on a GPU with limited memory, it is not recommended to cache arrays between forward and backward.
   There is one exception for this: caching the output arrays does not change the memory consumption, because they are also held by the output Variable objects.

   .. warning::

      You should not assume a one-to-one match of calls of forward and backward.
      Some users may call backward more than once after one forward call.



.. _migrate-from-old-style:

Migrating From Old-Style Functions To New-Style Functions
---------------------------------------------------------

Here are the key differences between :class:`Function` and :class:`FunctionNode`.

* Implementing forward computation (difference between :meth:`chainer.Function.forward` and :meth:`chainer.FunctionNode.forward`)

    * There are no difference between :class:`Function` and :class:`FunctionNode` except that the input arrays are NOT retained by default.

      If you want the inputs to be retained to use them in ``backward``, call :meth:`~chainer.FunctionNode.retain_inputs` explicitly.
      In other words, ``self.retain_inputs(())`` has no effect in :class:`FunctionNode`.

* Implementing backward computation (difference between :meth:`chainer.Function.backward` and :meth:`chainer.FunctionNode.backward`)

    * Arguments to the method has been changed.

        * ``inputs`` argument is no longer passed.

          You can use :meth:`~chainer.FunctionNode.get_retained_inputs` and :meth:`~chainer.FunctionNode.get_retained_outputs` to retrieve the inputs/outputs retained in the ``forward`` method.
          Note that ``grad_outputs`` and these retained inputs/outputs are all given as :class:`Variable` objects, and ``backward`` method must return a tuple of :class:`Variable` objects.

        * ``target_input_indexes`` argument has been added.

          It contains a sorted indices of the input variables w.r.t. which the gradients are required.
          You can use it to skip calculation of unneeded gradients.
          The use of ``target_input_indexes`` is optional; it is acceptable to calculate and return all gradients.

    * All inputs (``grad_outputs``) and retained values are given in :class:`~chainer.Variable` in :class:`~chainer.FunctionNode`, whereas ``ndarray`` in :class:`~chainer.Function`.

* Invoking forward computation

    * :class:`Function` is a callable, whereas :class:`FunctionNode` is not.

      You need to use ``f.apply((x,))`` instead of ``f(x)``.
      Note that :meth:`~chainer.FunctionNode.apply` always returns outputs as :class:`tuple` even if the function generates only one output value.


When migrating from old-style to new-style, typically you will need to write a new function class that implements the first-order gradient of the original function.
Here is an example of rewriting old-style ``MyOldFunc`` unary function to new-style ``MyFunc`` function.

.. testcode::

    class MyOldFunc(chainer.Function):

        def forward(self, inputs):
            x, = inputs
            ...  # forward computation code
            return y,

        def backward(self, inputs, grad_outputs):
            x, = inputs
            gy, = grad_outputs
            ...  # backward computation code
            return gx,

.. testcode::

    class MyFunc(chainer.FunctionNode):

        def forward(self, inputs):
            self.retain_inputs((0,))
            x, = inputs
            ...  # forward computation code in MyOldFunc
            return y,

        def backward(self, target_input_indexes, grad_outputs):
            x, = self.get_retained_inputs()
            gy, = grad_outputs
            gx, = MyFuncGrad().apply((x, gy))
            return gx,

    class MyFuncGrad(chainer.FunctionNode):

        def forward(self, inputs):
            x, gy = inputs
            ...  # backward computation code in MyOldFunc
            return gx,

        def backward(self, target_input_indexes, grad_outputs):
            # You can leave this unimplemented unless you need to compute
            # higher-order derivative using this function.
            raise NotImplementedError()



.. _implement-old-style-functions:

Implementing Old-Style Functions
--------------------------------

.. note::

    As noted in the :ref:`new-vs-old-style-functions`, we recommend you to use new-style for newly implemented functions.
    This section uses the same example as in :ref:`implement-new-style-functions` but using old-style.

First, suppose we want to define an elementwise function :math:`f(x, y, z) = x * y + z`.
While it is possible to implement this equation using a combination of the ``*`` and ``+`` functions,
defining it as a single function may reduce memory consumption, so it is *not* only a toy example.
Here we call this function *MulAdd*.

Let's start with defining MulAdd working on the CPU.
Old-style functions must inherit the :class:`Function` class.
The skeleton of a function looks like:

.. testcode::

   class MulAdd(Function):
       def forward_cpu(self, inputs):
           # do forward computation on CPU
           return some_tuple

       def backward_cpu(self, inputs, grad_outputs):
           # do backward computation on CPU
           return some_tuple

We must implement :meth:`~Function.forward_cpu` and :meth:`~Function.backward_cpu` methods.
The non-self arguments of these functions are tuples of array(s), and these functions must return a tuple of array(s).

.. warning::

   Be careful to return a tuple of arrays even if you have just one array to return.

MulAdd is simple and implemented as follows:

.. testcode::

   class MulAdd(Function):
       def forward_cpu(self, inputs):
           x, y, z = inputs
           w = x * y + z
           return w,

       def backward_cpu(self, inputs, grad_outputs):
           x, y, z = inputs
           gw, = grad_outputs

           gx = y * gw
           gy = x * gw
           gz = gw
           return gx, gy, gz

.. testcode::
   :hide:

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   w = MulAdd()(x, y, z)
   w.grad = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
   w.backward()

As per the warning above, the ``forward_cpu`` method returns a tuple of single element.
Note that all arrays appearing in CPU functions are :class:`numpy.ndarray`.
The forward function is straightforward; it unpacks the input tuple, computes the output, and packs it into a tuple.
The backward function is a bit more complicated.
Recall the rule of differentiation of multiplication.
This example just implements the rule.
Look at the return values, the function just packs the gradient of each input in the same order and returns them.

By just defining the core computation of forward and backward,
:class:`~chainer.Function` class provides a chaining logic on it (i.e., storing the
history of computation, etc.).

.. note::
   Assuming we implement a (forward) function :math:`y=f(x)` which takes as input the
   vector :math:`x \in \mathbb{R}^n` and produces as output a vector
   :math:`y \in \mathbb{R}^m`. Then the ``backward`` method has to compute

   .. math::
      \lambda_i = \sum_{j=1}^m \frac{\partial y_j}{\partial x_i} \,
      \gamma_j \,\, \text{for}\, i = 1 \dots n

   where :math:`\gamma` is the ``grad_outputs``. Note, that the
   resulting vector :math:`\lambda` must have the same shape as the arguments of the ``forward`` method.

Now let's define the corresponding GPU methods.
You can easily predict that the methods we have to write are named :meth:`~Function.forward_gpu` and :meth:`~Function.backward_gpu`:

.. testcode::

  class MulAdd(Function):
      def forward_cpu(self, inputs):
          ...

      def backward_cpu(self, inputs, grad_outputs):
          ...

      def forward_gpu(self, inputs):
          x, y, z = inputs
          w = x * y + z
          return w,

      def backward_gpu(self, inputs, grad_outputs):
          x, y, z = inputs
          gw, = grad_outputs

          gx = y * gw
          gy = x * gw
          gz = gw
          return gx, gy, gz

In GPU methods, arrays are of type :class:`cupy.ndarray`.
We use arithmetic operators defined for this class.
These operators implement the basic elementwise arithmetics.

You may find that the definitions of GPU methods are exactly same as those of CPU methods.
In that case, we can reduce them to :meth:`~Function.forward` and :meth:`~Function.backward` methods.

.. testcode::

   class MulAdd(Function):
       def forward(self, inputs):
           x, y, z = inputs
           w = x * y + z
           return w,

       def backward(self, inputs, grad_outputs):
           x, y, z = inputs
           gw, = grad_outputs

           gx = y * gw
           gy = x * gw
           gz = gw
           return gx, gy, gz

.. testcode::
   :hide:

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   w = MulAdd()(x, y, z)
   w.grad = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
   w.backward()

Since the :class:`cupy.ndarray` class implements many methods of :class:`numpy.ndarray`, we can write these unified methods in most cases.

The MulAdd function can be used as follows:

.. testcode::

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   w = MulAdd()(x, y, z)

It looks a bit ugly: we have to explicitly instantiate MulAdd before applying it to variables.
We also have to be careful that one instance of MulAdd must not be used multiple times, since it acts as a node in the computational graph.
In Chainer, we often define a thin wrapper Python function that hide the instantiation:

.. testcode::

   def muladd(x, y, z):
       return MulAdd()(x, y, z)

   w = muladd(x, y, z)

All functions under :mod:`chainer.functions` are implemented as wrapper functions like this.

Unified forward/backward methods with NumPy/CuPy functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy implements many functions that are compatible to those of NumPy.
We can write unified forward/backward methods with them.
Consider that we want to write a backprop-able function :math:`f(x, y) = \exp(x) + \exp(y)`.
We name it *ExpAdd* here.
It can be written straight-forward as follows:

.. testcode::

   from chainer.backends import cuda

   class ExpAdd(Function):
       def forward_cpu(self, inputs):
           x, y = inputs
           z = np.exp(x) + np.exp(y)
           return z,

       def backward_cpu(self, inputs, grad_outputs):
           x, y = inputs
           gz, = grad_outputs

           gx = gz * np.exp(x)
           gy = gz * np.exp(y)
           return gx, gy

       def forward_gpu(self, inputs):
           cupy = cuda.cupy
           x, y = inputs
           z = cupy.exp(x) + cupy.exp(y)
           return z,

       def backward_gpu(self, inputs, grad_outputs):
           cupy = cuda.cupy
           x, y = inputs
           gz, = grad_outputs

           gx = gz * cupy.exp(x)
           gy = gz * cupy.exp(y)
           return gx, gy

   def expadd(x, y):
       return ExpAdd()(x, y)


.. testcode::
   :hide:

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = expadd(x, y)
   z.grad = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
   z.backward()


.. note::
   Here we used :mod:`chainer.backends.cuda.cupy` instead of directly accessing :mod:`cupy`.
   This is because the :mod:`cupy` module cannot be imported if the CUDA is not installed.
   In order to keep the implementation valid in non-CUDA environment, we have to defer the access to the ``cupy`` module.
   Note that the :mod:`chainer.backends.cuda` module can be imported even if the CUDA is not installed.
   Of course, the module in such environment is almost useless, but if the interpreter does not run through the code accessing CUDA-dedicated functions, the code is still valid.

The CPU and GPU implementations are almost same, except that :mod:`numpy` is replaced by :mod:`cupy` in GPU methods.
We can unify these functions using the :func:`chainer.backend.get_array_module` function.
This function accepts arbitrary number of arrays, and returns an appropriate module for them.
See the following code:

.. testcode::

   class ExpAdd(Function):
       def forward(self, inputs):
           xp = backend.get_array_module(*inputs)
           x, y = inputs
           z = xp.exp(x) + xp.exp(y)
           return z,

       def backward(self, inputs, grad_outputs):
           xp = backend.get_array_module(*inputs)
           x, y = inputs
           gz, = grad_outputs

           gx = gz * xp.exp(x)
           gy = gz * xp.exp(y)
           return gx, gy

   def expadd(x, y):
       return ExpAdd()(x, y)

.. testcode::
   :hide:

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = expadd(x, y)
   z.grad = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
   z.backward()

Note that this code works correctly even if CUDA is not installed in the environment.
If CUDA is not found, :func:`~chainer.backend.get_array_module` function always returns :mod:`numpy`.
We often use the name ``xp`` for the variadic module name, which is analogous to the abbreviation ``np`` for NumPy and ``cp`` for CuPy.


Write an Elementwise Kernel Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's turn back to the MulAdd example.

The GPU implementation of MulAdd as shown above is already fast and parallelized on GPU cores.
However, it invokes two kernels during each of forward (``w = x * y + z``) and backward (``gx = y * gw`` and ``gy = x * gw``) computations.
It might hurt performance, since the intermediate temporary arrays are read and written by possibly different GPU cores, which consumes much bandwidth.
We can reduce the number of invocations by defining our own kernel.
It also reduce the memory consumption.

Most functions only require elementwise operations like MulAdd.
CuPy provides a useful tool to define elementwise kernels, the :class:`cupy.ElementwiseKernel` class, and Chainer wraps it by :func:`chainer.backends.cuda.elementwise` function.
Our MulAdd implementation can be improved as follows:

.. testcode::

  class MulAdd(Function):
      def forward_cpu(self, inputs):
          ...

      def backward_cpu(self, inputs, grad_outputs):
          ...

      def forward_gpu(self, inputs):
          cupy = cuda.cupy
          x, y, z = inputs
          w = cuda.elementwise(
              'float32 x, float32 y, float32 z',
              'float32 w',
              'w = x * y + z',
              'muladd_fwd')(x, y, z)
          return w,

      def backward_gpu(self, inputs, grad_outputs):
          x, y, z = inputs
          gw, = grad_outputs

          gx, gy = cuda.elementwise(
              'float32 x, float32 y, float32 gw',
              'float32 gx, float32 gy',
              '''
                 gx = y * gw;
                 gy = x * gw;
              ''',
              'muladd_bwd')(x, y, gw)

          gz = gw
          return gx, gy, gz

:func:`chainer.backends.cuda.elementwise` function accepts the essential implementation of the kernel function, and returns a kernel invocation function (actually, it returns :class:`~cupy.ElementwiseKernel` object, which is callable).
In typical usage, we pass four arguments to this function as follows:

1. Input argument list. This is a comma-separated string each entry of which consists of a type specification and an argument name.
2. Output argument list in the same format as the input argument list.
3. Body of *parallel loop*. We can use the input/output argument names as an element of these arrays.
4. Name of the kernel function, which is shown in debuggers and profilers.

Above code is not compiled on every forward/backward computation thanks to two caching mechanisms provided by :func:`chainer.backends.cuda.elementwise`.

The first one is *binary caching*:
:func:`chainer.backends.cuda.elementwise` function caches the compiled binary in the ``$(HOME)/.cupy/kernel_cache`` directory with a hash value of the CUDA code, and reuses it if the given code matches the hash value.
This caching mechanism is actually implemented in CuPy.

The second one is *upload caching*:
Given a compiled binary code, we have to upload it to the current GPU in order to execute it.
:func:`chainer.backends.cuda.elementwise` function memoizes the arguments and the current device, and if it is called with the same arguments for the same device, it reuses the previously uploaded kernel code.

The above MulAdd code only works for float32 arrays.
The :class:`~cupy.ElementwiseKernel` also supports the type-variadic kernel definition.
In order to define variadic kernel functions, you can use *type placeholder* by placing a single character as type specifier:

.. testcode::

  class MulAdd(Function):
      def forward_cpu(self, inputs):
          ...

      def backward_cpu(self, inputs, grad_outputs):
          ...

      def forward_gpu(self, inputs):
          cupy = cuda.cupy
          x, y, z = inputs
          w = cuda.elementwise(
              'T x, T y, T z',
              'T w',
              'w = x * y + z',
              'muladd_fwd')(x, y, z)
          return w,

      def backward_gpu(self, inputs, grad_outputs):
          x, y, z = inputs
          gw, = grad_outputs

          gx, gy = cuda.elementwise(
              'T x, T y, T gw',
              'T gx, T gy',
              '''
                 gx = y * gw;
                 gy = x * gw;
              ''',
              'muladd_bwd')(x, y, gw)

          gz = gw
          return gx, gy, gz

The type placeholder ``T`` indicates an arbitrary data type that CuPy supports.

There are more functionalities on user-defined kernels in CuPy.
:ref:`See the CuPy documentation on user-defined kernels for more details. <udkernel>`
