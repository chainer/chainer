Introduction to Chainer
-----------------------

.. currentmodule:: chainer

This is the first section of the Chainer Tutorial.
In this section, you will learn about the following things:

* Pros and cons of existing frameworks and why we are developing Chainer
* Simple example of forward and backward computation
* Usage of primitive links and their gradient computation
* Construction of user-defined links (a.k.a. "model" in most frameworks)
* Parameter optimization
* Serialization of links and optimizers

After reading this section, you will be able to:

* Compute gradients of some arithmetics
* Write a multi-layer perceptron with Chainer


Core Concept
~~~~~~~~~~~~

As mentioned on the front page, Chainer is a flexible framework for neural networks.
One major goal is flexibility, so it must enable us to write complex architectures simply and intuitively.

Most existing deep learning frameworks are based on the **"Define-and-Run"** scheme.
That is, first a network is defined and fixed, and then the user periodically feeds it with minibatches.
Since the network is statically defined before any forward/backward computation, all the logic must be embedded into the network architecture as *data*.
Consequently, defining a network architecture in such systems (e.g. Caffe) follows a declarative approach.
Note that one can still produce such a static network definition using imperative languages (e.g. Torch7 and Theano-based frameworks).

In contrast, Chainer adopts a **"Define-by-Run"** scheme, i.e., the network is defined on-the-fly via the actual forward computation.
More precisely, Chainer stores the history of computation instead of programming logic.
This strategy enables to fully leverage the power of programming logic in Python.
For example, Chainer does not need any magic to introduce conditionals and loops into the network definitions.
The Define-by-Run scheme is the core concept of Chainer.
We will show in this tutorial how to define networks dynamically.

This strategy also makes it easy to write multi-GPU parallelization, since logic comes closer to network manipulation.
We will review such amenities in later sections of this tutorial.


.. note::

   In example codes of this tutorial, we assume for simplicity that the following symbols are already imported::

     import numpy as np
     from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
     from chainer import Link, DictLink, ListLink
     import chainer.functions as F

   
   These imports appear widely in Chainer's codes and examples. For simplicity, we omit this idiom in this tutorial.


Forward/Backward Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As described above, Chainer uses "Define-by-Run" scheme, so forward computation itself *defines* the network.
In order to start forward computation, we have to set the input array to :class:`Variable` object.
Here we start with simple :class:`~numpy.ndarray` with only one element:

.. doctest::

   >>> x_data = np.array([5], dtype=np.float32)
   >>> x = Variable(x_data)

.. warning::

   Chainer currently only supports 32-bit float for most computations.

A Variable object has basic arithmetic operators.
In order to compute :math:`y = x^2 - 2x + 1`, just write:

.. doctest::

   >>> y = x**2 - 2 * x + 1

The resulting ``y`` is also a Variable object, whose value can be extracted by accessing the :attr:`~Variable.data` attribute:

.. doctest::

   >>> y.data
   array([ 16.], dtype=float32)

What ``y`` holds is not only the result value.
It also holds the history of computation (or computational graph), which enables us to compute its differentiation.
This is done by calling its :meth:`~Variable.backward` method:

.. doctest::

   >>> y.backward()

This runs *error backpropagation* (a.k.a. *backprop* or *reverse-mode automatic differentiation*).
Then, the gradient is computed and stored in the :attr:`~Variable.grad` attribute of the input variable ``x``:

.. doctest::

   >>> x.grad
   array([ 8.], dtype=float32)

Also we can compute gradients of intermediate variables.
Note that Chainer, by default, releases the gradient arrays of intermediate variables for memory efficiency.
In order to preserve gradient information, pass the ``retain_grad`` argument to the backward method:

.. doctest::

   >>> z = 2*x
   >>> y = x**2 - z + 1
   >>> y.backward(retain_grad=True)
   >>> z.grad
   array([-1.], dtype=float32)

All these computations are easily generalized to multi-element array input.
Note that if we want to start backward computation from a variable holding a multi-element array, we must set the *initial error* manually.
This is simply done by setting the :attr:`~Variable.grad` attribute of the output variable:

.. doctest::

   >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
   >>> y = x**2 - 2*x + 1
   >>> y.grad = np.ones((2, 3), dtype=np.float32)
   >>> y.backward()
   >>> x.grad
   array([[  0.,   2.,   4.],
          [  6.,   8.,  10.]], dtype=float32)

.. note::

   Many functions taking :class:`Variable` object(s) are defined in the :mod:`functions` module.
   You can combine them to realize complicated functions with automatic backward computation.


Primitive Links
~~~~~~~~~~~~~~~

In order to write neural networks, we have to combine functions with *parameters* and optimize the parameters.
You can use **links** to do this.
Link is an object that holds parameters (i.e. optimization targets) and internal states (if needed).

The most fundamental ones are links that behave like regular functions while replacing some arguments by their parameters.
Such links are called *primitive links*.
We will later state the defail of the concept of links, but here think primitive links just like parameterized functions.

.. note::
   Actually, these are corresponding to "parameterized functions" in versions up to v1.3.

One of the most frequently-used primitive links is the :class:`~functions.Linear` link (a.k.a. *fully-connected layer* or *affine transformation*).
It represents a mathematical function :math:`f(x) = Wx + b`, where the matrix :math:`W` and the vector :math:`b` are parameters.
This link is corresponding to its pure counterpart :func:`~functions.linear`, which accepts :math:`x, W, b` as arguments.
A linear link from three-dimensional space to two-dimensional space is defined by:

.. doctest::

   >>> f = F.Linear(3, 2)

.. note::
   Most functions and primitive links only accept minibatch input, where the first dimension of input arrays is considered as the *batch dimension*.
   In the above Linear link case, input must have shape of (N, 3), where N is the minibatch size.

Parameters of a link is stored in :attr:`~Link.params` attributes.
This is a dictionary with string keys.
Each parameter is stored as a Variable object.
Linear link has two parameters: `'W'` and `'b'`.
By default, the matrix W is initialized randomly, while the vector b is initialized with zeros.

.. doctest::

   >>> f.parmas['W'].data
   array([[ 1.01847613,  0.23103087,  0.56507462],
          [ 1.29378033,  1.07823515, -0.56423163]], dtype=float32)
   >>> f.params['b'].data
   array([ 0.,  0.], dtype=float32)

Instances of a link class act like usual functions:

.. doctest::

   >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
   >>> y = f(x)
   >>> y.data
   array([[ 3.1757617 ,  1.75755572],
          [ 8.61950684,  7.18090773]], dtype=float32)

Gradients of parameters are computed by :meth:`~Variable.backward` method.
Note that gradients are **accumulated** by the method rather than overwritten.
So first you must initialize gradients to zero to renew the computation.
It can be done by calling the :meth:`~Link.zerograds` method.

.. doctest::

   >>> f.zerograds()

Now we can compute the gradients of parameters by simply calling backward method.

.. doctest::

   >>> y.grad = np.ones((2, 2), dtype=np.float32)
   >>> y.backward()
   >>> f.params['W'].grad
   array([[ 5.,  7.,  9.],
          [ 5.,  7.,  9.]], dtype=float32)
   >>> f.params['b'].grad
   array([ 2.,  2.], dtype=float32)


Write a model as a link
~~~~~~~~~~~~~~~~~~~~~~~

Most neural network architectures contain multiple links.
For example, a multi-layer perceptron consists of multiple fully-connected layers.
We can write complex procedures with parameters by combining multiple links like:

.. doctest::

   >>> l1 = F.Linear(4, 3)
   >>> l2 = F.Linear(3, 2)
   >>> def my_forward(x):
   ...     h = l1(x)
   ...     return l2(h)

A procedure with parameters defined in this way is hard to reuse.
More Pythonic way is combining the links and procedures into a class:

.. doctest::

   >>> class MyProc(object):
   ...     def __init__(self):
   ...         self.l1 = F.Linear(4, 3)
   ...         self.l2 = F.Linear(3, 2)
   ...         
   ...     def forward(self, x):
   ...         h = self.l1(x)
   ...         return self.l2(h)

In order to make it more reusable, we want to support parameter management, CPU/GPU migration support, robust and flexible save/load features, etc.
These features are all supported by the :class:`Link` class in Chainer.
Then, what we have to do here is just defining the above class as a subclass of Link.

In this case, we can use the :class:`DictLink` class, which behaves like a dictionary of links:

.. doctest::

   >>> class MyLink(DictLink):
   ...     def __init__(self):
   ...         super(MyLink, self).__init__(
   ...             l1=F.Linear(4, 3),
   ...             l2=F.Linear(3, 2),
   ...         )
   ...        
   ...     def __call__(self, x):
   ...         h = self['l1'](x)
   ...         return self['l2'](h)

.. note::
   We often define a single forward method of a link by ``__call__`` operator.
   Such link is callable and behaves like a regular function of Variables.

It shows how a complex link is constructed by simpler links (like building a chain from small links).
Note that MyLink itself is a link.
It means we can define more complex links that hold MyLink objects.

Another option is the :class:`ListLink` class, which behvaes like a list of links:

.. doctest::

   >>> class MyLink2(ListLink):
   ...     def __init__(self):
   ...         super(MyLink2, self).__init__()
   ...         self.append(F.Linear(4, 3))
   ...         self.append(F.Linear(3, 2))
   ...         
   ...     def __call__(self, x):
   ...         h = self[0](x)
   ...         return self[1](h)

LinkList is convenient to define a sequence of variable number of links.
If the number of links is fixed like above case, DictLink is recommended as a base class.


Optimizer
~~~~~~~~~

In order to get good values for parameters, we have to optimize them by the :class:`Optimizer` class.
It runs a numerical optimization algorithm given a link.
Many algorithms are implemented in :mod:`optimizers` module.
Here we use the simplest one, called Stochastic Gradient Descent:

.. doctest::

   >>> model = MyLink()
   >>> optimizer = optimizers.SGD()
   >>> optimizer.setup(model)

The method :meth:`~Optimizer.setup` prepares for the optimization given a link.

In order to run optimization, you first have to compute gradients.
After computing gradient of each parameter, :meth:`~Optimizer.update` method runs one iteration of optimization:

.. doctest::

   >>> model.zerograds()
   >>> # compute gradient here...
   >>> optimizer.update()

Some parameter/gradient manipulations, e.g. weight decay and gradient clipping, can be done by setting *hook functions* to the optimizer.
Hook functions are called by the :meth:`~Optimizer.update` method before the actual update is done.
For example, we can set weight decay regularization by running the next line beforehand:

.. doctest::

   >>> # We should indicate the name of the hook by the first argument
   >>> optimizer.add_hook('WeightDecay', chainer.optimizer.WeightDecay(0.0005))


Serializer
~~~~~~~~~~

The last core feature described in this page is serializer.
Serializer is a simple interface to serialize or deserialize an object.
:class:`Link` and :class:`Optimizer` supports serialization by serializers.

Concrete serializers are defined in the :mod:`serializers` module.
Currently, it only contains a serializer and a deserializer for HDF5 format.

We can serialize a link object into HDF5 file by the :func:`serializers.save_hdf5` function:

.. doctest::

   >>> serializers.save_hdf5('my.model', model)

It saves the parameters of `model` into the file `'my.model'` in HDF5 format.
The saved model can be read by the :func:`serializers.load_hdf5` function:

.. doctest::

   >>> serializers.load_hdf5('my.model', model)

.. note::
   Note that only the parameters and the internal state arrays are serialized by these serialization code.
   Other attributes are not saved automatically.
   In order to save some arrays as same as parameters, we have to store them in the :attr:`Link.states` dictionary.

The state of an optimizer can also be saved by the same functions:

.. doctest::

   >>> serializers.save_hdf5('my.state', optimizer)
   >>> serializers.load_hdf5('my.state', optimizer)

.. note::
   Note that serialization of optimizer only saves its internal states like number of iterations, momentum vectors of MomentumSGD, etc.
   It does not save the model itself.
   We have to explicitly save the model with the optimizer to resume the optimization from saved states.


.. _mnist_mlp_example:

Example: Multi-layer Perceptron on MNIST
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you can solve a multiclass classification task using a multi-layer perceptron.
Here we use hand-written digits dataset called `MNIST <http://yann.lecun.com/exdb/mnist/>`_, which is the long-standing de-facto "hello world" of machine learning.
This MNIST example is also found in ``examples/mnist`` directory of the official repository.

In order to use MNIST, we prepared ``load_mnist_data`` function at ``examples/mnist/data.py``::

   >>> import data
   >>> mnist = data.load_mnist_data()

.. testcode::
   :hide:

   mnist = {'data': np.random.randint(255, size=(70000, 784)).astype(np.uint8),
            'target': np.random.randint(10, size=70000).astype(np.uint8)}

The mnist dataset consists of 70,000 grayscale images of size 28x28 (i.e. 784 pixels) and corresponding digit labels.
First, we scale pixels to [0, 1] values, and divide the dataset into 60,000 training samples and 10,000 test samples.

.. doctest::

   >>> x_all = mnist['data'].astype(np.float32) / 255
   >>> y_all = mnist['target'].astype(np.int32)
   >>> x_train, x_test = np.split(x_all, [60000])
   >>> y_train, y_test = np.split(y_all, [60000])

Next, we want to define the architecture.
We use a simple three-layer rectifier network with 100 units per layer as an example.

.. doctest::

   >>> class MLP(DictLink):
   ...     def __init__(self):
   ...         super(MLP, self).__init__(
   ...             l1=F.Linear(784, 100),
   ...             l2=F.Linear(100, 100),
   ...             l3=F.Linear(100, 10),
   ...         )
   ...         
   ...     def __call__(self, x):
   ...         h1 = F.relu(self['l1'](x))
   ...         h2 = F.relu(self['l2'](h1))
   ...         y = self['l3'](h2)
   ...         return y

This link uses :func:`~functions.relu` as an activation function.
Note that the ``'l3'`` link is the final linear layer whose output corresponds to scores for the ten digits.
We can also define a general classifier based on an arbitrary network:

.. doctest::

   >>> class Classifier(DictLink):
   ...     def __init__(self, predictor):
   ...         super(Classifier, self).__init__(predictor=predictor)
   ...         
   ...     def evaluate(self, x, t):
   ...         y = self['predictor'](x)
   ...         return F.softmax_cross_entropoy(y, t), F.accuracy(y, t)

:func:`~functions.softmax_cross_entropy` computes the loss value given prediction and groundtruth labels.
:func:`~functions.accuracy` computes the prediction accuracy.
We now define a *model* object and a corresponding optimizer:

.. doctest::

   >>> model = Classifier(MLP())
   >>> optimizer = optimizers.SGD()
   >>> optimizer.setup(model)

Finally, we can write a leraning loop as following:

.. testcode::
   :hide:

   datasize = 600

.. doctest::

   >>> batchsize = 100
   >>> datasize = 60000  #doctest: +SKIP
   >>> for epoch in range(20):
   ...     print('epoch %d' % epoch)
   ...     indexes = np.random.permutation(datasize)
   ...     for i in range(0, datasize, batchsize):
   ...         x_batch = x_train[indexes[i : i + batchsize]]
   ...         y_batch = y_train[indexes[i : i + batchsize]]
   ...
   ...         x = Variable(x_batch)
   ...         t = Variable(y_batch)
   ...         model.zerograds()
   ...         loss, accuracy = model.evaluate(x, t)
   ...         loss.backward()
   ...         optimizer.update()
   epoch 0...

Only the last six lines are the code related to Chainer, which are already described above.

Here you find that, at each iteration, the network is defined by forward computation, used for backprop, and then disposed.
By leveraging this "Define-by-Run" scheme, you can imagine that recurrent nets with variable length input are simply handled by just using loop over different length input for each iteration.

After or during optimization, we want to evaluate the model on the test set.
It can be achieved simply by calling forward function:

.. doctest::

   >>> sum_loss, sum_accuracy = 0, 0
   >>> for i in range(0, 10000, batchsize):
   ...     x_batch = x_test[i : i + batchsize]
   ...     y_batch = y_test[i : i + batchsize]
   ...     x = Variable(x_batch)
   ...     t = Variable(y_batch)
   ...     loss, accuracy = model.evaluate(x, t)
   ...     sum_loss += loss.data * batchsize
   ...     sum_accuracy += accuracy.data * batchsize
   ...
   >>> mean_loss     = sum_loss / 10000
   >>> mean_accuracy = sum_accuracy / 10000

The example code in the `examples/mnist` directory contains GPU support, though the essential part is same as the code in this tutorial.
We will review in later sections how to use GPU(s).
