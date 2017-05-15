How to write a training loop in Chainer
```````````````````````````````````````

.. currentmodule:: chainer

In this tutorial section we will learn how to train a deep neural network to
classify hand-written digits using the popular MNIST dataset. This dataset
contains 50000 training examples and 10000 test examples. Each example contains
a 28x28 greyscale image and a corresponding class label for the digit. Since
the digits 0-9 are used, there are 10 class labels.

Chainer provides a feature called :class:`~chainer.training.Trainer` that can
be used to simplify the training process. However, we think it is good for
first-time users to understand how the training process works before using the
:class:`~chainer.training.Trainer` feature. Even advanced users might sometimes
want to write their own training loop and so we will explain how to do so here.

The complete training process consists of the following steps:

1. Prepare a datasets that contain the train/validation/test examples.
2. Optionally set of iterators for the datasets.
3. Write a training loop that performs the following operations in each iteration:
    A. Retreive batches of examples from the training dataset.
    B. Feed the batches into the model.
    C. Run the forward pass on the model to compute the loss.
    D. Run the backward pass on the model to compute the gradients.
    E. Run the optimizer on the model to update the parameters.
    F. (Optional): Ocassionally check the performance on a validation/test set.

1. Prepare the dataset
''''''''''''''''''''''

Chainer contains some built-in functions that can be used to download and
return Chainer-formated versions of popular datasets used by the ML and deep
learning communities. In this example, we will use the buil-in function that
retreives the MNIST dataset.

.. testcode::

    from chainer.datasets import mnist

    # Download the MNIST data if you haven't downloaded it yet
    train, test = mnist.get_mnist(withlabel=True, ndim=1)

    # set matplotlib so that we can see our drawing inside this notebook
    %matplotlib inline
    import matplotlib.pyplot as plt

    # Display an example from the MNIST dataset.
    # `x` contains the input image array and `t` contains that target class
    # label as an integer.
    x, t = train[0]
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.show()
    print('label:', t)

Output
......

.. image:: ../../image/train_loop/5.png

label: 5

2. Create the datset iterators
''''''''''''''''''''''''''''''

Although this is an optional step, it can often be convinient to use iterators
that operate on a dataset and return a certain number of examples (often called
a "mini-batch") at a time. The number of examples that is returned at a time is
called the "batch size" or "mini-batch size." Chainer already has an Iterator
class and some subclasses that can be used for this purpose and it is
straightforward for users to write their own as well.

We will use the :class:`~chainer.iterators.SerialIterator` subclass of
:class:`~chainer.dataset.Iterator` in this example. The
:class:`~chainer.iterators.SerialIterator` can either return the examples in
the same order that they appear in the dataset (that is, in sequetial order)
or can shuffle the examples so that they are returned in a random order.

An :class:`~chainer.dataset.Iterator` can return a new minibutch by calling its
:meth:`~chainer.dataset.Iterator.next` method. An
:class:`~chainer.dataset.Iterator` also has properties to manage the training
such as :attr:`~chainer.dataset.Iterator.epoch`: how many times we have gone
through the entire dataset, :attr:`~chainer.dataset.Iterator.is_new_epoch`:
whether the current iteration is the first iteration of a new epoch.

.. testcode::

    # Choose the minibatch size.
    batchsize = 128

    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize,
                                         repeat=False, shuffle=False)

Details about SerialIterator
............................

- :class:`~chainer.iterators.SerialIterator` is a built-in subclass of :class:`~chainer.dataset.Iterator` that can be used to retreive a dataset in either sequential or shuffled order.
- The :class:`~chainer.dataset.Iterator` initializer takes two arguments: the dataset object and a batch size.
- When data need to be used repeatedly for training, set the ``repeat`` argument to ``True`` (the default). When data is needed only once and no longer necessary for retriving the data anymore, set ``repeat`` to ``False``.
- When you want to shuffle the training dataset for every epoch, set the ``shuffle`` argument ``True``.

In the example above, we set ``batchsize = 128``, ``train_iter`` is the
:class:`~chainer.dataset.Iterator` for the training dataset, and ``test_iter``
is the :class:`~chainer.dataset.Iterator` for test dataset. These iterators
will therefore return 128 image examples as a bundle.

3. Define the model
'''''''''''''''''''

Now let's define a neural network that we will train to classify the MNIST
images. For simplicity, we will use a fully-connected network with three
layers. We will set each hidden layer to have 100 units and set the output
layer to have 10 units, corresponding to the 10 class labels for the MNIST
digits 0-9.

We first briefly explain :class:`~chainer.Link`, :class:`~chainer.Function`,
:class:`~chainer.Chain`, and :class:`~chainer.Variable` which are the basic
components used for defining and running a model in Chainer.

Link and Function
.....................................................

In Chainer, each layer of a neural network is decomposed into one of two broad
types of functions (actually, they are function objects):
:class:`~chainer.Link` and :class:`~chainer.Function`.

- **:class:`~chainer.Function` is a function without learnable paremeters.**
- **:class:`~chainer.Link` is a function that conatins (learnable) parameters.** We can think of :class:`~chainer.Link` as wrapping a :class:`~chainer.Function` to give it parameters. That is, :class:`~chainer.Link` will contain the parameters and when it is called, it will also call a corresponding :class:`~chainer.Function`.

We then describe a model by implementing code that performs the "forward pass"
computations. This code will call various links and chains (recall that
:class:`~chainer.Link` and :class:`~chainer.Chain` are callable objects).
Chainer will take care of the "backward pass" automatically and so we do not
need to worry about that unless we want to write some custom funcions.

- For examples of links, see the :mod:`chainer.links` module.
- For examples of functions, see the :mod:`chainer.functions` module.
- For example, see the :class:`~chainer.links.Linear` link, which wraps the :class:`~chainer.functions.linear` function to give it weight and bias parameters.
- Before we can start using them, we first need to import the modules as shown below.

.. testcode::

    import chainer.links as L
    import chainer.functions as F

The Chainer convention is to use ``L`` for links and ``F`` for functions, like
``L.Convolution2D(...)`` or ``F.relu(...)``.

Chain
.....

- :class:`~chainer.Chain` is a class that can hold multiple links and/or functions. It is a subclass of :class:`~chainer.Link` and so it is also a :class:`~chainer.Link`.
- This means that a :class:`~chainer.Chain` can contain parameters, which are the parameters of any links that it deeply contains.
- In this way, :class:`~chainer.Chain` allows us to construct models with a potentially deep hierarchy of functions and links.
- It is often convinient to use a single :class:`~chainer.Chain` that contains all of the layers (other chains, links, and functions) of the model. This is because we will need to optimize the model's parameters during training and if all of the parameters are contained by a single :class:`~chainer.Chain`, it turns out to be straightfoward to pass these parameters into an optimizer (which we describe in more detail below).

Variable
........

In Chainer, both the activations (that is, the inputs and outputs of function
and links) and the model parmaeters are instances of the
:class:`~chainer.Variable` class. A :class:`~chainer.Variable` holds two
arrays: a data array that contains the values that are read/written during the
forward pass (or the parameter values), and a :attr:`~chainer.Variable.grad`
array that contains the corresponding gradients that will are computed during
the backward pass.

A :class:`~chainer.Variable` can potentially contain two types of arrays as
well, depending whether the array resides in CPU or GPU memory. By default,
the CPU is used and these will be NumPy arrays. However, it is possible to move
or create these arrays on the GPU as well, in which case they will be CuPy
arrays. Fortunately, CuPy uses an API that is nearly identical to NumPy. This
is convinient because in addition to making it easier for users to learn (there
is almost nothing to learn if you are already familiar with NumPy), it often
allows us to reuse the same code for both NumPy and CuPy arrays.

Create our model as a subclass of Chain
.......................................

We can create our model be writing a new subclass of :class:`~chainer.Chain`.
The two main steps are:

1. Any :class:`~chainer.Link` objects (possibly also including other :class:`chainer.Chain` objects) that we wish to call during the forward computation of our :class:`~chainer.Chain` must first be supplied to the :class:`~chainer.Chain`'s :meth:`~chainer.Chain.__init__` method. After the :meth:`~chainer.Chain.__init__` method has been called, these :class:`~chainer.Link` objects will then be accessable as attributes of our :class:`~chainer.Chain` object. This means that we also need to provide the attribute name that we want to use for each :class:`~chainer.Link` object that is supplied. We do this by providing the attribute name and corresponding :class:`~chainer.Link` object as keyword arguments to :meth:`~chainer.Chain.__init__`, as we will do in the MLP chain below.
2. We need to define a :meth:`~chainer.Chain.__call__` method that allows our :class:`~chainer.Chain` to be called like a function. This method takes one or more :class:`~chainer.Variable` objects as input (that is, the input activations) and returns one or more :class:`~chainer.Variable` objects. This method executes the forward pass of the model by calling any of the links that we supplied to :meth:`~chainer.Chain.__init__` earlier as well as any functions.

Note that only the :class:`~chainer.Link` objects need to be supplied to
:meth:`~chainer.Chain.__init__`. This is because they contain parameters. Since
:class:`~chainer.Function`s do not contain any parameters, they can be called
in :meth:`~chainer.Chain.__call__` without having to supply them to the
:class:`~chainer.Chain` beforehand. For example, we can use a
:class:`~chainer.Function` such as :meth:`~chainer.functions.relu` by simply
calling it in :meth:`~chainer.Chain.__call__` but a :class:`~chainer.Link` such
as :class:`~chainer.links.Linear` would need to first be supplied to the
:class:`~chainer.Chain`'s :meth:`~chainer.Chain.__init__` in order to call it
in :meth:`~chainer.Chain.__call__`.

If we decide that we want to call a :class:`~chainer.Link` in a
:class:`~chainer.Chain` after :meth:`~chainer.Chain.__init__` has already been
called, we can use the :meth:`~chainer.Chain.add_link` method of
:class:`~chainer.Chain` to add a new :class:`~chainer.Link` object at any time.

In Chainer, the Python code that implements the forward computation code itself
represents the model. In other words, we can conceptually think of the
computation graph for our model being constructed dynamically as this forward
computation code executes. This allows Chainer to describe networks in which
different computations can be performed in each iteration, such as branched
networks, intuitively and with a high degree of flexibiity. This is the key
feature of Chainer that we call **Define-by-Run**.

How to run a model on GPU
.........................

- The :class:`~chainer.Link` and :class:`~chainer.Chain` classes have a :meth:`~chainer.Chain.to_gpu` method that takes a GPU id argument specifying which GPU to use. This method sends all of the model paremeters to GPU memory.
- By default, the CPU is used.

.. testcode::

    class MLP(chainer.Chain):

        def __init__(self, n_mid_units=100, n_out=10):
            # register layers with parameters by super initializer
            super(MLP, self).__init__(
                l1=L.Linear(None, n_mid_units),
                l2=L.Linear(None, n_mid_units),
                l3=L.Linear(None, n_out),
            )

        def __call__(self, x):
            # describe the forward pass, given x (input data)
            h1 = F.relu(self.l1(x))
            h2 = F.relu(self.l2(h1))
            return self.l3(h2)

    gpu_id = 0

    model = MLP()
    model.to_gpu(gpu_id)

NOTE
____

The :class:`~chainer.links.Linear` class is a :class:`~chainer.Link` that
represents a fully connected layer. When ``None`` is passed as the first
argument, this allows the number of necessary input units (``n_input``) and
also the size of the weight parameter to be automatically determined and
computed at runtime during the first forward pass. We call this feature
parameter **shape placeholder**. This can be a very helpful feature when
defining deep neural network models, since it would often be tedious to
manually determine these input sizes.

As mentioned previously, a :class:`~chainer.Link` can contain multiple
parameter arrays. For example, the :class:`~chaienr.links.Linear` link conatins
two parameter arrays: the weigts :attr:`~chainer.links.Linear.W` and bias
:attr:`~chainer.links.Linear.b`. Recall that for a given link or chain, such as
the MLP chain above, the links it contains can be accessed as attributes. The
paramaters of a link can also be accessed as attributes. For example, following
code shows how to access the bias parameter of layer ``l1``:

.. testcode::

    print('The shape of the bias of the first layer, l1, in the model:', model.l1.b.shape)
    print('The values of the bias of the first layer in the model after initialization:\n', model.l1.b.data)

Output
^^^^^^

.. code-block::

    The shape of the bias of the first layer, l1, in the model: (100,)
    The values of the bias of the first layer in the model after initialization:
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

4. Select an optimization algorithm
'''''''''''''''''''''''''''''''''''

Chainer provides a wide variety of optimization algorithms that can be used to
optimize the model parameters during training. They are located in the
:mod:`~chainear.optimizers` module.

Here, we are going to use the basic stochastic gradient descent (SGD) method,
which is implemented by :class:`~chainer.optimizers.SGD`. The model (recall
that it is a :class:`~chainer.Chain` object) we created is passed to the
optimizer object by providing the model as an argument to the optimizer's
:meth:`~chainer.Optimizer.setup` method. In this way, the
:class:`~chainer.Optimizer` can automatically find the model paremeters to be
optimized.

You can easily try out other optimizers as well. Please test and observe the
results of various optimizers. For example, you could try to change ``SGD`` of
:class:`chainer.optimizers.SGD` to :class:`~chainer.optimizers.MomentumSGD`,
:class:`~chainer.optimizers.RMSprop`, :class:`~chainer.optimizers.Adam`, etc.
and run your training loop.

.. testcode::

    # Choose an optimizer algorithm
    optimizer = optimizers.SGD(lr=0.01)
    # Give the optimizer a reference to the model so that it
    # can locate the model's parameters.
    optimizer.setup(model)
