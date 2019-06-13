Static Subgraph Optimizations: Usage
====================================

.. note::
 This is an experimental feature and so the API might change in the future as it is developed.

This feature intends to improve runtime performance by optimizing the execution
of the static subgraphs in a model. When this feature is enabled, the first iteration
runs as normal except that an execution trace is also collected. The
trace is then used to generate optimized code that is will be called instead of the define-by-run
code starting from the second iteration.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.static_graph

Basic usage
-----------

To enable static graph optimizations, it is only necessary to add the
:func:`chainer.static_graph` decorator to a chain's ``__call__()`` method. We will now show how the
Chainer MNIST example can be modified to use this feature. The modified version
with static subgraph optimizations is located at :tree:`examples/static_graph_optimizations/mnist`.

The first step is to import the necessary packages:

.. literalinclude:: ../../../examples/static_graph_optimizations/mnist/train_mnist.py
   :language: python
   :lines: 15-16
   :caption: train_mnist.py

Since the neural network model ``MLP`` corresponds to a static graph, we can annotate it as a static graph by
using the :func:`chainer.static_graph` decorator on the chain's ``__call__()`` method. This lets the framework know that
that the define-by-run code of the chain always creates the same graph (that is, it always performs the same
sequence of computations) each time it is called. We will refer to such a chain as a **static chain** in
the documentation.

.. literalinclude:: ../../../examples/static_graph_optimizations/mnist/train_mnist.py
   :language: python
   :lines: 22-41
   :caption: train_mnist.py

.. note::
   If your model's define-by-run code has any control flow operations that could cause it to potentially call
   different Chainer functions/links each time it is called, then you cannot use this decorator.

.. note::
   There are currently some restrictions on how variables can be passed into a static chain's ``__call__()``
   method. Refer to the documentation of :func:`chainer.static_graph` for details.

Recall that the define-by-run code of a static chain's ``__call__()`` method only actually runs during the
first iteration and is then replaced by optimized static schedule code. The current implementation only
knows how to do this auto-replacement for calls to Chainer functions and links. Any other code that the
user puts in ``__call__()`` (which we refer to as "side-effect code") will only ever get called once
by default, since the define-by-run code is
only executed during the first iteration. In order to make sure such "side effect" code actually gets
called each iteration, we need to put it inside a function or method decorated by :meth:`static_code()`.
We expect there will rarely be a need to use side-effect code but for completeness, an example of
a model that uses it is available in the ``MLPSideEffect`` Chain of the static graph MNIST example.

In this example, we only need to use :func:`chainer.static_graph` on the model chain, since the whole model is static.
However, in more general dynamic models, each of the largest static subgraphs (which should each be
written as a chain) should also use :func:`chainer.static_graph`.

.. note::
   Nested application of :func:`chainer.static_graph` is not allowed. That is, if a :func:`chainer.static_graph`-decorated chain
   calls another chains, only the outermost chain should use the decorator.


Calling a static chain multiple times in the same iteration
-----------------------------------------------------------

In a general dynamic graph network, it is not possible
to know in advance how many times a static chain will be called in any particular iteration.
Note that during training, it is necessary to maintain separate internal state (such as intermediate
activations) for each of these calls so that the gradients can be computed in the backward pass.
So, although the layer functions of the static schedule will be identical each time the same
static chain is called, any internal state must be distinct. It is also possible that a static
chain could be called multiple times with inputs of different shapes and/or types during the same iteration.
To avoid confuction, "static schedule" will refer to both the functions and any corresponding internal state
such as activations.

If backpropagation mode is disabled (``chainer.config.enable_backprop`` is ``False``),
it is safe for the implementation to simply compute a
static schedule for the first call and reuse it for subsequent calls, provided that the cached schedule
is compatible with the input shapes/types. However, during training,
it is necessary to maintain distinct internal state for each call in order to compute
the gradients for the backward pass, which prevents us from reusing the same static schedule for each of
the multiple calls of a static chain in an iteration.

The current implementation handles this issues as follows. A cache of static schedules, which is intially
empty, is associated with each static chain. The size of this cache will be equal to the maximum number of
times that the static chain has been called in any previous iteration, and the cache is reset whenever
certain chain configuration flags change, such as training mode and backpropagation model. At the start
of a given iteration, all cached schedules are available for use and the number of available schedules
is decremented each time the static chain is called. If the chain is called when the cache is size zero,
then its define-by-run code will execute to create a new schedule cache.

In order for such an implementation to work, each static chain must be notified when the forward pass
has ended (or when the forward pass is started) so that all cached schedules can be made available for use
again. In the current implementation, this is accomplished by calling the ``backward()`` method on a loss
variable in the model. This is expected to handle the typical use cases. However, in some models it may be necessary to
perform multiple forward passes before calling ``backward()``. In such a case, to signel to a static chain that the
forward pass (and the iteration) has ended, call ``my_chain.schedule_manager.end_forward()``.
The `schedule_manager` attribute of a static chain is an instance of a class called
``StaticScheduleFunction`` that will be available after the chain has been called.


Effects on model debugging
--------------------------

Note that since the code in the static chain's ``__call__()`` only runs during the
first iteration, you will only be able to debug this code as define-by-run during
the first iteration. It is assumed that if the chain is actually is static,
any problems in its define-by-run code should be apparent during the first
iteration and it should not be (as) necessary to debug this code in later iterations.
However, this feature does provide some functionality to help with debugging.
For example, it is possible to obtain and inspect the current static schedules.
It is also possible to directly step through the code of the static schedule if
you wish (by debugging the ``forward()`` method of :class:`~chainer.static_graph.StaticScheduleFunction`
in :mod:`~chainer.static_graph`).


Limitations and future work
---------------------------

- Optimization switches to let the user select the trade-off between runtime performance and memory usage: The current implementation achieves its speedups mainly by reducing the amount of Python code that needs to run, but does not yet implement advanced optimizations for memory usage or runtime performance. Ideally, the user should be able to adjust performance tuning parameters to control the trade-off between memory consumption and runtime performance.

- Incompatibility with GRU and LSTM links: This feature requires that all input variables to a chain need to explicitly appear in the arguments to the chain's ``__call__()`` method. However, the GRU and LSTM links with state maintain variable attributes of the chain for the RNN state variables. Design changes to support such links and/or modifications to these links are being considered. These links may still be used with the current implementation, as long as the corresponding RNN is unrolled inside of a static chain. For an example of this, see the modified ptb example at :tree:`examples/static_graph_optimizations/ptb`

- Memory usage: The current implementation caches all static schedules which can lead to high memory usage in some cases. For example, separate schedules are created when the training mode or mini-batch size changes.

- Advanced graph optimizations: Advanced optimizations such as fusion of operations is not yet implemented.

- Constraints on arguments to a static chain: The current version requires that all input variables used inside ``__call__()`` of a static chain must either appear in the arguments of this method or be defined in the define-by-run code. Furthermore, any variables that appear in the arguments list must appear by themselves or be contained inside a list or tuple. Arbitrary levels of nesting are allowed.

- Model export: In the case where the complete computation graph for the model is static, it should be possible in principle to export the static schedule in a format that can be run on other platforms and languages. One of the other original motivations for this feature was to support exporting static Chainer models to run on C/C++ and/or optimize the static schedule execution code in Cython/C/C++. However, it seems that ONNX is now fulfilling this purpose and there is a separate ONNX exporter already in development for Chainer. Perhaps these two features can be merged at some point in the future.

- Double-backward support: This feature was designed to support double-backward (gradient of gradient) but it has not been tested.

Examples
--------

For additional examples that use this feature, refer to the examples in :tree:`examples/static_graph_optimizations`.
