Static Subgraph Optimizations (Static Define-By-Run)
====================================================

.. module:: chainer.graph_optimizations


.. note::
 This is an experimental feature and so the API might change in the future as it is developed.
 
Introduction
------------

Existing deep learning frameworks can roughly be classified as either a "static graph" or "dynamic graph" framework. In a static graph framework, which we also call "define-and-run", the computation graph is defined before the model is run. This implies that the same neural network model will be used each iteration without modifications, hence the name "static." This allows various graph optimizations to potentially be performed to improve the runtime performance and/or reduce memory usage. The optimized code for the computation graph is then used when the model is run.

However, in a "dynamic graph" (also called "define-by-run") framework such as Chainer, the computation 
graph is not defined before the model is run. Rather, it is constructed incrementally and automatically 
by the framework as the computations of the forward pass are executed. In Chainer, the user writes code 
to perform the computations of the forward pass in terms of Chainer functions, which have an API similar 
to an array library like NumPy. As these 
functions execute, the computation graph is incrementally built so that it will be available after the last
function in the forward pass has been called. This has some advantages, such as allowing easier
debugging compared to a static graph framework, since the user can step through the computations of the
forward pass in a debugger. Define-by-run also provides the flexibility
to include control flow operations so that a modified or even completely different graph can be
constructed each iteration. Unfortunately, this flexibility also tends to make dynamic graph frameworks
slower than static graph frameworks. For example, in Chainer there is a performance penalty involved in
dynamically constructing the graph each iteration, since it involves creating many objects; each function
call creates a new `FunctionNode` object as well as creating new `Variable` and array memory allocation
for each output of the function. There are also various dynamic type checks and graph
traversal that need to be performed, adding to the runtime overhead. Further, we cannot perform some
optimizations such as function/kernel fusion and in-place operations.

Static subgraph optimizations feature
-------------------------------------------------------------

This feature is motivated by the observation that typical deep neural networks correspond to a static computation graph and that even those that correspond to a dynamic graph are typically mostly static. By "mostly static", we mean that the largest static subgraphs each tend to contain many function nodes (that is, layers) so that the total number of function nodes in the graph tends to be much larger than the total number of largest static subgraphs. If the graph is at least mostly static, then a naive implementation of define-by-run will result in a large amount of redundant operations being performed each iteration to rebuild exactly the same subgraphs, perform the same dynamic type-checking operations, etc., which can sometimes be slow in Python; it will also result in lost opportunities to perform potential graph optimizations. A key assumption motivating this feature is that the main performance bottlenecks tend to occur inside the largest static subgraphs. So, if we can optimize these static subgraphs, it might be fine for any remaining framework code to remain implemented in pure Python. Although such Python code would be slow, it could have negligible runtime overhead.

The solution proposed by this feature is to retain the existing define-by-run style for specifying the model, but to also optionally allow the user to annotate the largest static subgraphs in a model. These "static graph" annotations will then allow the framework to automatically replace the define-by-run code of the static subgraphs with more performance-optimized code. The define-by-run code will still execute during the first iteration, to retain ease of debugging. However, as this code executes, a trace of the needed computations is also collected so that optimized static schedules can be generated for the annotated static subgraphs. Then, starting from the second iteration, this optimized code will automatically be run in place of of the original define-by-run code. Note that in the common case in which the whole model is static, the user only needs to add a single "static graph" annotation and their code will then run with the performance of a static graph framework, while still supporting the define-by-run coding style.

The benefit of annotating the static subgraphs in the model is that it allows the define-by-run code to be replaced with an optimized static schedule, which can then potentially support a user-controllable trade-off between runtime performance and memory usage. This is possible because having the full computation graph available enables various optimizations that cannot safely or automatically be performed in define-by-run. Examples (which we have not yet implemented; contributions from the open source community are welcomed) include sub-linear memory usage [1], exploiting graph parallelism, operator fusion, and in-place optimizations.

The current implementation achieves its speedup by retaining only the code that is actually needed to compute the forward pass, backward pass, and so on. This allows us to remove most of the Python interpreter overhead because the Python code that performs dynamic operations such as allocating `FunctionNode` and `Variable` objects, checking types, and traversing the backward graph is not included in the optimized static schedule code.

Usage
-----

In most cases, it should be possible to make existing code work with the static subgraph optimizations feature by making only a few minor modifications. We will now show how the Chainer MNIST example can be modified to use this feature. The modified version with static subgraph optimizations is located at `chainer.examples.static_graph_optimizations.mnist`.

The first step is to import the necessary packages:

.. literalinclude:: ../../../examples/static_graph_optimizations/mnist/train_mnist.py
   :language: python
   :lines: 14-16
   :caption: train_mnist.py

Since the neural network model `MLP` corresponds to a static graph, we can annotate it as a static graph by using the `@static_graph` decorator on the chain's ``__call__()`` method. This lets the framework know that that the define-by-run code of the chain always creates the same graph (that is, it always performs the same sequence of computations) each time it is called. We will refer to such a chain as a **static chain** in the documentation. 

.. literalinclude:: ../../../examples/static_graph_optimizations/mnist/train_mnist.py
   :language: python
   :lines: 22-50
   :caption: train_mnist.py

.. note::
   If your model's define-by-run code has any control flow operations that could cause it to potentially call different Chainer functions/links each time it is called, then you cannot use this decorator.

.. note::
   There are currently some restrictions on how variables can be passed into a static chain's ``__call__()`` method. Refer to the documentation of `@static_graph` for details.

Recall that the define-by-run code of a static chain's ``__call__()`` method only actually runs during the first iteration and is then replaced by optimized static schedule code. The current implementation only knows how to do this auto-replacement for calls to Chainer functions and links. Any other code that the user puts in ``__call__()`` will only ever get called once by default, since the define-by-run code is only executed during the first iteration. In order to make sure such "side effect" code actually gets called each iteration, we need to put it inside a function or method decorated by :meth:`static_code()` as shown in the chain above for the :meth:`example_side_effect()` method.

In this example, we only need to use `@static_graph` on the model chain, since the whole model is static. However, in more general dynamic models, each of the largest static subgraphs (which should each be written as a chain) should also use `@static_graph`.

.. note::
   Nested application of `@static_graph` is not allowed. That is, if a `@static_graph`-decorated chain calls another chains, only the outermost chain should use the decorator. This restriction might be lifted in a future version.


Understanding when define-by-run is used and when static schedules are used
---------------------------------------------------------------------------

When back propagation mode is enabled (``chainer.config.enable_backprop`` is ``True``), the define-by-run code in a static chain's ``__call__()`` method is always executed 
during the first iteration, even if the chain is called multiple times during
the iteration. An example of this is when the static chain
corresponds to one time slice of an RNN. Another example would be when the
chain corresponds to a block in a stochastic depth network that gets called
a random number of times each iteration. 
It is necessary to execute the define-by-run code and generate a separate corresponding
static schedule for each call because the intermediate results that are computed
inside the chain might be needed during the backward pass to compute gradients,
and so it would not be safe to reuse the same static schedule (which might include
statically-allocated arrays) over multiple calls during the same iteration. Otherwise,
such statically-allocated intermediate results could get overwritten, which would
result in the wrong values being used during the backward pass.

Recall that starting from the second iteration, a static chain will 
automatically switch from define-by-run to running a static schedule, if possible. It does this using the schedules that were created from the define-by-run code during the first iteration. These can be considered "cached" schedules, and there will be one cached schedule for each time the define-by-run code was executed for a particular combination of input variable type signatures.

Note that in some cases, even after the first iteration, it might not be possible to use a static schedule and the define-by-run code will need to be called. This can happen for example if the input variable types to a static chain change. It can also happen if a static chain is called more times in a single iteration than the number of cached schedules that are currently available. For example, if a static chain
is called 4 times during the first iteration and 7 times during the second iteration (such as in an RNN time slice that is unrolled a variable number of times each iteration),
then only the first 4 calls of the chain during the second iteration will use
static schedules and the remaining 3 calls will use define-by-run. Now, 7
cached static schedules will be available and so define-by-run will only
be used again if the chain is called more than 7 times during the same iteration. Similarly, if the mini-batch size changes at any point, the define-by-run code will
need to be executed again if there are no available cached schedules. 

Note that since a distinct static schedule must be used for each call of a static chain during the same iteration, it is therefore necessary that a static chain somehow be informed when a given iteration has completed so that it will know when it is safe to start reusing its cached schedules again. 
When training mode is active, we currently use the first call of ``backward()`` on the 
computation graph to signal to the static chain that 
the forward pass has completed. The user may also manually inform the chain that
the iteration has completed by calling ``end_forward()`` on the chain's schedule
manager. A static chain will have a ``schedule_manager`` attribute once it has
been called. So, if the chain is called ``my_chain``, you can call
``my_chain.schedule_manager.end_forward()`` to let the chain know that the iteration
has completed. Since we make no assumptions regarding the number of times that
a static chain may be called in a given iteration, we cannot assume that the
static chain will continue to be called the same number of times after the 
first iteration. It is therefore necessary to allays call ``end_forward()`` at
the end of the iteration if you do not wish to call ``backward()``.

When evaluation mode is 
active or if backrop mode is off, it is not necessary to retain separate internal
arrays, since backpropagation will not be used. Thus, in this case the chain
can simply begin using a static schedule the second time it is called even
when it is called multiple times in the same iteration. That is, the same cached schedule may be called multiple times in a given iteration.

Effects on model debugging
--------------------------

Note that since the code in the static chain's ``__call__()`` only runs during the first iteration, you will only be able to debug this code as define-by-run during the first iteration. It is assumed that if the chain is actually is static, any problems in its define-by-run code should be apparent during the first iteration and it should not be (as) necessary to debug this code in later iterations. However, this feature does provide some functionality to help with debugging. For example, it is possible to obtain and inspect the current static schedules. It is also possible to directly step through the code of the static schedule if you wish (by debugging the ``forward()`` method of :class:`StaticScheduleFunction` in :mod:`~chainer.static_graph`).


Limitations and future work
---------------------------

- Optimization switches to let the user select the trade-off between runtime performance and memory usage: The current implementation achieves its speedups mainly by reducing the amount of Python code that needs to run, but does not yet implement advanced optimizations for memory usage or runtime performance. Ideally, the user should be able to adjust performance tuning parameters to control the trade-off between memory consumption and runtime performance.

- Incompatibility with GRU and LSTM links: This feature requires that all input variables to a chain need to explicitly appear in the arguments to the chain's ``__call__()`` method. However, the GRU and LSTM links with state maintain variable attributes of the chain for the RNN state variables. Design changes to support such links and/or modifications to these links are being considered. These links may still be used with the current implementation, as long as the corresponding RNN is unrolled inside of a static chain. For an example of this, see the modified ptb example at `chainer.examples.static_graph_optimizations.ptb`

- Memory usage: The current implementation caches all static schedules which can lead to high memory usage in some cases. For example, separate schedules are created when the training mode or mini-batch size changes. 

- Advanced graph optimizations: Advanced optimizations such as fusion of operations is not yet implemented.

- Constraints on arguments to a static chain: The current version requires that all input variables used inside `__call__()` of a static chain must either appear in the arguments of this method or be defined in the define-by-run code. Furthermore, any variables that appear in the arguments list must appear by themselves or be contained inside a list or tuple. Arbitrary levels of nesting are allowed.

- Model export: In the case where the complete computation graph for the model is static, it should be possible in principle to export the static schedule in a format that can be run on other platforms and languages. One of the other original motivations for this feature was to support exporting static Chainer models to run on C/C++ and/or optimize the static schedule execution code in Cython/C/C++. However, it seems that ONNX is now fulfilling this purpose and there is a separate ONNX exporter already in development for Chainer. Perhaps these two features can be merged at some point in the future.

- Double-backward support: This feature was designed to support double-backward (gradient of gradient) but it has not been tested.

Examples
--------

For additional examples that use this feature, refer to the examples in `chainer.examples.static_graph_optimizations`.

Adding support to existing functions
----------------------------------------

Most functions and links will not need to be modified at all in order to support this feature, since the framework code will attempt to auto-wrap them inside a `@static_code`-decorated function. However, some functions might see a performance benefit if static graph support is added manually, since it may result in less redundant code being included in the static schedule. For example, any dynamic checking code that will return the same result every iteration does not need to be included in the static schedule. 

An existing function (that is, a subclass of `FunctionNode`) can be modified to support static graph optimizations as follows. The basic idea is to wrap any code that needs to be called each iteration inside a method that is decorated with ``@static_code``. Note that code that should only run once, such as initializing parameters, should not be wrapped.

It is also necessary to set the ``_supports_static_optimizations = True`` class attribute. Note that this attribute is ``False`` by default in ``FunctionNode``.


Since the function is part of a static graph, any parameters and output arrays should ideally be statically allocated during the first iteration (while the define-by-run code is executing) and then reused starting from the second iteration. The ``@static_code``-decorated functions that are called each iteration will perform the various deep learning computations, writing results in-place into these static arrays. Since the results are written in-place, there is no need for an `@static_code`-decorated function to explicitly return a result. Rather, any results arrays should be passed as inputs along with any other input arguments to the function. However, it also is allowed to return dynamically allocated arrays so that existing Chainer functions can be easily supported.
The following code shows the typical pattern for performing the forward computations in a `FunctionNode`::

    @static_code
    def static_forward(self, inputs, outputs):
        # This function will get
 included in the static
        # schedule and called each iteration.
        # Any input arrays must be passed in a list
        # to the `inputs` keyword argument.
        x = inputs[0]
        # Any output arrays must be passed in a list
        # to the `outputs` keyword argument, and must
        # have already been initialized to the required
        # shape. Results are written in-place into output
        # arrays.
        y = outputs[0]

        # Read from x, write results into y in-place.
        # Don't forget to zero y if necessary.
        y *= 0.0 # (if necessary)
        y[:] = 3.0*x # for example

    def forward(self, inputs):
        # Initialization/type checking code.
        # (only gets called once, during first iteration)
        type_check_blah(inputs)

        # Allocate output array. Note that since this line
        # is not wrapped using @static_code, it
        # will only ever get called once, during the first
        # iteration.
        y = xp.empty(y_shape).astype(x.dtype)

        # Call static function
        # (it will get called every iteration from optimized schedule)
        self.static_forward(inputs=[x], outputs=[y])
        return y,



It should not be necessary to modify the `backward()` implementation. As of Chainer v3 when double-backward (i.e., grad of grad) support was added, the ``backward()`` method of :class:`FunctionNode` actually calls the `forward()` method of other `FunctionNode`s, and so it is only necessary that the `forward()` functions be wrapped.

For an example of how to add support to an existing function, see the ``Linear`` function.

Adding support to existing links
------------------------------------

Most existing links will work as-is and do not need to be modified. However, if a link needs to perform computations each iteration that are performed in code other than calling chainer functions, this code will need to be manually placed in a `@static_code`-decorated function or method of the link.

If a link performs different computations depending on the training mode but is otherwise static, then it does not need to be modified.

Reference
---------

[1] `Training deep nets with sublinear memory cost <https://arxiv.org/abs/1604.06174>`_

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.graph_optimizations.static_graph.static_graph

