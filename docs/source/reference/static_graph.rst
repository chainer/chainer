Static Graph Optimizations
====================================

.. module:: chainer.graph_optimizations

This is an experimental feature that can be used to improve the runtime performance of a model by performing static sub-graph optimizations.

Background
----------

Existing deep learning frameworks can basically be classified as either a "static graph" or "dynamic graph" framework. In a static graph framework (which we also call "define-and-run"), the full computation graph must first be defined before the model can be run. Once the graph is defined, varios optimizations such as fusion of kernels/operations may also be performed. Finally, the optimized graph is then repeatedly called inside a training and/or evaluation loop.

On the other hand, in a dynamic graph framework like Chainer, the full computation graph is not defined beforehand. That is, when the training (or evaluation) loop is started and batches of data are fed into the model, there is no computation graph available yet! What happens is that when a batch of data is fed into the model, some code is executed that simply specifies the forward computations to perform. The framework will then build the computation graph incrementally as it executes this code. For example, in Chainer, such code for the forward computations is written in Python and looks quite similar to the style of code that would be written to perform the same computations in an array library like Numpy. Basically any Python code is allowed, as long as the actual forward computations are expressed using Chainer functions. In Chainer, control flow operations such as conditional statements and loops are also allowed in the define-by-run code. Since this code is executed each iteration to incrementally build the computation graph from scratch, this also means that a potentially different graph can be constructed each time.

Since graph optmizations such as fusion are not performed, this can make models easier to debug. For example, the user can step through the define-by-run code in a debugger, set breakpoints to stop at a certain layer to check values etc.. Users may also find it more natural to simply the write code that implements the forward computations, rather than having to define a computation graph beforehand.

However, a drawback of dynamic frameworks is that the runtime performance is typically slower than a static graph framework. This is due to the overhead of building the computatin graph from scratch each iteration, performing various checks to make debugging easier (such as dynamic type checking), and not being able to use graph optimizations such as operator fusion. For example, executing the define-by-run code in Chainer involves creating many objects (each function call creates a new `FunctionNode` object as well as creating new `Variable` and array memory allocation for each output of the function). In addition to this, there is also code to perform graph traversal for the backward pass.

Static sub-graph optimizations (Static define by run) feature
-------------------------------------------------------------

The motivation for this feature is to maintain the benefits of define-by-run while improving the runtime performance to make it more competitive with static graph frameworks. The basic idea is simple: Perform the first iteration using define-by-run so that ease of debugging is maintained. However, also during the first iteration, trace the execution of each function and incrementally construct optimized static schedules for the largest static sub-graphs in the model. Then, starting from the second iteration, Chainer will automatically switch over to using the optimized schedules in place of the corresponding define-by-run code when possible. Although it might be possible to make this completely automated, the current version requires the user to annotate each of the largest subgraphs (which correspond to a Chain) using the `@static_graph` decorator.

This is motivated by the assumption that the performance bottlenecks in typical deep learning models occur mainly inside these largest static subgraphs. If we can find a way to improve the runtime performance of these static subgrpahs, perhaps it could allow Chainer to become competitive with static graph frameworks.

The current version of this feature constructs and uses static schedules as described above, but there are some inefficiencies in the code and many potential optimizations that are not yet implemented (see the limitations section for details). We expect the performance of this feature to improve as it continues to be developed.

Usage
-----

If the model corresponds to a static graph, simply apply the `@static_graph` decorator to the chain containing the model. If only a sub-part of the model corresponds to a static graph, apply the decorator to the chains that contain the largest static sub-graphs in the model. We will also refer to a chain that uses `@static_graph` as a "static chain" in the documentation. Nested application of `@static_graph` is not allowed. That is, if a `@static_graph`-decorated chain calls another chains, only the outermost chain should use the decorator.


Understanding when define-by-run is used:

Note that the define-by-run code in a static chain's `__call__()` is executed during the first iteration and then starting from the second iteration, a static schedule is used instead. Thus, a static chain needs to know when the first iteration (or at least the first forward pass) has completed so that it can then start using the corresponding static schedule. You might be thinking that we could just switch over to the static schedule the second time the chain is called. However, that would not work during training mode if the same chain happens to be called multiple times in the forward pass. For example, consider the case where the chain corresponds to an RNN the is called several times as it is unrolled over multiple time slices or a chain that corresponds to a block of layers that is called several times such as a block in a ResNet. Yet another example would be a block in a stochastic deptch network that gets called a random number of times each iteration.

When training mode is active, we currently use the first call of `backward()` to signal to the static chain that the forward pass has completed. This might change in the future and we might also add a way for the user to manually inform the chain that the iteration has started/completed. When evaluation mode is active or if backrop mode is off, the chain switches to a static schedule the second time it is called. Note that when training mode is on, it is therefore important to call backward each iteration. Otherwise, the chain would stay in define-by-run mode while it incrementally builds a longer and longer static schedule (that is never actually used), and would eventually run out of memory.


Side effects
------------

It is important to be careful that there is no code containing side effects inside a static chain's `__call__()`. This is because the chains' define-by-run code normally only runs once (or only when/if it needs to generate a new schedule). Any code with side effects would therefore also only every run once, or at most, run infrequently. It actually is possible to include code with side effects, but it most be explicitly marked as such using `chainer.static_graph_utilities.static_schedule_func()`. See the static graph examples that the implementation of the `Linear` function for usage examples.

Effects on model debugging
--------------------

Note that since the code in the static chain's `__call__()` only runs during the first iteration, you will only be able to debug this code during the first iteration. It is assumed that if the chain is actually is static, any problems in its define-by-run code should be apparant during the first iteration and it should not be (as) necessary to debug this code in later iterations. However, this feature does provide some functionality to help with debugging. For example, it is possible to obtain and inspect the current static schedules. It is also possible to directly step through the code of the static schedule if you wish (by debugging the `forward()` method of `StaticScheduleFunction` in `chainer.static_graph`).


Limitations and future work
---------------------------

Optimization switches to let the user select the tradeoff between runtime performance and reduced memory usage:
The current version is neither well-optimized for runtime performance nor memory efficienct, but currently acheives its speedups mainly by reducing the amount of Python code that needs to run. Ideas for future improvments include a "static allocation" mode that statically allocates all intermediate arrays and writes the results in-place into these arrays each iteration to reduce memory allocation overhead. At the other extreme would be a "dynamic allocation" mode that dynamically allocates intermediate arrays like in existing define-by-run and then deletes them once they are no longer needed. Ideally, in-place operations should be used as much as possible.

Incompatibility with GRU and LSTM links:
This feature requires that all input variables to a chain need to explicitly appear in the arguments to the chain's `__call__()` method. However, the GRU and LSTM links with state maintain variable attributes of the chain for the RNN state variables. Design changes to support such links and/or modifications to these links are being considered. These links may still be used with this feature, as long as the corresponding RNN is unrolled inside of a static chain.

Unecessary copy operations:
The current version of this feature makes redundant copies of intermediate arrays in many cases, in order to easily support existing Chainer functions with minimal code changes. These copies might actually result in a slight performance decrease for some models in which the GPU was already the performance bottleneck. A fix for this is currently in development.

Memory usage:
Existing Chainer define-by-run code deletes intermediate arrays once they are no longer needed in the forward and backward passes. However, in the current version of this feature, all intermediate arrays are statically allocated, which can result in significantly higher memory usage in some models. Such optimizations are currently in development and expected to be available soon.

Advanced graph optimizations:
Advanced optimizations such as fusion of operations is not yet implemented.

Constraints on arguments to a static chain:
The current version requires that all input variables used insde `__call__()` of a static
chain must either appear in the arguments of this method or be defined in the define-by-run
code. Furthermore, any variables that appear in the arguments list must appear by
themeselves or be contained inside a list or tuple. Arbitrary levels of nesting are
allowed.

Model export:
In the case where the complete computation graph for the model is static, it should be possible in principle to export the static schedule in a format that can be run on other platforms and languages. One of the other original motivations for this feature was to support exporting static Chainer models to run on C/C++ and/or optimize the static schedule execution code in Cython/C/C++. However, it seems that ONNX is now fullfilling this purpose and there is a separate ONNX exporter already in development for Chainer. Perhaps these two features can be merged at some point in the future.

Double-backward support: This feature was designed to support double-backward but it has not been tested because there are no examples yet that suport this feature.

Examples
--------

For example usage of this feature, refer to the examples in `chainer.static_graph_optimizations`.

How to add support to existing functions
----------------------------------------

Most functions and links will not need to be modified at all in order to support this feature. However, some functions might see a performance benifit if static graph support is added manually, since it may result in less redundant code being included in the static schedule. For example, any dynamic checking code that will return the same result every iteration does not need to be included in the static schedule. 

An existing function (that is, a subclass of `FunctionNode`) can be manually modified to support static graph optimizations as follows. The basic idea is to wrap any code that needs to be called each iteration inside a method that is decorated with `@static_schedule_func`. Therefore, code that performs initialization such as initializing parameters does not need to (and should not) be wrapped.

Since the function is part of a static graph, any parameters and output arrays should ideally be statically allocated only once during the first iteration (while the define-by-run code is executing) and then reused starting from the second iteration. The `@static_schedule_func`-decorated functions that are called each iteration will perform the various deep learning computations, writing results in-place into these static arrays. Since the results are written in-place, there is no need for an `@static_schedule_func`-decorated function to explicitly return a result and so we disallow it (although it might be allowed in the future to reduce memory usage). Rather, any results arrays should be passed as inputs along with any other input arguments to the function. 
The following code shows the typical pattern for performing the forward computations in a `FunctionNode`::

    @static_schedule_func
    def static_forward(self, x, y):
        # This function will get included in the static
        # schedule and called each iteration.
        # This function must not return a result!
        # Any output arrays (such as y) must be
        # passed as an input argument.

        # Read from x, write results into y in-place.
        # Don't forget to zero y if necessary.
        # y *= 0.0 # (if necessary)
        y[:] = 3.0*x # for example

    def forward(self, inputs):
        # Initialization/type checking code.
        # (only gets called once, during first iteration)
        type_check_blah(inputs)

        # Allocate output array. Note that since this line
        # is not wrapped using @static_schedule_func, it
        # will only ever get called once, during the first
        # iteration.
        y = xp.empty(y_shape).astype(x.dtype)

        # Call static function
        # (it will get called every iteration from optimized schedule)
        self.static_forward(x, y)
        return y,



It should not be necessary to modify the `backward()` implementation. As of Chainer v3 when double-backward (i.e., grad of grad) support was added, the `backward()` method of `FunctionNode` actually calls the `forward()` method of other `FunctionNode`s, and so it is only necessary to handle the forward case.

How to add support to existing links
------------------------------------

Most existing links will work as-is and do not need to be modified. However, if a link needs to perform computations each iteration that are performed in code other than calling chainer functions, this code will need to be manually placed in a `@static_schedule_func`-decorated function or method of the link.

If a link performs different computations depending on the training mode but is otherwise static, then it does not need to be modified.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.graph_optimizations.static_graph.static_graph

