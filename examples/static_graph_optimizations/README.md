
#### Demonstration of  "static define-by-run" feature

This is an experimental feature that can be used to improve the runtime performance of a model by performing static graph optimizations. Chainer is a "dynamic graph" deep learning framework that makes it easy to create a model by writing Python code that imperatively executes the forward-pass computations. This is what we call "define-by-run." Since the define-by-run code executes every iteration, this means that conditional statements can be potentially included to create a modified or even completely different computation graph each time it is called.

An advantage of dynamic graph frameworks is that they typically allow easier debugging compared to static graph frameworks. Since the forward-pass computations are written in Python, it is straightforward to step through this code in a debugger. If there is an error in the model, it will usually cause the Python interpreter to throw an exception and print a stack trace near the line of code containing the bug. However, a static graph framework might instead first optimize the graph before trying to execute the model, so that finding the bug in the optimized code could be considerably more difficult.

Unfortunately, the flexibility of define-by-run can also carry a performance penalty. This is because the full computation graph is recreated every iteration. In Chainer, this involves creating many objects (for example, each function call creates a new `FunctionNode` object as well as creating new `Variable` and array instances for each output of the function). In addition to this, there is also code to perform dynamic type checking, and graph traversal code for the backward pass etc. It turns out that this dynamic code can sometimes become a significant performance bottleneck for models that contain many layers and/or perform a small amount of computation per mini-batch.

However, it seems that the most of the currently popular deep learning models correspond to either a static graph or a mostly static graph. This suggests that for most models most of the time, it may not actually be necessary to recreate the computation graph and perform these dynamic checks each iteration. That is, if a large part of the computation graph is static, then it should be possible to only create that subgraph once, optimize it, and then reuse it again in future iteration when needed. For example, as long as the input types to a static sub-graph do not change, then there is no need to keep performing dynamic type checking each iteration for nodes inside the sub-graph. Furthermore, it should be possible to simply replace the subgraph with a static schedule and potentially perform other graph optimizations as well.

The purpose of the "static define-by-run" feature is to provide a simple way for the user to tell the framework that a model or sub-graph is static so that the framework can then enable various performance optimizations. The current implementation only performs some basic optimizations, such as removing most dynamic type checking, using statically-allocated arrays (for both the activations and parameters), and using a static schedule for the forward and backward pass. However, these kinds of optimizations are still sufficient to significantly reduce the performance bottlenecks observed in several models.

Note that conceptually, this optimization strategy corresponds to optimization by removal of redundant code in addition to optional graph optimizations. That is, during the first iteration as the define-by-run code executes, the functions that perform the necessary deep learning computations are executed, but so are the various other operations that perform various dynamic type-checking, allocation of Python objects to build the backward computation graph, etc. If we must assume a dynamic graph, then such dynamic code must execute during every iteration, slowing down performance. However, if a static graph can instead be assumed, then there is no longer any reason to run these dynamic allocations and/or checks once the first iteration has completed.

In designing this feature, we wish to retain the ease of model implementation and debugging that is already provided by Chainer while also making the runtime performance more competitive with static graph frameworks. We think a reasonable compromise consists of the following design choices.

* The user-facing API should be almost unchanged compared to existing Chainer. Ideally, the user should only need to add a decorator the `Chain`s in the model corresponding to the largest static sub-graphs. We will call such a chain a "static chain."
* The first iteration of the model (that is, the first forward and backward pass) should execute as define-by-run to allow for easy debugging.
* Starting from the second iteration, the execution mode will change so that optimized static schedule code will be used to execute any static chains, potentially making the runtime performance similar to that of a static graph framework. This switch from define-by-run to static mode should be invisible to the user. We assume that if a static sub-graph in the model is able to successfully execute as define-by-run for one iteration, that it is safe to then replace the define-by-run code with corresponding optimized code.
* It should be possible to export the static sub-graph to optimized C or C++ code that can execute without any dependencies on Python. This is intended to support easier deployment of Chainer models to platforms that do not support Python. The current implementation does not support this capability, but it may be considered as future work.



#### Usage

To use this feature, you will need to identify the largest static sub-graphs in your model and mark them as being "static." Each of these static sub-graphs should correspond to a `Chain` such that the define-by-run code inside its `__call__()` method corresponds to a static sub-graph. Then, to use this feature, simply use the following import

```
from chainer.graph_optimizations.static_graph import static_graph
```

and decorate `__call__()` with `@static_graph` to inform the framework that it is a static chain. No other changes to user code should be necessary. There is also no need to decorate any other chains that are deeply called since they are implicitly assumed to be static as well.

#### Limitations

* This feature has not been tested with models that compute second order gradients.

* The generated static schedules cannot yet be easily inspected. Such a feature might be useful for debugging.

* Do not place any code with side effects inside `__call__()` when using `@static_graph`. For example, code that increments a counter, prints status messages etc. will not be included by default in the optimized static schedule and will therefore never be called again starting from the second iteration. If you want to place such code inside `__call__()` and have it actually get called again each iteration, you will need put it inside a function decorated with `@static_schedule_func` and call that function.

* All existing Chainer functions will need to have their implementation slightly modified to support this feature (see below for details). todo: perhaps this can be automated? Currently only a small number of functions have been modified. If you try to run a model that calls an unsupported function, it will exit with an exception.

* Advanced optimizations such as kernel fusion are not yet implemented.

* The current implementation is not memory-optimized and also performs some redundant copy operations. This is planned to be improved in the future.

* Be careful not to delete the `data` or `grad` attributes of any `Parameter`s in a static graph and do not change their references. Any updates to these attributes, such as those performed by an optimizer, `cleargrads()` etc. should be in-place. (This limitation could potentially be lifted in the future, but also with some performance penalty).

* The current implementation assumes that the arguments to `__call__()` of a static chain consist of an individual arrays or variables. Therefore, if you pass in the variables in a different way, such as a tuple of variables, it will not work. This limitation could be removed or made less restrictive in the future.

#### How to add support to existing functions

An existing function (instance of `FunctionNode`) can be modified to support static graph optimizations as follows. The basic idea is to wrap any code that needs to be called each iteration inside a function that is decorated with `@static_schedule_func`. Therefore, code that performs initialization such as initializing parameters does not need to (and should not) be wrapped.

Since the function is part of a static graph, any parameters and output arrays should ideally be statically allocated only once during the first iteration (while the define-by-run code is executing) and then reused starting from the second iteration. The `@static_schedule_func`-decorated functions that are called each iteration will perform the various deep learning computations, writing results in-place into these static arrays. Since the results are written in-place, there is no need for an `@static_schedule_func`-decorated function to explicitly return a result and so we disallow it. Rather, any results arrays should be passed as inputs along with other input arguments to the function. The following code shows the typical pattern for performing the forward computations in a `FunctionNode`:

```

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

```

Lazy approach: If you are in a hurry and want to quickly wrap an existing function, try the following method. Note that in some functions, it is not documented or not immediately clear what the output array shapes should be. In this case, we can take the lazy approach and just call the existing code that computes the outputs. If the function is used inside a static chain, these outputs will be used as initialization for the `@static_schedule_func`-decorated function. For an example of this, see the following from a hastily-modified relu.py:


```
def _dynamic_forward_cpu(self, x):
    self.retain_outputs((0,))
    return utils.force_array(numpy.maximum(x[0], 0, dtype=x[0].dtype))

@static_schedule_func
def _static_forward_cpu(self, x, y):
    y[:] = self._dynamic_forward_cpu(x)

def forward_cpu(self, x):
    #self.retain_outputs((0,))
    #return utils.force_array(numpy.maximum(x[0], 0, dtype=x[0].dtype)),
    y = self._dynamic_forward_cpu(x)
    if is_trace_mode():
        # Recompute the sum, but do not reallocate the results array.
        self._static_forward_cpu(x, y)
    return y,

def _dynamic_forward_gpu(self, x):
    if chainer.should_use_cudnn('==always') and x[0].flags.c_contiguous:
        # cupy.activation_backward requires the input.
        # So, we retain it for backward computation.
        self.retain_inputs((0,))
        self._use_cudnn = True
        y = cudnn.activation_forward(x[0], _mode)
    else:
        y = cuda.cupy.maximum(x[0], 0)
    self.retain_outputs((0,))
    return y

@static_schedule_func
def _static_forward_gpu(self, x, y):
    y[:] = self._dynamic_forward_gpu(x)

def forward_gpu(self, x):
    #if chainer.should_use_cudnn('==always') and x[0].flags.c_contiguous:
    #    # cupy.activation_backward requires the input.
    #    # So, we retain it for backward computation.
    #    self.retain_inputs((0,))
    #    self._use_cudnn = True
    #    y = cudnn.activation_forward(x[0], _mode)
    #else:
    #    y = cuda.cupy.maximum(x[0], 0)
    #self.retain_outputs((0,))
    y = self._dynamic_forward_gpu(x)
    if is_trace_mode():
        # Recompute the sum, but do not reallocate the results array.
        self._static_forward_gpu(x, y)
    return y,
```

The `is_trace_mode()` function returns `True` only if the function is called from inside a static chain.

It should not be necessary to modify the `backward()` implementation. As of Chainer v3 when double-backward (i.e., grad of grad) support was added, the `backward()` method of `FunctionNode` actually calls the `forward()` method of other `FunctionNode`s, and so it is only necessary to handle the forward case.
