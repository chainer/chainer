Static Subgraph Optimizations: Design Notes
===============================================

.. module:: chainer.graph_optimizations

This documentation is intended provide information on the architecture and design 
of the static subgraph optimizations feature for those who are interested in 
contributing to its development. This documentation also describes how existing
Chainer functions can be modified to run more efficiently when static
subgraph optimizations are enabled.

Overview of dynamic and static graph frameworks
------------------------------------------------

Existing deep learning frameworks can roughly be classified as either a 
"static graph" or "dynamic graph" framework. In a static graph framework, 
which we also call "define-and-run", the computation graph is defined 
before the model is run. This implies that the same neural network model 
will be used each iteration without modifications, hence the name "static." 
This allows various graph optimizations to potentially be performed to 
improve the runtime performance and/or reduce memory usage. The optimized 
code for the computation graph is then used when the model is run.

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
call creates a new `FunctionNode` object as well as creating new `VariableNode` and array memory allocation
for each output of the function. There are also various dynamic type checks and graph
traversal that need to be performed, adding to the runtime overhead. Further, we cannot perform some
optimizations such as function/kernel fusion and in-place operations.

Static subgraph optimizations feature
-------------------------------------------------------------

This feature is motivated by the observation that typical deep neural networks correspond 
to a static computation graph and that even those that correspond to a dynamic graph 
are typically mostly static. By "mostly static", we mean that the largest static 
subgraphs each tend to contain many function nodes (that is, layers) so that the 
total number of function nodes in the graph tends to be much larger than the total 
number of largest static subgraphs. If the graph is at least mostly static, then a 
naive implementation of define-by-run will result in a large amount of redundant 
operations being performed each iteration to rebuild exactly the same subgraphs, 
perform the same dynamic type-checking operations, etc., which can sometimes be 
slow in Python; it will also result in lost opportunities to perform potential graph 
optimizations. A key assumption motivating this feature is that the main performance 
bottlenecks tend to occur inside the largest static subgraphs. So, if we can optimize 
these static subgraphs, it might be fine for any remaining framework code to remain 
implemented in pure Python. Although such Python code would be slow, it could have 
negligible runtime overhead.

The solution proposed by this feature is to retain the existing define-by-run style 
for specifying the model, but to also optionally allow the user to annotate the 
largest static subgraphs in a model. These "static graph" annotations will then 
allow the framework to automatically replace the define-by-run code of the static 
subgraphs with more performance-optimized code. The define-by-run code will still 
execute during the first iteration, to retain ease of debugging. However, as this 
code executes, a trace of the needed computations is also collected so that optimized 
static schedules can be generated for the annotated static subgraphs. Then, starting 
from the second iteration, this optimized code will automatically be run in place 
of the original define-by-run code. Note that in the common case in which the whole 
model is static, the user only needs to add a single "static graph" annotation and 
their code will then run with the performance of a static graph framework, while 
still supporting the define-by-run coding style.

The benefit of annotating the static subgraphs in the model is that it allows the 
define-by-run code to be replaced with an optimized static schedule, which can 
then potentially support a user-controllable trade-off between runtime performance 
and memory usage. This is possible because having the full computation graph 
available enables various optimizations that cannot safely or automatically be 
performed in define-by-run. Examples (which we have not yet implemented; 
contributions from the open source community are welcomed) include sub-linear 
memory usage [1], exploiting graph parallelism, operator fusion, and in-place optimizations.

The current implementation achieves its speedup by retaining only the code that 
is actually needed to compute the forward pass, backward pass, and so on. This 
allows us to remove most of the Python interpreter overhead because the Python 
code that performs dynamic operations such as allocating `FunctionNode` and 
`Variable` objects, checking types, and traversing the backward graph is not 
included in the optimized static schedule code.


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
