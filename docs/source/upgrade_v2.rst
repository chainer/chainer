.. currentmodule:: chainer

Upgrade Guide from v1 to v2
===========================

This documentation provides detailed information of differences between Chainer v1 and v2.
You will know by reading it which part of your code is required (or recommended) to be fixed when you upgrade Chainer from v1 to v2.

.. contents::
   :local:

CuPy
----

.. _upgrade-cupy-separation:

CuPy has been separated from Chainer into a separate package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy, which was originally a part of Chainer, has been separated into a different Python package since Chainer v2.
It changes the way to set up Chainer with CUDA support.
In particular, you have to separately install :mod:`cupy` package to enable CUDA support.
See :ref:`install-guide` for the recommended installation steps.

**Fortunately, there is no need of updating your source code to catch up with this change.**


Global configurations
---------------------

.. _upgrade-train-mode:

Training mode is configured by a thread-local flag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, the concept of *training mode* is added.
It is represented by a thread-local flag ``chainer.config.train``, which is a part of :ref:`the unified configuration <configuration>`.
When ``chainer.config.train`` is ``True``, functions of Chainer run in the training mode, and otherwise they run in the test mode.
For example, :class:`~links.BatchNormalization` and :func:`~functions.dropout` behave differently in each mode.

In Chainer v1, such a behavior was configured by the ``train`` or ``test`` argument of each function.
**This train/test argument has been removed in Chainer v2.**
If your code is using the ``train`` or ``test`` argument, you have to update it.
In most cases, what you have to do is just removing the ``train`` / ``test`` argument from any function calls.

.. admonition:: Example

   Consider the following model definition and the code to call it in test mode written for Chainer v1.

   .. code-block:: py

      # Chainer v1
      import chainer.functions as F

      class MyModel(chainer.Link):
          ...

          def __call__(self, x, train=True):
              return f(F.dropout(x, train=train))

      m = MyModel(...)
      y = m(x, train=False)

   In Chainer v2, it should be updated into the following code:

   .. code-block:: py

      # Chainer v2
      import chainer.functions as F

      class MyModel(chainer.Link):
          ...

          def __call__(self, x):
              return f(F.dropout(x))

      m = MyModel(...)
      with chainer.using_config('train', False):
          y = m(x)

.. _upgrade-configurations:

Configurations are added and replace some of existing global flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are many global settings moved to :ref:`the unified configuration <configuration>` other than the training mode.
Following is the complete list of the configuration entries that have corresponding features in Chainer v1.

``chainer.config.cudnn_deterministic``
    It is corresponding to the ``deterministic`` argument of some convolution functions in Chainer v1.
    **This argument has been removed since Chainer v2.**
    If you are using this argument, you have to use the ``chainer.config.cudnn_deterministic`` flag to change the behavior of the convolution functions.
``chainer.config.debug``
    It is corresponding to the debug mode in Chainer v1, which was configured by :func:`set_debug` and extracted by :func:`is_debug`.
    These functions are also available in Chainer v2, so you basically do not need to update the code related to the debug mode.
``chainer.config.enable_backprop``
    It is corresponding to the *backprop mode* in Chainer v1.
    The functions :func:`no_backprop_mode` and :func:`force_backprop_mode` are still available in Chainer v2, which automatically turns on/off the ``enable_backprop`` flag.
    One important difference from Chainer v1 is that **the** ``volatile`` **flag is removed from** :class:`Variable`.
    Therefore, there are more situations that you need to modify the ``enable_backprop`` flag.
``chainer.config.keep_graph_on_report``
    This flag configures whether or not to keep the computational graph alive for a reported variable.
    In Chainer v2, when a :class:`Variable` object is reported by :func:`report`, a copy of the variable isolated from the computational graph is created and stored by default.
    Setting ``True`` to this flag, you can change this behavior and then the original :class:`Variable` object is stored as is.
    See :ref:`upgrade-reporter-purge-variable` for the details.
``chainer.config.train``
    It is corresponding to the ``train`` or ``test`` argument of some functions in Chainer v1.
    **This argument has been removed since Chainer v2.**
    If you are using this argument, you have to use the ``chainer.config.train`` flag instead.
    See :ref:`upgrade-train-mode` for more details.
``chainer.config.type_check``
    It is corresponding to the ``Function.type_check_enable`` flag.
    If your code touches this flag, **you have to use** ``chainer.config.type_check`` **instead.**
    Note that the environment variable ``CHAINER_TYPE_CHECK`` is still available in Chainer v2, so if you are only using the environment variable, there is no need of updating your code.
``chainer.config.use_cudnn``
    It is corresponding to the ``use_cudnn`` argument of many functions that have cuDNN implementations.
    **This argument has been removed since Chainer v2.**
    If you are using this argument, you have to use the ``chainer.config.use_cudnn`` flag instead.
    *Note that this flag is ternary, not binary.*
    See :ref:`configuration` for more details.

These configurations can be modified in two ways.

- Simply substituting a new value to an entry, like ``chainer.config.train = False``.
- Using the ``chainer.using_config`` context manager.
  It can be used with the ``with`` statement of Python as follows::

      with chainer.using_config('train', False):
          do something  # this code runs with chainer.config.train == False

  It recovers the original configuration after quitting the ``with`` block.

The ``chainer.config`` manages *the thread-local configuration*.
You can also set the global configuration by modifying ``chainer.global_config``.
Note that the global configuration is used only if the entry of the thread-local configuration is not explicitly set up.


Variable
--------

.. _upgrade-volatile-removed:

Volatile flag is removed
~~~~~~~~~~~~~~~~~~~~~~~~

The :attr:`Variable.volatile` flag has been removed since Chainer v2.

Instead, the configuration ``chainer.config.enable_backprop`` can be used to enable/disable the automatic differentiation feature.
If it is ``True``, Chainer always creates a computational graph on the forward propagation, which corresponds to passing non-volatile variables in Chainer v1.
Otherwise, Chainer does not create a graph, which corresponds to passing volatile variables in Chainer v1.
The biggest difference is that ``enable_backprop`` is a thread-local flag, whereas ``volatile`` was a flag local to each :class:`Variable` object.
Note that ``enable_backprop`` flag has already existed in Chainer v1, which took effect only if all the inputs to the function have ``volatile == 'auto'``.

The ``chainer.config.enable_backprop`` flag can be modified directly or by using :func:`~chainer.using_config`.
See :ref:`configuration` for details.
There is also a convenience function, :func:`no_backprop_mode`, to turn off the flag.

If you are using the ``Variable.volatile`` flag, you have to stop setting this flag (it will not take effect), and set the ``enable_backprop`` flag instead.

.. admonition:: Example

   Let ``model`` be your model, and consider the following code that calls it in volatile mode.

   .. code-block:: py

      # Chainer v1
      x_data = ...   # ndarray
      x = chainer.Variable(x_data, volatile=True)
      y = model(x)

   In Chainer v2, it should be updated as follows.

   .. code-block:: py

      # Chainer v2
      x_data = ...   # ndarray
      x = chainer.Variable(x_data)
      with chainer.no_backprop_mode():
          y = model(x)

.. _upgrade-variable-node:

Variable is not a part of a computational graph anymore
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`Variable` class has been separated into two distinct classes, the :class:`Variable` class and the :class:`VariableNode` class, since Chainer v2.
Every :class:`Variable` object owns its own :class:`VariableNode` object.
A computational graph consists of :class:`Function` objects and :class:`VariableNode` objects.
When one applies a :class:`Function` to a :class:`Variable`, the :class:`VariableNode` object of the variable is extracted and set to one of the inputs of the function.

Note that the underlying data array of the variable is till held by the :class:`Variable` object.
It allows each :class:`Function` implementation to release unneeded arrays from the computational graph, resulting in greatly reduced memory consumption.

**This change does not affect most users' code.**
If you are directly traversing the computational graph by yourself or modifying the graph ad-hoc, you may have to update your code.
In most cases, it is enough to just change :class:`Variable` into :class:`VariableNode` in the code traversing the computational graph.

.. _upgrade-parameter:

Parameter has to be an instance of Parameter class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chainer v2 has a subclass of :class:`Variable` called :class:`Parameter`.
This class has an interface convenient on setting up a parameter variable registered to :class:`Link`.

You basically do not need to update your code because :meth:`Link.add_param` creates a :class:`Parameter` object in Chainer v2.
There is a new recommended way of registering parameters to a link in Chainer v2, though.
:ref:`See here <upgrade-new-param-register>` for the recommended way of parameter registration.

.. _upgrade-variable-changes:

Small changes to Variable
~~~~~~~~~~~~~~~~~~~~~~~~~

There are some changes on the interface and specification of methods.

- ``len(variable)`` returns the length of the first axis of the underlying array in Chainer v2.
  This is equivalent to ``len(variable.data)``.
  It is different from the behavior of Chainer v1, in which ``len`` returned the total number of elements in the underlying array.
- ``repr(variable)`` returns a NumPy-like text representation of the underlying array in Chainer v2.
  In Chainer v1, it just returns a string that shows the name of the variable.


Function
--------

.. _upgrade-split-axis-force-tuple:

The force_tuple option of split_axis is True by default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, the ``force_tuple`` argument of :func:`functions.split_axis` is set to ``True`` by default.
Therefore, it always returns a tuple regardless of the number of sections made after the split.
It was ``False`` by default in Chainer v1.

Type check APIs are updated to enable lazy building of the error messages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, the type check APIs are updated so that the overhead of checking types is greatly reduced.
In order to achieve the overhead reduction, some APIs are changed.

**If you have custom Function implementations that do type checking, you have to update your code.**
The following list shows which part has to be updated.

- Use :func:`utils.type_check.eval` instead of ``Expr.eval``.
- Use :func:`utils.type_check.make_variable` to create a :class:`utils.type_check.Variable` object instead of directly constructing it by yourself.
- Stop using ``.name`` attribute of any expression.

*Background of this change:*
In Chainer v1, the type checking APIs build an abstract syntax tree (AST) based on each expression that tests some condition.
The AST is used to emit a kind error message.
However, building an AST requires constructions of many Python objects, which adds large Python overheads.
In Chainer v2, the :meth:`Function.type_check_forward` method is called once or twice.
At the first call, the type checking APIs run in *light-weight mode*, where it does not build an AST and just checks the condition.
The second call is made only if there is a test that fails, where it builds an AST.
This change makes the ordinary path of running the type checking much faster, while keeping the kind error messages.


Methods to release unneeded arrays are added
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`As is written above <upgrade-variable-node>`, Chainer v2 introduced a new mechanism to reduce the memory consumption of each :class:`Function` implementation.
In many cases, a :class:`Function` implementation does not need some input arrays in its backward computation.
A new method called :meth:`Function.retain_inputs` can be used to specify which input arrays are actually needed.
This method must not be called from the outside of :meth:`Function.forward`.

.. admonition:: Example

   For example, consider the following simple addition function.

   .. code-block:: py

       class AddFunction(chainer.Function):
           def forward(self, inputs):
               return inputs[0] + inputs[1],

           def backward(self, inputs, grad_outputs):
               return grad_outputs[0], grad_outputs[0]

   It can be seen that the backward computation of this function does not use any of the inputs.
   Then, specifying an empty tuple of indexes to :meth:`~Function.retain_inputs` will reduce the memory overhead.

   .. code-block:: py

       class AddFunction(chainer.Function):
           def forward(self, inputs):
               self.retain_inputs(())  # does not retain both inputs
               return inputs[0] + inputs[1],

           def backward(self, inputs, grad_outputs):
               return grad_outputs[0], grad_outputs[0]

In some cases, the function can (or have to) use the output arrays instead of the inputs in its backward computation.
In Chainer v1, we have written code that store the output arrays to attributes of the :class:`Function` object and reuse them in the :meth:`~Function.backward` method.
In Chainer v2, it is recommended to use :meth:`Function.retain_outputs` to declare which outputs are required in the backward computation.
The retained output arrays can be accessed via :attr:`Function.output_data`.

.. note::

   The existing :class:`Function` implementations that store the output arrays to its attributes will run correctly in Chainer v2.
   There is no any memory overhead right now.
   It is recommended to use :meth:`~Function.retain_outputs`, though, so that we can incorporate more memory optimization in the future.

.. admonition:: Example

   For example, consider the following simple implementation of the tanh function.

   .. code-block:: py

       class TanhFunction(chainer.Function):
           def forward(self, inputs):
               xp = chainer.cuda.get_array_module(inputs[0])
               self.y = xp.tanh(inputs[0])
               return self.y,

           def backward(self, inputs, grad_outputs):
               one = self.y.dtype.type(1)  # avoid type promotion
               return grad_outputs[0] * (one - self.y * self.y),

   We can use :meth:`~Function.retain_outputs` instead of preserving the output array by ourselves as follows.

   .. code-block:: py

       class TanhFunction(chainer.Function):
           def forward(self, inputs):
               self.retain_outputs((0,))
               xp = chainer.cuda.get_array_module(inputs[0])
               return xp.tanh(inputs[0]),

           def backward(self, inputs, grad_outputs):
               y = self.output_data[0]
               one = y.dtype.type(1)  # avoid type promotion
               return grad_outputs[0] * (one - y * y)


Link/Chain/ChainList
--------------------

.. _upgrade-wscale-removed:

wscale option is removed from links
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``wscale`` option has been removed from links since Chainer v2.
**If you are using wscale option, you have to update your code.**
The recommended way is to explicitly set the initializer.

.. admonition:: Example

   Consider the case of adding a :class:`~links.Linear` link with the weight initialized by 0.5x of the default initialization.

   .. code-block:: py

       # Chainer v1
       linear = chainer.links.Linear(10, 5, wscale=0.5)

   Note that the default initializer of the weight matrix of :class:`~links.Linear` is a normal distribution of the standard deviation :math:`1 / \sqrt{fan in}`.
   Therefore, it can be fixed as follows.

   .. code-block:: py

       # Chainer v2
       linear = chainer.links.Linear(10, 5, initialW=chainer.initializers.Normal(0.5 / math.sqrt(10)))

   Or, by using the fact that :class:`initializers.HeNormal` provides the initialization with a normal distribution of the standard deviation :math:`scale * \sqrt{2 / fan in}`, the following code is also equivalent to the original.

   .. code-block:: py

       # Chainer v2, using HeNormal
       linear = chainer.links.Linear(10, 5, initialW=chainer.initializers.HeNormal(0.5 / math.sqrt(2))

.. _upgrade-bias-removed:

bias option is removed from links
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, the ``bias`` option is removed from the following links: :class:`~links.Linear`, :class:`~links.Convolution2D`, :class:`~links.Deconvolution2D`, and :class:`~links.DilatedConvolution2D`.
The effect of this argument was duplicated with the ``initial_bias`` option.
Use ``initial_bias`` instead.

The bias vector is enabled by default in N-dimensional convolution links
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, the bias parameter is enabled by default in :class:`~links.ConvolutionND` and :class:`~linkd.DeconvolutionND`.
It was unintentionally disabled by default in Chainer v1.

**If you are using ConvolutionND or DeconvolutionND without specifying the** ``initial_bias`` **argument, you have to fix your code.**
If you want to keep the old behavior (i.e., no bias vector is created by the link), pass ``nobias=True`` to the link at the construction.
Otherwise it will automatically create a bias vector.

.. _upgrade-init-weight-removed:

init_weight function is removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``chainer.initializers.init_weight`` function that was used on weight initialization has been removed since Chainer v2.

**You have to update your code if you are using** ``init_weight``.
In most cases, the update is simple: pass an initializer to :class:`Parameter`.

.. admonition:: Example

   Consider the following code that initializes a weight matrix randomly and a bias vector by zero.

   .. code-block:: py

      # Chainer v1
      class MyLink(chainer.Link):
          def __init__(self):
              super(MyLink, self).__init__(
                  W=(10, 5),
                  b=(5,),
              )
              chainer.initializers.init_weight(self.W, chainer.initializers.Normal(0.05))
              self.b.data.fill(0)
          ...

   This code should be fixed as follows (see the next topic for the use of :class:`Parameter`).

   .. code-block:: py

      # Chainer v2
      class MyLink(chainer.Link):
          def __init__(self):
              super(MyLink, self).__init__()
              self.W = chainer.Parameter(chainer.initializers.Normal(0.05), (10, 5))
              self.b = chainer.Parameter(0, (5,))
          ...

.. _upgrade-gru-changed:

The order of arguments of GRU is changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, the first two arguments of :class:`~links.GRU` is the input size and the output size.
It was reversed in Chainer v1, causing an inconsistent interface compared to other links including :class:`~links.LSTM`.
**If you are using** :class:`~links.GRU`, **you have to update your code.**
The update is done by simply flipping the first two arguments.

.. admonition:: Example

   Consider the following code that creates a :class:`~links.GRU` link.

   .. code-block:: py

       # Chainer v1
       gru = chainer.links.GRU(20, 10)

   It should be fixed into the following code.

   .. code-block:: py

       # Chainer v2
       gru = chainer.links.GRU(10, 20)

   Note that if you were omitting the output size, the code works as is because :class:`~links.GRU` supports :ref:`the omitted input size <update-omit-input-size>`.

   .. code-block:: py

       # Chainer v1/v2
       gru = chainer.links.GRU(20)

.. _upgrade-forget-bias:

The default value of the forget bias for LSTM and StatelessLSTM is changed to 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, the default forget bias value of :class:`~links.LSTM` and :class:`~links.StatelessLSTM` links is changed to 1.
This change is based on `the paper reporting that using a large forget bias improves the training performance <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_.
The new behavior is also consistent with `the implementation of BasicLSTMCell in TensorFlow <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py#L138>`_.

It will improve the most use cases of LSTMs, although this change would break the reproducibility of the existing experiments.
**If you want to keep the same initialization procedure, you have to update your code.**
The change is simple: pass ``forget_bias_init=0`` to :class:`~links.LSTM` and :class:`~links.StatelessLSTM`.

The interfaces of GRU and LSTM are aligned
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v1, :class:`~chainer.links.GRU` was *stateless*, as opposed to the current implementation.
To align with the naming convention of LSTM links, we have changed the naming convention from Chainer v2 so that the shorthand name points the stateful links.
**If you are using** :class:`~links.GRU`**, you have to update your code.**
You can use :class:`~chainer.links.StatelessGRU` for stateless version, whose implementation is identical to ``chainer.linksGRU`` in v1.

Aliases of links in chainer.functions are removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the compatibility reason, there were some links that have aliases in the :mod:`chainer.functions` module.
These aliases are removed in Chainer v2.
Use :mod:`chainer.links` instead.

Parameter link is removed
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``chainer.links.Parameter`` link is removed in Chainer v2.
This link existed in Chainer v1 only for the backward compatibility.
Use :class:`chainer.Parameter` instead (for the new :class:`Parameter` class, see :ref:`upgrade-parameter`).

.. _upgrade-new-param-register:

New-style parameter registration APIs are added to Link
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, :meth:`Link.init_scope` method returns a context manager that automatically registers a :class:`Parameter` object to the link at setting it to an attribute.
If you are using IDE like PyCharm, it is recommended to use this new-style parameter registration so that IDEs can easily detect the existence of the parameter as an attribute.
It is also a good practice to use the new-style API even if you are not using IDEs, if you are planning to make the code public.

.. note::

   The existing code that uses the conventional way of registering parameters are still valid.

.. admonition:: Example

   For example, the following link initialization code

   .. code-block:: py

      # Chainer v1
      class MyLink(chainer.Link):
          def __init__(self):
              super(MyLink, self).__init__(
                  W=(10, 5),
                  b=(5,),
              )
              chainer.initializers.Normal(0.05)(self.W.data)
              self.b.data.fill(0)
          ...

   is recommended to be updated as follows.

   .. code-block:: py

      # Chainer v2
      class MyLink(chainer.Link):
          def __init__(self):
              super(MyLink, self).__init__()
              with self.init_scope():
                  self.W = chainer.Parameter(chainer.initializers.Normal(0.05), (10, 5))
                  self.b = chainer.Parameter(0, (5,))  # initialize by zero
          ...

.. note::

   To keep a :class:`Parameter` object as an attribute without registration, you can set the attribute without using the ``with self.init_scope():`` block.

New-style child link registration APIs are added to Chain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like :class:`Parameter`, a :class:`Link` object is also automatically registered to a :class:`Chain` object by substitution to an attribute within a :meth:`~Link.init_scope` scope.
If you are using IDE like PyCharm, it is recommended to use the new-style child link registration so that IDEs can easily detect the existence of the child link as an attribute.
It is also a good practice to use the new-style API even if you are not using IDEs, if you are planning to make the code public.

.. note::

   The existing code that uses the conventional way of registering child links are still valid.

.. admonition:: Example

   For example, the following chain initialization code

   .. code-block:: py

      # Chainer v1
      class MyMLP(chainer.Chain):
          def __init__(self):
              super(MyMLP, self).__init__(
                  layer1=L.Linear(None, 20),
                  layer2=L.Linear(None, 30),
              )
          ...

   is recommended to be updated as follows.

   .. code-block:: py

      # Chainer v2
      class MyMLP(chainer.Chain):
          def __init__(self):
              super(MyMLP, self).__init__()
              with self.init_scope():
                  self.layer1 = L.Linear(20)
                  self.layer2 = L.Linear(30)

   Note that this example also demonstrates the use of new APIs with :ref:`the omitted input size <update-omit-input-size>`, explained below.

.. note::

   To keep a :class:`Link` object as an attribute without registration, you can set the attribute without using the ``with self.init_scope():`` block.

.. _update-omit-input-size:

The input-size placeholder of links are made optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, the input size of many links, including :class:`~links.Linear` and :class:`~links.Convolution2D`, is made optional.
In Chainer v1, we had to use ``None`` as the placeholder to specify that the input size should be determined at the first iteration.
The placeholder can also be used in Chainer v2, although it is easier to just omit the input size.

See the previous item for the example of omitting the input size of :class:`~links.Linear`.
The following links currently support the omitted input size.

- :class:`~links.Convolution2D`
- :class:`~links.Deconvolution2D`
- :class:`~links.DilatedConvolution2D`
- :class:`~links.Linear`
- :class:`~links.LSTM`
- :class:`~links.MLPConvolution2D`
- :class:`~links.StatelessLSTM`


Optimizer
---------

Deprecated methods of Optimizer are removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following methods are removed from :class:`Optimizer`.
These methods have been already deprecated in the past versions.
**If you are using these methods, you have to update your code.**

- ``zero_grads``: use :meth:`Link.zerograds` instead.
- ``compute_grads_norm``: you can compute the gradient norm by iterating the list of parameters by :meth:`Link.params`.
- ``clip_grads``: use :class:`~optimizer.GradientClipping` instead.
- ``weight_decay``: use :class:`~optimizer.WeightDecay` instead.
- ``accumulate_grads``: use :meth:`Link.addgrads` instead.

.. _upgrade-gradient-method-cleargrads:

GradientMethod uses Link.cleargrads instead of Link.zerograds by default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, :class:`GradientMethod` clears the gradient before running backprop by :meth:`Link.cleargrads`.
It means that the gradient of each parameter is initialized by ``None`` instead of a zero array.
Note that all the optimizer implementations provided by Chainer are subclasses of :class:`GradientMethod`, and therefore this change affects all of them.

**In most cases, you do not need to update your code.**
If your code relies on the zeroing initialization, you have to fix your code to explicitly initialize the gradient by zero, or to pass ``False`` to :meth:`GradientMethod.use_cleargrads`.

.. _upgrade-update-rule:

GradientMethod is redesigned to allow parameter-specific update rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, the new class :class:`UpdateRule` is used to define an update rule specific to each :class:`Parameter` object.
The :class:`UpdateRule` is set to each :class:`Parameter` object, and is used at each update step.
This object implements an *update formula* using the data and gradient arrays.

Each :class:`UpdateRule` object has :attr:`~UpdateRule.enabled` flag, which configures if the update rule should be applied to that parameter on update.
By setting the flag to ``False``, you can *freeze* the parameter.
There is also a convenient method :meth:`Link.enable_update` and :meth:`Link.disable_update`, which configure the flag of each parameter under the link hierarchy.
In other frameworks, a similar feature is called *layer freezing*.
In Chainer v2, this is officially supported by these methods.

Each :class:`UpdateRule` object can also hold its own hook functions similar to :class:`Optimizer`.
The built-in hook functions except for :class:`~optimizer.GradientClipping` can also be used as a hook function of :class:`UpdateRule`.

**In most cases, you do not have to update your code** because each optimizer automatically sets up an appropriate :class:`UpdaterRule` object to each parameter.

**If you are using a custom gradient-based optimizer implementation, you need to update the implementation.**
The following list shows what you have to do.

- Write a subclass of :class:`UpdateRule` that implements the update rule.
- Rewrite your :class:`GradientMethod` implementation.
  The new implementation only has to set up the update rule for each parameter in the target link.

You can see live examples in `the optimizer implementations provided by Chainer <https://github.com/chainer/chainer/tree/master/chainer/optimizers>`_.


Serializer
----------

None is serializable
~~~~~~~~~~~~~~~~~~~~

In Chainer v2, all serializers start supporting ``None`` value to be serialized and deserialized.
Users' code can rely on this feature, i.e., it can serialize and deserialize ``None`` value with any given serializer.
This change only affects your code if it provides its own serializer implementations.


Trainer and Extension
---------------------

.. _upgrade-pass-raw-arrays-to-loss:

Updater and Evaluator pass raw data arrays to the loss function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, :class:`~training.Updater` and :class:`~training.extensions.Evaluator` pass raw data arrays to the loss function without wrapping them with :class:`Variable`.
**You might need to update your code so that the loss function (in most cases, the model's** ``__call__`` **) accepts raw arrays.**

Note that raw arrays can be directly passed to any :class:`Function`; they are automatically wrapped by :class:`Variable`.
For example, if the input is directly passed to a :class:`Function` object (or any function under :mod:`chainer.functions`), you do not need to update the code.

.. admonition:: Example

   Consider the following code that obtains the shape of the input via :attr:`Variable.data`.

   .. code-block:: py

       # Chainer v1
       class MyLink(chainer.Link):
           def __call__(self, x):
               shape = x.data.shape  # valid if x is Variable, invalid if x is ndarray
               ...

   It should be updated so that the link also accepts a raw array as the input.
   In this case, we have :attr:`Variable.shape` which is equivalent to ``data.shape``, so you can simply write as follows.

   .. code-block:: py

       # Chainer v2
       class MyLink(chainer.Link):
           def __call__(self, x):
               shape = x.shape  # valid regardless of x being Variable or ndarray
               ...

.. _upgrade-snapshot-trigger-removed:

trigger option is removed from snapshot and snapshot_object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, the ``trigger`` option is removed from the :func:`~training.extensions.snapshot` and :func:`~training.extenisons.snapshot_object` extensions.
The effect of the option was duplicated with the ``trigger`` option of :meth:`Trainer.extend <training.Trainer.extend>`.
**If you are passing the** ``trigger`` **argument to these extensions, you have to update your code.**
The update can be done by passing the value to the corresponding :meth:`Trainer.extend <training.Trainer.extend>`.

.. admonition:: Example

   Assume that ``trainer`` is an instance of :class:`~training.Trainer`, and consider that you were adding a :func:`~training.extensions.snapshot` extension as follows.

   .. code-block:: py

       # Chainer v1
       trainer.extend(chainer.training.extensions.snapshot(trigger=(1000, 'iteration')))

   It should be updated as follows (note that this code also works with Chainer v1).

   .. code-block:: py

       # Chainer v1/v2
       trainer.extend(chainer.training.extensions.snapshot(), trigger=(1000, 'iteration'))

.. _upgrade-invoke-before-training-removed:

Extension.invoke_before_training is removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, The attribute ``invoke_before_training`` of :class:`~training.Extension` is removed.
Instead, the :meth:`Extension.initialize <training.Extension.initialize>` method is added.
This method is called by :meth:`Trainer.run <training.Trainer.run>` before entering the training loop.

In Chainer v1, the extension is just called before entering the training loop when ``invoke_before_training`` is ``True``.
**If you have a custom extension that has** ``invoke_before_training=True`` **, you have to update the code.**
What you have to do is to remove the ``invoke_before_training`` flag and override :meth:`~training.Extension.initialize` method.
If you are using the :func:`~training.make_extension` decorator, you can set the ``initialize`` function by passing the ``initializer`` argument to :func:`~training.make_extension`.

.. _upgrade-dump-graph-only-once:

The dump_graph extension dumps the valid graph only at its first invocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, the :func:`~training.extensions.dump_graph` extension dumps the valid computational graph only at its first invocation.
**If you want to dump the graph more than once, you have to fix the code.**
The easiest fix is setting the ``chainer.config.keep_graph_on_report`` flag to ``True``.
*Note that this fix will cancel the improvement on the memory consumption made in Chainer v2.*
More memory-efficient fix is to dump the graph without using an extension, e.g. by customizing the loss function or the updater.

Here is the background of this change.
In Chainer v2, :ref:`the Reporter copies reported variables with purging the computational graph by default. <upgrade-reporter-purge-variable>`
On the other hand, the :func:`~training.extensions.dump_graph` extension requires the computational graph reachable from the reported variable.
In order to make the graph available, the :func:`~training.extensions.dump_graph` extension turns on the ``chainer.config.keep_graph_on_report`` flag at its initializer (i.e., it turns on the graph before entering the training loop).
Since we also wanted to achieve the memory efficiency, the :func:`~training.extensions.dump_graph` extension **turns off the flag after dumping the graph at its first invocation** (strictly speaking, it recovers the original value).
As a result, the computational graph is not available from the second invocation.

Since the :func:`~training.extensions.dump_graph` recovers the original flag value at its invocation, you can keep the graph dumped more than once by changing the original flag value.


Reporter
--------

.. _upgrade-reporter-purge-variable:

When a variable is reported, the variable is copied with the graph purged
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Chainer v2, when a :class:`Variable` object is reported using :func:`report` function (or directly using :class:`Reporter`), a copy of the variable is made without preserving the computational graph.
**If your code depends on the reachability of the computational graph from the reported variable, you have to update your code.**
The easiest way to update your code is setting ``chainer.config.keep_graph_on_report`` to ``True``, then Chainer will keep the computational graph reachable from the reported variable.

The possible examples that are affected by this change are as follows (not exhaustive).

- A custom extension that runs backprop from a reported variable.
  It is definitely an example of assuming the reachability of the computational graph from the reported variable.
- An extension that visualizes the computational graph from a reported variable.
  If you are writing such an extension by yourself, you have to turn on the ``keep_graph_on_report`` flag.
  The :func:`~training.extensions.dump_graph` extension is another example, for which see :ref:`the above item <upgrade-dump-graph-only-once>` for the details.

This change is made for the memory performance reason; with this change, the memory used by the computational graph for training is immediately released before invoking extensions.
Therefore, *changing the behavior by overwriting* ``chainer.config.keep_graph_on_report`` *may increase the memory consumption.*
It may cause an out-of-memory error if the computational graph of the loss function consumes almost all the memory available in your environment and there is an extension that uses a certain amount of memory (e.g. :class:`~training.extensions.Evaluator`).

Other utilities
---------------

Some obsolete classes and functions are removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following classes and functions are removed in Chainer v2.

- ``chainer.Flag``
- ``chainer.FunctionSet`` (Use :class:`Chain` or :class:`ChainList` instead)
- ``chainer.cuda.init`` (It did nothing except for calling :func:`~cuda.check_cuda_available`)
- ``chainer.cuda.empty`` (Use :func:`cupy.empty`)
- ``chainer.cuda.empty_like`` (Use :func:`cupy.empty_like`)
- ``chainer.cuda.full`` (Use :func:`cupy.full`)
- ``chainer.cuda.full_like`` (Use :func:`cupy.full_like`)
- ``chainer.cuda.ones`` (Use :func:`cupy.ones`)
- ``chainer.cuda.ones_like`` (Use :func:`cupy.ones_like`)
- ``chainer.cuda.zeros`` (Use :func:`cupy.zeros`)
- ``chainer.cuda.zeros_like`` (Use :func:`cupy.zeros_like`)
