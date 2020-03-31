from __future__ import print_function

from collections import OrderedDict
import warnings

import chainer

try:
    import onnx
    from onnx import checker
    from onnx import helper
    from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
    from onnx import numpy_helper
    from onnx import shape_inference

    from onnx_chainer.context import Context
    from onnx_chainer.graph import Graph
    from onnx_chainer import mapping
    from onnx_chainer.onnx_helper import is_support_non_standard_domain

    _available = True
except ImportError:
    _available = False

MINIMUM_OPSET_VERSION = 7
MAXIMUM_OPSET_VERSION = 11


def _check_available():
    if not _available:
        raise ImportError(
            'ONNX is not installed on your environment. Exporting your model '
            'in ONNX format needs the onnx package.\n\n'
            '\t$ pip install \'onnx<1.7.0\'\n\n')


def convert_parameter(parameter, context):
    if isinstance(parameter, chainer.Parameter):
        array = parameter.array
    elif isinstance(parameter, chainer.Variable):
        array = parameter.array
    elif isinstance(parameter, chainer.get_array_types()):
        array = parameter
    else:
        raise ValueError(
            'The type of parameter is unknown. It should be either Parameter '
            'or Variable or ndarray, but the type was {}.'.format(
                type(parameter)))
    array = chainer.cuda.to_cpu(array)
    tensor = numpy_helper.from_array(array, context.get_name(parameter))
    return tensor


def rename_variable_name(
        context, variables, named_vars, new_names, prefix='Input'):
    # Update ``named_vars`` keys to ``new_names``
    if isinstance(variables, (list, tuple)):
        if new_names is None:
            new_names = ['{}_{}'.format(prefix, i)
                         for i in range(len(named_vars))]
        if not isinstance(new_names, (list, tuple)) or\
                len(variables) != len(new_names):
            raise ValueError(
                'Replacing name list is not match with input (or output) '
                'variables')
        for i, var in enumerate(variables):
            del named_vars[context.get_name(var)]
            new_name = new_names[i]
            named_vars[new_name] = var
            context.set_name(var, new_name, pinned=True)
    elif isinstance(variables, dict):
        if new_names is None:
            new_names = {k: '{}_{}'.format(prefix, i)
                         for i, k in enumerate(variables.keys())}
        if not isinstance(new_names, (list, tuple, dict)) or\
                len(variables) != len(new_names):
            raise ValueError(
                'Replacing name dict is not match with input (or output) '
                'variables')
        if isinstance(new_names, (list, tuple)):
            new_names = {k: v for k, v in zip(variables.keys(), new_names)}
        for k, v in variables.items():
            if k not in new_names:
                raise ValueError(
                    'Key of replacing name is not found in variables')
            del named_vars[context.get_name(v)]
            new_name = new_names[k]
            named_vars[new_name] = v
            context.set_name(v, new_name, pinned=True)
    elif isinstance(variables, chainer.Variable):
        if not new_names:
            new_names = prefix + '_0'
        if isinstance(new_names, (list, tuple)):
            if len(new_names) != 1:
                raise ValueError('Replacing name must be single')
            new_name = new_names[0]
        elif isinstance(new_names, str):
            new_name = new_names
        else:
            raise ValueError(
                'Type {} is not supported for single variable'.format(
                    type(new_name)))
        del named_vars[context.get_name(variables)]
        named_vars[new_name] = variables
        context.set_name(variables, new_name, pinned=True)


def format_customized_shapes(args, shapes):
    if isinstance(args, (list, tuple)):
        if not isinstance(shapes, list) or len(args) != len(shapes):
            raise ValueError('Customized shapes cannot fit for input list')
        for i, (arg, shape) in enumerate(zip(args, shapes)):
            if len(arg.shape) != len(shape):
                raise ValueError(
                    'Index-{} shape length must be same as input'.format(i))
        return shapes
    elif isinstance(args, dict):
        if not isinstance(shapes, (list, dict)) or\
                len(args) != len(shapes):
            raise ValueError('Customized shapes cannot fit for input dict')
        if isinstance(shapes, list):
            shapes = {k: v for k, v in zip(args.keys(), shapes)}
        formatted_shapes = []
        for k, arg in args.items():
            if k not in shapes:
                raise ValueError(
                    'Key "{}" is not found in customized shapes'.format(k))
            if len(arg.shape) != len(shapes[k]):
                raise ValueError(
                    'Key "{}" shape length must be same as input'.format(k))
            formatted_shapes.append(shapes[k])
        return formatted_shapes
    else:
        assert isinstance(args, (chainer.Variable, chainer.get_array_types()))
        if isinstance(shapes, list):
            if len(shapes) != 1:
                raise ValueError('Customized shape must be single')
        elif not isinstance(shapes, tuple):
            raise ValueError(
                'Type {} is not supported for single input'.format(
                    type(shapes)))
        else:
            shapes = [shapes]
        if len(args.shape) != len(shapes[0]):
            raise ValueError('Shape length must be same as input')
        return shapes


class RetainInputHook(chainer.LinkHook):
    """Retain temporary inputs

    Function nodes manage inputs variable nodes using weak reference. When
    variable is made as temporary value, exporter cannot get the corresponded
    variable from the variable node because the reference is collected. To
    resolve it, retain all inputs and will use when make computational graph.

    To reduce memory size, this hook retains only variables not showed in link
    inputs. To enable this feature, links are required to use ``forward``, not
    ``__call__``.
    """

    def __init__(self):
        self.link_inputs = set()
        self.retain_inputs = []
        self.replaced_inputs = []

        self.org_apply = chainer.function_node.FunctionNode.apply

        def hooked_apply(_self, inputs):
            ret = self.org_apply(_self, inputs)
            func_inodes = list(_self.inputs)
            for i, inode in enumerate(func_inodes):
                referenced_var = inode.get_variable_or_none()
                if referenced_var is None:
                    # This variable is created within function node and weakref
                    # is lost. Make temporary variable and retain it.
                    temp_var = chainer.as_variable(inputs[i])
                    func_inodes[i] = temp_var.node
                    self.retain_inputs.append(temp_var)
                else:
                    if id(referenced_var) not in self.link_inputs:
                        # This variable is created within link forward, outside
                        # of function node. To avoid to lose reference out
                        # of the forward, retain the variable.
                        self.retain_inputs.append(referenced_var)
            self.replaced_inputs.append((_self, _self.inputs))
            _self.inputs = tuple(func_inodes)
            return ret
        self.hooked_apply = hooked_apply

    def _extract_inputs(self, args):
        # Retain only chainer.Variable (and its collection)
        # Other type args are ignored and not checked instance IDs
        # If these variable are used in FunctionNode, they will be retained
        ret = set()
        if isinstance(args, chainer.Variable):
            ret.add(id(args))
        elif isinstance(args, (list, tuple)):
            for arg in args:
                ret |= self._extract_inputs(arg)
        elif isinstance(args, dict):
            for arg in args.values():
                ret |= self._extract_inputs(arg)
        return ret

    def forward_preprocess(self, args):
        self.link_inputs |= self._extract_inputs(args.args)
        self.link_inputs |= self._extract_inputs(args.kwargs)

    def forward_postprocess(self, args):
        self.link_inputs.clear()

    def __enter__(self):
        chainer.function_node.FunctionNode.apply = self.hooked_apply
        return super().__enter__()

    def __exit__(self, *exc_details):
        chainer.function_node.FunctionNode.apply = self.org_apply
        for _self, inputs in self.replaced_inputs:
            _self.inputs = inputs
        super().__exit__(*exc_details)


def export(model, args, filename=None, export_params=True,
           graph_name='Graph', save_text=False, opset_version=None,
           input_names=None, output_names=None, train=False,
           return_named_inout=False, external_converters=None,
           external_opset_imports=None, input_shapes=None, no_testcase=False):
    """Export function for chainer.Chain in ONNX format.

    This function performs a forward computation of the given
    :class:`~chainer.Chain`, ``model``, by passing the given arguments ``args``
    directly. It means, the output :class:`~chainer.Variable` object ``y`` to
    make the computational graph will be created by:

    ``y = model(*args)``

    ``external_converters`` and ``external_opset_imports`` are for external
    custom operator. When some ~chainer.FunctionNode are expected to convert to
    own customized operator, set converter function with ~chainer.FunctionNode
    name.

    >>> import onnx
    >>> def custom_converter(param):
    ...     return onnx.helper.make_node(
    ...         'CustomizedRelu', param.input_names, param.output_names,
    ...         domain='chainer'),
    >>>
    >>> external_converters = {'ReLU': custom_converter}
    >>> external_imports = {'chainer': 0}
    >>>
    >>> model = chainer.Sequential(F.relu)  # set the target model
    >>> args = chainer.Variable(np.random.rand(1,10))  # set dummy input
    >>> onnx_graph = onnx_chainer.export(
    ...     model, args,
    ...     external_converters=external_converters,
    ...     external_opset_imports=external_imports)

    Returned model has ``CustomizedRelu`` node.

    Args:
        model (~chainer.Chain): The model object you want to export in ONNX
            format. It should have :meth:`__call__` method because the second
            argument ``args`` is directly given to the model by the ``[]``
            accessor.
        args (list or dict): The arguments which are given to the model
            directly.
        filename (str or file-like object): The filename used for saving the
            resulting ONNX model. If None, nothing is saved to the disk.
        export_params (bool): If True, this function exports all the parameters
            included in the given model at the same time. If False, the
            exported ONNX model doesn't include any parameter values.
        graph_name (str): A string to be used for the ``name`` field of the
            graph in the exported ONNX model.
        save_text (bool): If True, the text format of the output ONNX model is
            also saved with ``.txt`` extention.
        opset_version (int): The operator set version of ONNX. If not specified
            or ``None`` is given, the latest opset version of the onnx module
            is used. If an integer is given, it will be ensured that all the
            operator version in the exported ONNX file is less than this value.
        input_names (str, list or dict): Customize input names of the graph.
            Number of ``input_names`` must be same as number of ``args``.
            When set dict type, keys must be same as ``args``'s keys.
        output_names (str, list or dict): Customize output name of the graph.
            Number of ``output_names`` must be same as actual outputs from
            ``model``. When set dict type, keys must be same as the key of
            ``model`` output.
        train (bool): If True, output computational graph with train mode.
        return_named_inout (bool): If set True, return ONNX model with named
            inputs, and named outputs.
        external_converters (dict): Add-on converter. Convert functions
            keyed by ~chainer.FunctionNode name.
        external_opset_imports (dict): Import external opset. opset version
            number keyed by domain name.
        input_shapes (tuple, list, dict): Input shape of output graph follows
            the customized shapes if set. When input are collection type, set
            list or dict. Tuple of tuple is not allowed.

    Returns:
        ~onnx.ModelProto or tuple:
            When ``return_named_inout`` is ``False``, return ModelProto as an
            ONNX model. Otherwise return the tuple of ModelProto, named inputs
            and outputs, both inputs and outputs are list of ~chainer.Variable.

    """

    _check_available()

    if not no_testcase and filename is not None:
        warnings.warn(
            'Exporting ONNX without testcases is deprecated. '
            'Use export_testcase instead',
            DeprecationWarning)

    with chainer.using_config('train', train),\
            chainer.using_config('in_recomputing', True),\
            chainer.using_config('enable_backprop', True):
        return _export(
            model, args, filename, export_params, graph_name, save_text,
            opset_version, input_names, output_names, return_named_inout,
            external_converters, external_opset_imports, input_shapes)


def _export(model, args, filename, export_params, graph_name, save_text,
            opset_version, input_names, output_names, return_named_inout,
            external_converters, external_opset_imports, input_shapes):
    if opset_version is None:
        opset_version = min(
            int(onnx.defs.onnx_opset_version()), MAXIMUM_OPSET_VERSION)
    elif opset_version < MINIMUM_OPSET_VERSION or \
            opset_version > MAXIMUM_OPSET_VERSION:
        warnings.warn(
            'ONNX-Chainer has been tested only with opset_version {} ~ {}'
            'The ONNX file exported with your requested opset_version ({}) '
            'may cause some problems because the converters used for the '
            'opset_version have not been tested.'.format(
                MINIMUM_OPSET_VERSION, MAXIMUM_OPSET_VERSION, opset_version))

    if input_shapes is not None:
        # if input shapes are invalid, raise exception before forwarding.
        input_shapes = format_customized_shapes(args, input_shapes)

    with RetainInputHook(), mapping.patch_functions():
        # Forward computation
        context = Context(model)
        network_inputs = OrderedDict()
        if isinstance(args, tuple):
            args = list(args)
        if isinstance(args, list):
            for i, arg in enumerate(args):
                if isinstance(arg, chainer.get_array_types()):
                    args[i] = chainer.Variable(arg)
                network_inputs[context.get_name(args[i])] = args[i]
            outputs = model(*args)
        elif isinstance(args, dict):
            for key, arg in args.items():
                if isinstance(arg, chainer.get_array_types()):
                    args[key] = chainer.Variable(arg)
                network_inputs[context.get_name(args[key])] = args[key]
            outputs = model(**args)
        elif isinstance(args, chainer.get_array_types()):
            args = chainer.Variable(args)
            network_inputs[context.get_name(args)] = args
            outputs = model(args)
        elif isinstance(args, chainer.Variable):
            network_inputs[context.get_name(args)] = args
            outputs = model(args)
        else:
            raise ValueError(
                'The \'args\' argument should be a list, tuple, dict, '
                'numpy array, or Chainer Variable. But a {} object was '
                'given.'.format(type(args)))
        rename_variable_name(context, args, network_inputs, input_names)

        initializers = []
        input_tensors = []
        param_names = set()
        for org_name, param in model.namedparams():
            # `model.namedparams()` has `include_uninit` flag but not use, to
            # output user warning
            if param.array is None:
                warnings.warn(
                    'The parameter \'{}\' is not initialized, skip setting to '
                    'ONNX graph'.format(org_name))
                continue
            name = context.get_name(param)
            param_names.add(name)
            tensor = convert_parameter(param, context)
            initializers.append(tensor)
            input_tensors.append(helper.make_tensor_value_info(
                name, tensor.data_type, tensor.dims))

        for i, (name, var) in enumerate(network_inputs.items()):
            shape = var.shape if input_shapes is None else input_shapes[i]
            input_tensors.append(helper.make_tensor_value_info(
                name, NP_TYPE_TO_TENSOR_TYPE[var.dtype], shape))

        if external_converters:
            chainer.utils.experimental('external_converters')
            converters = dict(mapping.converters, **external_converters)
        else:
            converters = mapping.converters

        if isinstance(outputs, (list, tuple)):
            flat_outputs = outputs
        elif isinstance(outputs, dict):
            flat_outputs = list(outputs.values())
        elif isinstance(outputs, chainer.Variable):
            flat_outputs = [outputs]
        else:
            raise RuntimeError(
                'Unexpected output type from the model: {}'.format(
                    type(outputs)))
        if not all([isinstance(o, chainer.Variable) for o in flat_outputs]):
            raise ValueError('The all \'outputs\' must be Chainer Variable')
        network_outputs = OrderedDict(
            [(context.get_name(var), var) for var in flat_outputs])
        if output_names:
            rename_variable_name(
                context, outputs, network_outputs, output_names)

        o = Graph(context, converters, opset_version,
                  param_names | set(network_inputs.keys()),
                  network_outputs)
        o.to_onnx_graph()

    implicit_input_names = set(context.implicit_inputs.keys())
    for name in implicit_input_names:
        tensor = convert_parameter(context.implicit_inputs[name], context)
        initializers.append(tensor)
        input_tensors.append(helper.make_tensor_value_info(
            name, tensor.data_type, tensor.dims))

    # If additional parameters are created during conversion
    for param in context.parameters:
        tensor = convert_parameter(param, context)
        initializers.append(tensor)
        input_tensors.append(helper.make_tensor_value_info(
            context.get_name(param), tensor.data_type, tensor.dims))

    # Convert output tensors
    output_tensors = []
    for name, var in network_outputs.items():
        output_tensors.append(helper.make_tensor_value_info(
            name, NP_TYPE_TO_TENSOR_TYPE[var.dtype], var.shape))

    if not export_params:
        initializers = []

    onnx_graph = helper.make_graph(
        o.graph, graph_name, input_tensors, output_tensors,
        initializer=initializers)

    opset_imports = [helper.make_operatorsetid('', opset_version)]
    if external_opset_imports:
        chainer.utils.experimental('external_opset_imports')
        for domain, version in external_opset_imports.items():
            opset_imports.append(helper.make_operatorsetid(domain, version))
    model = helper.make_model(
        onnx_graph,
        producer_name='Chainer',
        producer_version=chainer.__version__,
        opset_imports=opset_imports
    )

    model.ir_version = onnx.IR_VERSION
    check_onnx_model(model, external_converters, external_opset_imports)

    if input_shapes is not None:
        for output in model.graph.output:
            for d in output.type.tensor_type.shape.dim:
                d.Clear()
        model = shape_inference.infer_shapes(model)
        check_onnx_model(model, external_converters, external_opset_imports)

    if filename is not None and isinstance(filename, str):
        with open(filename, 'wb') as fp:
            fp.write(model.SerializeToString())
        if save_text:
            with open(filename + '.txt', 'w') as fp:
                print(model, file=fp)
    elif hasattr(filename, 'write'):
        filename.write(model.SerializeToString())

    if return_named_inout:
        chainer.utils.experimental('return_named_inout')
        return model, network_inputs, network_outputs
    return model


def check_onnx_model(onnx_model, external_converters, external_opset_imports):
    try:
        checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        if external_converters is None:
            raise e
        else:
            # ONNX version >= 1.5: default checker skips schema check when
            # non standard domain is set. In ONNX-Chainer, external ops without
            # doamin is also accepted, but show warning.
            # ONNX version < 1.5: the checker does not skip schema check
            # regardless domain is set or not. In ONNX-Chainer, ignore
            # errors when external ops are set.
            if is_support_non_standard_domain():
                if external_opset_imports:
                    raise e
                else:
                    warnings.warn(
                        'ValidationError is occurred but ignored. '
                        'ONNX-Chainer recommends to set '
                        '`external_opset_imports` when using '
                        '`external_converters` on exporting. Please take care '
                        'about ONNX format check is insufficient. Error '
                        'message:\n{}'.format(str(e)), UserWarning)
            else:
                warnings.warn(
                    'ValidationError is occurred but ignored because '
                    'exporting with `external_converters`. Please take care '
                    'about ONNX format check is insufficient. Error '
                    'message:\n{}'.format(str(e)), UserWarning)
