from __future__ import print_function

import heapq

import chainer
from chainer import function_node
from chainer import variable
import numpy

from onnx_chainer import functions
from onnx_chainer import mapping

try:
    from onnx import checker
    from onnx import helper
    from onnx import numpy_helper

    _available = True


except (ImportError, TypeError):
    _available = False


def _check_available():
    if not _available:
        raise ImportError(
            'ONNX is not installed on your environment. Exporting your model '
            'in ONNX format needs the onnx package.\n\n'
            '  $ pip install onnx==0.2.1\n')


def convert_parameter(parameter, param_names):
    if isinstance(parameter, chainer.Parameter):
        array = parameter.array
    elif isinstance(parameter, chainer.Variable):
        array = parameter.array
    elif isinstance(parameter, numpy.ndarray):
        array = parameter
    if array.shape == ():
        array = array[None]
    return numpy_helper.from_array(array, param_names[id(parameter)])


def create_node(func_name, cand, input_names, param_names, parameters,
                input_tensors):
    converter_name = 'convert_{}'.format(func_name)
    if hasattr(functions, converter_name):
        converter = getattr(functions, converter_name)
        nodes = converter(
            cand, input_names, param_names, parameters, input_tensors)
    else:
        raise ValueError('{} is not supported.'.format(func_name))
    for node in nodes:
        checker.check_node(node)
    return nodes


def export(model, args, filename=None, export_params=True,
           graph_name='Graph', save_text=False):
    """Export function for chainer.Chain in ONNX format.

    This function performs a forward computation of the given
    :class:`~chainer.Chain`, ``model``, by passing the given argments ``args``
    directly. It means, the output :class:`~chainer.Variable` object ``y`` to
    make the computational graph will be created by:

    y = model(*args)

    Args:
        model (~chainer.Chain): The model object you want to export in ONNX
            format. It should have :meth:`__call__` method because the second
            argment ``args`` is directly given to the model by the ``[]``
            accessor.
        args (list or dict): The argments which are given to the model
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

    Returns:
        A ONNX model object.

    """

    _check_available()

    model.to_cpu()
    args = list(args) if isinstance(args, (list, tuple)) else [args]
    for i, arg in enumerate(args):
        if not isinstance(arg, chainer.Variable):
            args[i] = chainer.Variable(arg)

    if isinstance(args, list):
        outputs = model(*args)
    elif isinstance(args, dict):
        outputs = model(**args)
    else:
        raise ValueError(
            'The \'args\' argument should be a list or dict. But a {} object '
            'was given.'.format(type(args)))

    input_tensor_ids = [id(arg) for arg in args]

    graph = []
    parameters = []
    param_names = {}
    input_tensors = []
    for name, param in model.namedparams():
        param_names[id(param)] = name
        parameters.append(
            convert_parameter(param, param_names))
        param_shape = (1,) if param.shape == () else param.shape
        input_tensors.append(helper.make_tensor_value_info(
            name, mapping.dtypes[param.array.dtype], param_shape))

    if isinstance(outputs, dict):
        outputs = list(outputs.values())
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)
    output_tensor_ids = [id(output) for output in outputs]

    cands = []
    seen_edges = set()
    nodes = set()
    push_count = [0]

    def add_cand(cand):
        heapq.heappush(cands, (-cand.rank, push_count[0], cand))
        push_count[0] += 1

    for o in outputs:
        if isinstance(o, variable.Variable):
            o = o.node
        add_cand(o)
        nodes.add(o)

    while cands:
        _, _, cand = heapq.heappop(cands)
        if isinstance(cand, variable.VariableNode):
            creator = cand.creator_node
            if creator is not None and (creator, cand) not in seen_edges:
                add_cand(creator)
                seen_edges.add((creator, cand))
                nodes.add(creator)
                nodes.add(cand)

        elif isinstance(cand, function_node.FunctionNode):
            func_name = cand.__class__.__name__
            input_names = []
            for input_ in cand.inputs:
                if input_ is not cand and (input_, cand) not in seen_edges:
                    add_cand(input_)
                    seen_edges.add((input_, cand))
                    nodes.add(input_)
                    nodes.add(cand)

                # When input_ is a parameter
                if input_.name is not None:
                    input_names.append(id(input_.get_variable()))
                    setattr(cand, input_.name, input_.get_variable())
                else:
                    if id(input_.get_variable()) in input_tensor_ids:
                        input_id = id(input_.get_variable())
                    else:
                        input_id = id(input_)
                    input_names.append(input_id)

            for out_ in cand.outputs:
                out_ = out_()
                if out_.get_variable() is not None:
                    out_var = out_.get_variable()
                    if id(out_var) in output_tensor_ids:
                        idx = output_tensor_ids.index(id(out_var))
                        output_tensor_ids[idx] = (
                            str(id(out_)), mapping.dtypes[out_var.array.dtype],
                            out_var.shape)

            if func_name in mapping.operators.keys():
                onnx_nodes = create_node(
                    func_name, cand, input_names, param_names, parameters,
                    input_tensors)
                graph.extend(onnx_nodes)

    # Add all the input values for the network to input_tensors
    for i, arg in enumerate(args):
        name = str(id(arg))
        input_tensors.append(helper.make_tensor_value_info(
            name, mapping.dtypes[arg.array.dtype], arg.shape))

    output_tensors = []
    for out_ in output_tensor_ids:
        output_tensors.append(helper.make_tensor_value_info(*out_))

    if not export_params:
        parameters = []

    onnx_graph = helper.make_graph(
        reversed(graph), graph_name, input_tensors, output_tensors,
        initializer=parameters)

    checker.check_graph(onnx_graph)

    model = helper.make_model(
        onnx_graph,
        producer_name='Chainer',
        producer_version=chainer.__version__)

    # TODO(mitmul): Remove this
    model.ir_version = 1

    checker.check_model(model)

    if filename is not None and isinstance(filename, str):
        with open(filename, 'wb') as fp:
            fp.write(model.SerializeToString())
        if save_text:
            with open(filename + '.txt', 'w') as fp:
                print(model, file=fp)
    elif hasattr(filename, 'write'):
        filename.write(model.SerializeToString())

    return model
