import heapq
import os

import numpy

import chainer
from chainer import function_node
from chainer import variable

try:
    from onnx.onnx_pb2 import TensorProto  # NOQA
    from onnx import checker
    from onnx import helper
    from onnx import numpy_helper

    _available = True

    _dtype = {
        numpy.float16.__name__: TensorProto.FLOAT16,
        numpy.float32.__name__: TensorProto.FLOAT,
        numpy.bool.__name__: TensorProto.BOOL,
    }

    _layers = {
        'LinearFunction': 'Gemm',
        'Reshape': 'Reshape',
        'Convolution2DFunction': 'Conv',
        'AveragePooling2D': 'AveragePool',
        'MaxPooling2D': 'MaxPool',
        'BatchNormalization': 'BatchNormalization',
        'ReLU': 'Relu',
        'Softmax': 'Softmax',
        'Add': 'Add',
    }

except (ImportError, TypeError) as e:
    print(e)
    _available = False


def _check_available():
    if not _available:
        raise ImportError('ONNX is not installed on your environment. '
                          'Exporting your model in ONNX format needs the onnx '
                          'package.\n\n'
                          '  $ pip install onnx\n')


def convert_parameter(parameter, param_names):
    return numpy_helper.from_array(parameter.array, param_names[id(parameter)])


def convert_convolution_2d_function(link, input_names, param_names):
    W = convert_parameter(link.W, param_names)
    input_names[input_names.index(id(link.W))] = W.name
    if hasattr(link, 'b'):
        b = convert_parameter(link.b, param_names)
        input_names[input_names.index(id(link.b))] = b.name

    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = _layers[link.__class__.__name__]
    out_names = [str(id(out())) for out in link.outputs]

    return helper.make_node(
        layer_name, input_names, out_names,
        kernel_shape=link.W.shape,
        strides=(link.sy, link.sx),
        pads=(link.ph, link.pw)
    )


def convert_linear_function(link, input_names, param_names):
    W = convert_parameter(link.W, param_names)
    input_names[input_names.index(id(link.W))] = W.name
    if hasattr(link, 'b'):
        b = convert_parameter(link.b, param_names)
        input_names[input_names.index(id(link.b))] = b.name
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = _layers[link.__class__.__name__]
    out_names = [str(id(out())) for out in link.outputs]

    return helper.make_node(
        layer_name, input_names, out_names,
        alpha=1.,
        beta=1.,
        broadcast=True,
        transA=False,
        transB=False,
    )


def convert_reshape(func, input_names, param_names):
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = _layers[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    return helper.make_node(
        layer_name, input_names, out_names,
        shape=func.shape
    )


def convert_average_pooling_2d(func, input_names, param_names):
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = _layers[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    return helper.make_node(
        layer_name, input_names, out_names,
        kernel_shape=(func.kh, func.kw),
        pads=(func.ph, func.pw),
        strides=(func.sy, func.sx)
    )


def convert_max_pooling_2d(func, input_names, param_names):
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = _layers[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    return helper.make_node(
        layer_name, input_names, out_names,
        kernel_shape=(func.kh, func.kw),
        pads=(func.ph, func.pw),
        strides=(func.sy, func.sx)
    )


def convert_batch_normalization(link, input_names, param_names):
    gamma_idx = input_names.index(id(link.gamma))
    input_names[gamma_idx] = param_names[id(link.gamma)]
    beta_idx = input_names.index(id(link.beta))
    input_names[beta_idx] = param_names[id(link.beta)]
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)
    input_names.append(param_names[id(link.running_mean)])
    input_names.append(param_names[id(link.running_var)])

    layer_name = _layers[link.__class__.__name__]
    unique_layer_name = os.path.dirname(input_names[1])
    out_names = [str(id(out())) for out in link.outputs]
    if chainer.config.train:
        out_names += [
            os.path.join(unique_layer_name, 'mean'),
            os.path.join(unique_layer_name, 'var'),
            os.path.join(unique_layer_name, 'saved_mean'),
            os.path.join(unique_layer_name, 'saved_var')
        ]

    return helper.make_node(
        layer_name, input_names, out_names,
        epsilon=link.eps,
        is_test=not chainer.config.train,
        momentum=link.decay,
        spatial=True,
        consumed_inputs=[False, False, False, True, True],
    )


def convert_relu(func, input_names, param_names):
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = _layers[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]
    return helper.make_node(layer_name, input_names, out_names)


def convert_softmax(func, input_names, param_names):
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = _layers[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    return helper.make_node(
        layer_name, input_names, out_names,
        axis=func.axis
    )


def convert_add(func, input_names, param_names):
    for i, input_name in enumerate(input_names):
        if type(input_name) is not str:
            input_names[i] = str(input_name)

    layer_name = _layers[func.__class__.__name__]
    out_names = [str(id(out())) for out in func.outputs]

    return helper.make_node(layer_name, input_names, out_names)


def export(model, args, filename, export_params=True, graph_name='Graph',
           producer='Chainer'):
    _check_available()

    model.to_cpu()
    args = list(args) if isinstance(args, (list, tuple)) else [args]
    for i, arg in enumerate(args):
        if not isinstance(arg, chainer.Variable):
            args[i] = chainer.Variable(arg)
    outputs = model(*args)

    input_tensor_ids = [id(arg) for arg in args]

    graph = []
    parameters = []
    param_names = {}
    input_tensors = []
    for name, param in model.namedparams():
        param_names[id(param)] = name
        parameters.append(
            convert_parameter(param, param_names))
        input_tensors.append(helper.make_tensor_value_info(
            name, _dtype[param.array.dtype.name], param.shape))

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

                # If it's a parameter
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
                            str(id(out_)), _dtype[out_var.array.dtype.name],
                            out_var.shape)

            if func_name in _layers.keys():
                if func_name == 'Convolution2DFunction':
                    graph.append(
                        convert_convolution_2d_function(
                            cand, input_names, param_names))
                elif func_name == 'LinearFunction':
                    graph.append(
                        convert_linear_function(
                            cand, input_names, param_names))
                elif func_name == 'Reshape':
                    graph.append(
                        convert_reshape(cand, input_names, param_names))
                elif func_name == 'AveragePooling2D':
                    graph.append(
                        convert_average_pooling_2d(
                            cand, input_names, param_names))
                elif func_name == 'MaxPooling2D':
                    graph.append(
                        convert_max_pooling_2d(
                            cand, input_names, param_names))
                elif func_name == 'BatchNormalization':
                    layer_name = os.path.dirname(param_names[id(cand.gamma)])

                    # Add running_mean and running_var to graph
                    param_names[id(cand.running_mean)] = os.path.join(
                        layer_name, 'running_mean')
                    parameters.append(
                        numpy_helper.from_array(
                            cand.running_mean,
                            param_names[id(cand.running_mean)]))
                    input_tensors.append(
                        helper.make_tensor_value_info(
                            param_names[id(cand.running_mean)],
                            _dtype[cand.running_mean.dtype.name],
                            cand.running_mean.shape)
                    )

                    param_names[id(cand.running_var)] = os.path.join(
                        layer_name, 'running_var')
                    parameters.append(
                        numpy_helper.from_array(
                            cand.running_var,
                            param_names[id(cand.running_var)]))
                    input_tensors.append(
                        helper.make_tensor_value_info(
                            param_names[id(cand.running_var)],
                            _dtype[cand.running_var.dtype.name],
                            cand.running_var.shape)
                    )

                    graph.append(
                        convert_batch_normalization(
                            cand, input_names, param_names))
                elif func_name == 'ReLU':
                    graph.append(convert_relu(cand, input_names, param_names))
                elif func_name == 'Softmax':
                    graph.append(
                        convert_softmax(cand, input_names, param_names))
                elif func_name == 'Add':
                    graph.append(
                        convert_add(cand, input_names, param_names))

    # Add all the input values for the network to input_tensors
    for i, arg in enumerate(args):
        name = str(id(arg))
        input_tensors.append(helper.make_tensor_value_info(
            name, _dtype[arg.array.dtype.name], arg.shape))

    output_tensors = []
    for out_ in output_tensor_ids:
        output_tensors.append(helper.make_tensor_value_info(*out_))

    if not export_params:
        parameters = []

    onnx_graph = helper.make_graph(
        reversed(graph), graph_name, input_tensors, output_tensors,
        initializer=parameters)

    model = helper.make_model(
        onnx_graph,
        producer_name='Chainer',
        producer_version=chainer.__version__)

    checker.check_model(model)

    with open(filename, 'wb') as fp:
        fp.write(model.SerializeToString())

    print(model)


if __name__ == '__main__':
    import chainer.links as L
    import chainer.functions as F

    # Network definition
    class MLP(chainer.Chain):

        def __init__(self, n_units, n_out):
            super(MLP, self).__init__()
            with self.init_scope():
                # the size of the inputs to each layer will be inferred
                self.l1 = L.Convolution2D(None, n_units, 3, 1, 1)  # n_in -> n_units
                self.b1 = L.BatchNormalization(n_units)
                self.l2 = L.Linear(None, n_units)  # n_units -> n_units

        def __call__(self, x):
            h = self.b1(F.relu(self.l1(x)))
            return self.l2(h)

    model = MLP(1, 10)
    args = numpy.random.rand(1, 1, 5, 5).astype(numpy.float32)
    # model = L.ResNet50Layers()
    # args = numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)
    export(model, args, 'resnet50.onnx')
