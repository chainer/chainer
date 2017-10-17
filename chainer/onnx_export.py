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
    print(input_names)

    layer_name = _layers[link.__class__.__name__]
    unique_layer_name = os.path.dirname(input_names[1])
    out_names = [str(id(out())) for out in link.outputs]
    out_names += [
        os.path.join(unique_layer_name, 'mean'),
        os.path.join(unique_layer_name, 'var'),
        os.path.join(unique_layer_name, 'saved_mean'),
        os.path.join(unique_layer_name, 'saved_var')
    ]
    print(out_names)

    node = helper.make_node(
        layer_name, input_names, out_names,
        epsilon=link.eps,
        is_test=chainer.config.train,
        momentum=link.decay,
        spatial=True,
    )
    checker.check_node(node)
    return node


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


def export(model, args, filename, export_params=True, name='Graph',
           producer='Chainer'):
    _check_available()

    model.to_cpu()
    args = args if isinstance(args, (list, tuple)) else (args,)
    outputs = model(*args)

    graph = []
    parameters = []
    param_names = {}
    for name, param in model.namedparams():
        param_names[id(param)] = name
        parameters.append(
            convert_parameter(param, param_names))

    if isinstance(outputs, dict):
        outputs = list(outputs.values())
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)

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
                    input_names.append(id(input_))
                    # layer_unique_name = os.path.join(
                    #     func_name, '{}'.format(cand.rank))
                    # print('-' * 20)
                    # print(layer_unique_name)
                    # if input_.creator_node is not None:
                    #     print('input_:', input_)
                    #     print('input_.creater_note.outputs[0]():',
                    #           input_.creator_node.outputs[0]())
                    #     print('cand.outputs:', cand.outputs[0]())

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
                    param_names[id(cand.running_var)] = os.path.join(
                        layer_name, 'running_var')
                    parameters.append(
                        numpy_helper.from_array(
                            cand.running_mean,
                            param_names[id(cand.running_mean)]))
                    parameters.append(
                        numpy_helper.from_array(
                            cand.running_var,
                            param_names[id(cand.running_var)]))

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

    input_tensors = []
    for i, arg in enumerate(args):
        name = 'data{}'.format(i)
        if isinstance(arg, chainer.Variable):
            arg = arg.array
        input_tensors.append(helper.make_tensor_value_info(
            name, _dtype[arg.dtype.name], arg.shape))

    output_tensors = []
    for i, output in enumerate(outputs):
        name = 'output{}'.format(i)
        if isinstance(output, chainer.Variable):
            output = output.array
        output_tensors.append(helper.make_tensor_value_info(
            name, _dtype[output.dtype.name], output.shape))

    if not export_params:
        parameters = []
    onnx_graph = helper.make_graph(
        reversed(graph), name, input_tensors, output_tensors,
        initializer=parameters)
    model = helper.make_model(
        onnx_graph,
        producer_name='Chainer',
        producer_version=chainer.__version__)
    checker.check_model(model)
    with open(filename, 'wb') as fp:
        fp.write(model.SerializeToString())

if __name__ == '__main__':
    import chainer.links as L
    import chainer.functions as F

    # # Network definition
    # class MLP(chainer.Chain):
    #
    #     def __init__(self, n_units, n_out):
    #         super(MLP, self).__init__()
    #         with self.init_scope():
    #             # the size of the inputs to each layer will be inferred
    #             self.l1 = L.Linear(None, n_units)  # n_in -> n_units
    #             self.l2 = L.Linear(None, n_units)  # n_units -> n_units
    #             self.l3 = L.Linear(None, n_out)  # n_units -> n_out
    #
    #     def __call__(self, x):
    #         h1 = F.relu(self.l1(x))
    #         h2 = F.relu(self.l2(h1))
    #         return self.l3(h2)
    #
    # model = MLP(1, 10)
    model = L.ResNet50Layers()
    args = numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)
    export(model, args, 'resnet50.onnx')
