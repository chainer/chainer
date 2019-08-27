import collections
import heapq
import os

import numpy
import six

import chainer
from chainer import function
from chainer import function_node
from chainer.links.caffe.protobuf3 import caffe_pb2 as caffe_pb
from chainer import variable


_function_types = (function.Function, function_node.FunctionNode)


def _add_blob(layer, shape, data):
    # The following part is ridiculously slow!!
    # TODO(okuta): Replace with C++ extension call
    blob = layer.blobs.add()
    blob.shape.dim[:] = shape
    blob.data[:] = data.flatten()


def _dump_graph(outputs):
    fan_out = collections.defaultdict(int)
    cand_funcs = []

    def add_cand_to_check(cands):
        for cand in cands:
            x = cand.creator
            if x is None:
                continue
            if x not in fan_out:
                # `len(fan_out)` is in order to avoid comparing `x`
                heapq.heappush(cand_funcs, (-x.rank, len(fan_out), x))
            fan_out[x] += 1

    add_cand_to_check(outputs)
    while cand_funcs:
        _, _, func = heapq.heappop(cand_funcs)
        assert isinstance(func, _function_types)
        add_cand_to_check(func.inputs)

    ret = []
    cand_funcs = []
    seen_set = set()

    def add_cand(cands):
        cands = [cand.creator for cand in cands if cand.creator is not None]
        for x in cands:
            if x in seen_set:
                continue
            order = 1
            if fan_out[x] == 1 and len(cands) == 1:
                order = -len(seen_set)
            # Negate since heapq is min-heap
            # `len(seen_set)` is in order to avoid comparing `x`
            heapq.heappush(cand_funcs, (order, -x.rank, -len(seen_set), x))
            seen_set.add(x)

    add_cand(outputs)
    while cand_funcs:
        _, _, _, func = heapq.heappop(cand_funcs)
        ret.append(func)
        add_cand(func.inputs)

    return ret[::-1]


class _RetrieveAsCaffeModel(object):

    debug = False

    def __init__(self, prototxt, caffemodel=None):
        self.caffemodel = caffemodel
        self.prototxt = prototxt
        # key:string, val:dict(key: func, val: index)
        self.naming_map = collections.defaultdict(dict)

    def _get_layer_name(self, layer):
        """Generate layer name like "Convolution2DFunction-10-2".

        The first number means rank of the layer (depth from the top),
        and the second number is for preventing duplication
        (different layer objects can have same rank)

        Args:
            layer (~chainer.Function_node): Function object
        Returns:
            str: A string to be used for the ``name`` field of the graph
                in the exported Caffe model.

        """
        label = '{}-{}'.format(layer.label, layer.rank)
        d = self.naming_map[label]
        if layer not in d.keys():
            d[layer] = len(d) + 1
        return '{}-{}'.format(label, d[layer])

    def _get_parent_name(self, parent_):
        if parent_ is None:
            return 'data'
        return self._get_layer_name(parent_)

    def _gen_layer_prototxt(self, layer_params, name='layer', depth=0,
                            indent=2):
        if isinstance(layer_params, (dict, collections.OrderedDict)):
            s = name + ' {\n'
            indent_s = ' ' * ((depth + 1) * indent)
            for key, val in layer_params.items():
                s += indent_s + \
                    self._gen_layer_prototxt(val, name=key, depth=depth + 1)
            s += ' ' * (depth * indent)
            s += '}\n'
            return s
        elif isinstance(layer_params, bool):
            return '{}: {}\n'.format(name, 'true' if layer_params else 'false')
        elif isinstance(layer_params, six.integer_types + (float,)):
            return '{}: {}\n'.format(name, layer_params)
        elif isinstance(layer_params, str):
            return '{}: "{}"\n'.format(name, layer_params)
        elif isinstance(layer_params, list):
            s = ''
            indent_s = ' ' * depth * indent
            for i, t in enumerate(layer_params):
                if i != 0:
                    s += indent_s
                s += self._gen_layer_prototxt(t, name=name, depth=depth + 1)
            return s
        else:
            raise ValueError('Unsupported type: ' + str(type(layer_params)))

    def dump_function_object(self, func, prototxt, net):
        assert isinstance(func, _function_types)
        layer_name = self._get_layer_name(func)
        parent_layer_names = [self._get_parent_name(input_.creator)
                              for input_ in func.inputs]
        params = collections.OrderedDict()
        params['type'] = None
        params['name'] = layer_name
        params['bottom'] = parent_layer_names
        params['top'] = [layer_name]
        layer = None
        if net is not None:
            layer = net.layer.add()
        if func.label == 'LinearFunction':
            if len(func.inputs) == 2:
                _, W = func.inputs
                b = None
            else:
                _, W, b = func.inputs
            n_out, n_in = W.shape
            inner_product_param = {
                'num_output': n_out,
                'bias_term': b is not None,
            }
            params['type'] = 'InnerProduct'
            params['inner_product_param'] = inner_product_param
            params['bottom'] = params['bottom'][:1]

            if net is not None:
                for k, v in six.iteritems(inner_product_param):
                    setattr(layer.inner_product_param, k, v)
                _add_blob(layer, list(W.shape), W.data)
                if b is not None:
                    b.retain_data()
                    _add_blob(layer, list(b.shape), b.data)

        elif func.label in ('Convolution2DFunction',
                            'Deconvolution2DFunction'):
            if len(func.inputs) == 2:
                _, W = func.inputs
                b = None
            else:
                _, W, b = func.inputs
            n_out, n_in, kw, kh = W.shape
            convolution_param = {
                'num_output': n_out,
                'bias_term': b is not None,
                'pad_w': func.pw,
                'pad_h': func.ph,
                'stride_w': func.sx,
                'stride_h': func.sy,
                'kernel_w': kw,
                'kernel_h': kh,
                'group': func.groups
            }

            params['bottom'] = params['bottom'][:1]
            if func.label == 'Convolution2DFunction':
                params['type'] = 'Convolution'
            else:
                params['type'] = 'Deconvolution'
                convolution_param['num_output'] = n_in
            params['convolution_param'] = convolution_param

            if net is not None:
                for k, v in six.iteritems(convolution_param):
                    setattr(layer.convolution_param, k, v)

                _add_blob(layer, [n_out, n_in, kh, kw], W.data)

                if b is not None:
                    b.retain_data()
                    _add_blob(layer, [n_out], b.data)

        elif func.label == 'AveragePooling2D':
            kw = func.kw
            kh = func.kh
            pooling_param = {
                'pool': 1,
                'pad_w': func.pw,
                'pad_h': func.ph,
                'stride_w': func.sx,
                'stride_h': func.sy,
                'kernel_w': kw,
                'kernel_h': kh,
            }
            params['type'] = 'Pooling'
            params['pooling_param'] = pooling_param
            if net is not None:
                for k, v in six.iteritems(pooling_param):
                    setattr(layer.pooling_param, k, v)

        elif func.label == 'MaxPoolingND' and func.ndim == 2:
            kh, kw = func.ksize
            sy, sx = func.stride
            ph, pw = func.pad
            pooling_param = {
                'pool': 0,
                'pad_w': pw,
                'pad_h': ph,
                'stride_w': sx,
                'stride_h': sy,
                'kernel_w': kw,
                'kernel_h': kh,
            }
            params['type'] = 'Pooling'
            params['pooling_param'] = pooling_param
            if net is not None:
                for k, v in six.iteritems(pooling_param):
                    setattr(layer.pooling_param, k, v)

        elif func.label == 'LocalResponseNormalization':
            lrn_param = {
                'norm_region': 0,  # ACROSS_CHANNELS
                'local_size': func.n,
                'k': func.k,
                'alpha': func.alpha * func.n,
                'beta': func.beta,
            }
            params['type'] = 'LRN'
            params['lrn_param'] = lrn_param
            if net is not None:
                for k, v in six.iteritems(lrn_param):
                    setattr(layer.lrn_param, k, v)

        elif func.label == 'FixedBatchNormalization':
            _, gamma, beta, mean, var = func.inputs
            batch_norm_param = {'use_global_stats': True, 'eps': func.eps}
            params['type'] = 'BatchNorm'
            params['bottom'] = params['bottom'][:1]
            params['batch_norm_param'] = batch_norm_param
            if net is not None:
                for k, v in six.iteritems(batch_norm_param):
                    setattr(layer.batch_norm_param, k, v)
                _add_blob(layer, [mean.data.size], mean.data)
                _add_blob(layer, [var.data.size], var.data)
                _add_blob(layer, [1], numpy.ones((1,), dtype=numpy.float32))

            if gamma.data is None and beta.data is None:
                pass
            else:
                bn_name = layer_name + '_bn'
                params['name'] = bn_name
                params['top'] = [bn_name]
                if prototxt is not None:
                    prototxt.write(self._gen_layer_prototxt(params))
                if net is not None:
                    layer.name = params['name']
                    layer.type = params['type']
                    layer.bottom[:] = params['bottom']
                    layer.top[:] = params['top']
                    layer.phase = caffe_pb.TEST
                del params, layer
                params = collections.OrderedDict()
                params['type'] = 'Scale'
                params['name'] = layer_name
                params['bottom'] = [bn_name]
                params['top'] = [layer_name]
                if net is not None:
                    layer = net.layer.add()
                beta.retain_data()
                bias_term = beta.data is not None
                scale_param = {
                    'axis': 1,
                    'bias_term': bias_term,
                }
                params['scale_param'] = scale_param
                if net is not None:
                    for k, v in six.iteritems(scale_param):
                        setattr(layer.scale_param, k, v)
                    _add_blob(layer, [gamma.data.size], gamma.data)
                    if bias_term:
                        _add_blob(layer, [beta.data.size], beta.data)

        elif func.label == 'ReLU':
            params['type'] = 'ReLU'

        elif func.label == 'LeakyReLU':
            relu_param = {'negative_slope': func.slope}
            params['type'] = 'ReLU'
            params['relu_param'] = relu_param
            if net is not None:
                for k, v in six.iteritems(relu_param):
                    setattr(layer.relu_param, k, v)

        elif func.label == 'Concat':
            axis = func.axis
            concat_param = {'axis': axis}
            params['type'] = 'Concat'
            params['concat_param'] = concat_param
            if net is not None:
                for k, v in six.iteritems(concat_param):
                    setattr(layer.concat_param, k, v)

        elif func.label == 'Softmax':
            params['type'] = 'Softmax'

        elif func.label == 'Sigmoid':
            params['type'] = 'Sigmoid'

        elif func.label == 'Reshape':
            input_ = func.inputs[0]
            parent = input_.creator
            parent_layer_name = parent_layer_names[0]
            if 'Reshape' in parent_layer_name:
                grandparent = parent.inputs[0].creator
                parent_layer_name = self._get_parent_name(grandparent)
            reshape_param = {'shape': {'dim': list(func.shape)}}
            params['type'] = 'Reshape'
            params['bottom'] = [parent_layer_name]
            params['reshape_param'] = reshape_param
            if layer is not None:
                dim = reshape_param['shape']['dim']
                layer.reshape_param.shape.dim[:] = dim

        elif func.label == '_ + _':
            params['type'] = 'Eltwise'

        else:
            raise Exception(
                'Cannot convert, name={}, rank={}, label={}, inputs={}'.format(
                    layer_name, func.rank, func.label, parent_layer_names))
        if prototxt is not None:
            prototxt.write(self._gen_layer_prototxt(params))

        if net is not None:
            layer.name = params['name']
            layer.type = params['type']
            layer.bottom[:] = params['bottom']
            layer.top[:] = params['top']
            layer.phase = caffe_pb.TEST

    def __call__(self, name, inputs, outputs):
        dumped_list = _dump_graph(outputs)
        f = None
        net = None
        if self.caffemodel is not None:
            net = caffe_pb.NetParameter()
        try:
            if self.prototxt is not None:
                f = open(self.prototxt, 'wt')
                f.write('name: "{}"\n'.format(name))
                assert len(inputs) == 1
                f.write('layer {\n'
                        '  name: "data"\n'
                        '  type: "Input"\n'
                        '  top: "data"\n'
                        '  input_param { shape: {')
                for i in inputs[0].shape:
                    f.write(' dim: ' + str(i))
                f.write(' } }\n'
                        '} \n')
            for i in dumped_list:
                self.dump_function_object(i, f, net)
        finally:
            if f is not None:
                f.close()

        if net is not None:
            with open(self.caffemodel, 'wb') as f:
                f.write(net.SerializeToString())
            if self.debug:
                import google.protobuf.text_format
                with open(self.caffemodel + '.txt', 'w') as f:
                    f.write(google.protobuf.text_format.MessageToString(net))


def export(model, args, directory=None,
           export_params=True, graph_name='Graph'):
    """(Experimental) Export a computational graph as Caffe format.

    Args:
        model (~chainer.Chain): The model object you want to export in Caffe
            format. It should have :meth:`__call__` method because the second
            argument ``args`` is directly given to the model by the ``()``
            accessor.
        args (list of ~chainer.Variable): The arguments which are given to the
            model directly.
        directory (str): The directory used for saving the resulting Caffe
            model. If None, nothing is saved to the disk.
        export_params (bool): If True, this function exports all the parameters
            included in the given model at the same time. If False, the
            exported Caffe model doesn't include any parameter values.
        graph_name (str): A string to be used for the ``name`` field of the
            graph in the exported Caffe model.

    .. note::
        Currently, this function supports networks that created by following
        layer functions.

        - :func:`~chainer.functions.linear`
        - :func:`~chainer.functions.convolution_2d`
        - :func:`~chainer.functions.deconvolution_2d`
        - :func:`~chainer.functions.max_pooling_2d`
        - :func:`~chainer.functions.average_pooling_2d`
        - :func:`~chainer.functions.batch_normalization`
        - :func:`~chainer.functions.local_response_normalization`
        - :func:`~chainer.functions.relu`
        - :func:`~chainer.functions.leaky_relu`
        - :func:`~chainer.functions.concat`
        - :func:`~chainer.functions.softmax`
        - :func:`~chainer.functions.reshape`
        - :func:`~chainer.functions.add`

        This function can export at least following networks.

        - GoogLeNet
        - ResNet
        - VGG

        And, this function use testing (evaluation) mode.

    .. admonition:: Example

       >>> from chainer.exporters import caffe
       >>>
       >>> class Model(chainer.Chain):
       ...    def __init__(self):
       ...        super(Model, self).__init__()
       ...        with self.init_scope():
       ...            self.l1 = L.Convolution2D(None, 1, 1, 1, 0)
       ...            self.b2 = L.BatchNormalization(1)
       ...            self.l3 = L.Linear(None, 1)
       ...
       ...    def __call__(self, x):
       ...        h = F.relu(self.l1(x))
       ...        h = self.b2(h)
       ...        return self.l3(h)
       ...
       >>> x = chainer.Variable(np.zeros((1, 10, 10, 10), np.float32))
       >>> caffe.export(Model(), [x], None, True, 'test')

    """

    assert isinstance(args, (tuple, list))
    if len(args) != 1:
        raise NotImplementedError()
    for i in args:
        assert isinstance(i, variable.Variable)
    with function.force_backprop_mode(), chainer.using_config('train', False):
        output = model(*args)

    if isinstance(output, variable.Variable):
        output = [output]
    assert isinstance(output, (tuple, list))
    for i in output:
        assert isinstance(i, variable.Variable)

    prototxt = None
    caffemodel = None
    if directory is not None:
        prototxt = os.path.join(directory, 'chainer_model.prototxt')
        if export_params:
            caffemodel = os.path.join(directory, 'chainer_model.caffemodel')
    retriever = _RetrieveAsCaffeModel(prototxt, caffemodel)
    retriever(graph_name, args, output)
