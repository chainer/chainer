from onnx_chainer import functions
from onnx_chainer.functions.converter import FunctionConverter


_supported_function_node_set = {
    # Activation
    'ClippedReLU',
    'ELU',
    'HardSigmoid',
    'LeakyReLU',
    'LogSoftmax',
    'PReLUFunction',
    'ReLU',
    'Selu',
    'Sigmoid',
    'Softmax',
    'Softplus',
    'Tanh',

    # Array
    'Cast',
    'Concat',
    'Copy',
    'Depth2Space',
    'Dstack',
    'ExpandDims',
    'GetItem',
    'Hstack',
    'Moveaxis',
    'Pad',
    'Repeat',
    'Reshape',
    'ResizeImages',
    'Separate',
    'Shape',
    'Space2Depth',
    'SplitAxis',
    'Squeeze',
    'Stack',
    'Swapaxes',
    'Tile',
    'Transpose',
    'Vstack',
    'Where',

    # Connection
    'Convolution2DFunction',
    'ConvolutionND',
    'Deconvolution2DFunction',
    'DeconvolutionND',
    'EmbedIDFunction',
    'LinearFunction',

    # Loss
    'SoftmaxCrossEntropy',

    # Math
    'Absolute',
    'Add',
    'AddConstant',
    'Arccos',
    'Arcsin',
    'Arctan',
    'ArgMax',
    'ArgMin',
    'BroadcastTo',
    'Clip',
    'Cos',
    'Cosh',
    'Div',
    'DivFromConstant',
    'Exp',
    'Identity',
    'LinearInterpolate',
    'Log',
    'LogSumExp',
    'MatMul',
    'Max',
    'Maximum',
    'Mean',
    'Min',
    'Minimum',
    'Mul',
    'MulConstant',
    'Neg',
    'PowConstVar',
    'PowVarConst',
    'PowVarVar',
    'Prod',
    'RsqrtGPU',
    'Sin',
    'Sinh',
    'Sqrt',
    'Square',
    'Sub',
    'SubFromConstant',
    'Sum',
    'Tan',

    # Noise
    'Dropout',

    # Normalization
    'BatchNormalization',
    'FixedBatchNormalization',
    'GroupNormalization',
    'LocalResponseNormalization',
    'NormalizeL2',

    # Pooling
    'AveragePooling2D',
    'AveragePoolingND',
    'MaxPooling2D',
    'MaxPoolingND',
    'ROIPooling2D',
    'Unpooling2D',
}

_converters = None


def _get_converters():
    global _converters

    if _converters is not None:
        return _converters

    _converters = {
        name: FunctionConverter(getattr(functions, 'convert_'+name, None))
        for name in _supported_function_node_set}
    return _converters


converters = _get_converters()
