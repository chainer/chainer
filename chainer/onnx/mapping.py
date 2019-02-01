TENSOR_TYPE_TO_NAME = {
    0: 'UNDEFINED',
    1: 'FLOAT',
    2: 'UINT8',
    3: 'INT8',
    4: 'UINT16',
    5: 'INT16',
    6: 'INT32',
    7: 'INT64',
    8: 'STRING',
    9: 'BOOL',
    10: 'FLOAT16',
    11: 'DOUBLE',
    12: 'UINT32',
    13: 'UINT64',
    14: 'COMPLEX64',
    15: 'COMPLEX128',
}

# Chainer Function -> (ONNX Operator, Operator set IDs)
operators = {
    # Activation
    'ClippedReLU': ('Clip', (1, 6)),
    'ELU': ('Elu', (1, 6)),
    'HardSigmoid': ('HardSigmoid', (1, 6)),
    'LeakyReLU': ('LeakyRelu', (1, 6)),
    'LogSoftmax': ('LogSoftmax', (1,)),
    'PReLUFunction': ('PRelu', (1, 6, 7)),
    'ReLU': ('Relu', (1, 6)),
    'Sigmoid': ('Sigmoid', (1, 6)),
    'Softmax': ('Softmax', (1,)),
    'Softplus': ('Softplus', (1,)),
    'Tanh': ('Tanh', (1, 6)),

    # Array
    'Cast': ('Cast', (1, 6)),
    'Concat': ('Concat', (1, 4)),
    'Copy': ('Identity', (1,)),
    'Depth2Space': ('DepthToSpace', (1,)),
    'Pad': ('Pad', (1, 2)),
    'Reshape': ('Reshape', (1, 5)),
    'Space2Depth': ('SpaceToDepth', (1,)),
    'SplitAxis': ('Split', (1, 2)),
    'Squeeze': ('Squeeze', (1,)),
    'Tile': ('Tile', (1, 6)),
    'Transpose': ('Transpose', (1,)),

    # Connection
    'Convolution2DFunction': ('Conv', (1,)),
    'ConvolutionND': ('Conv', (1,)),
    'Deconvolution2DFunction': ('ConvTranspose', (1,)),
    'DeconvolutionND': ('ConvTranspose', (1,)),
    'EmbedIDFunction': ('Gather', (1,)),
    'LinearFunction': ('Gemm', (1, 6, 7)),

    # Math
    'Add': ('Add', (1, 6, 7)),
    'AddConstant': ('Add', (1, 6, 7)),
    'Absolute': ('Abs', (1, 6)),
    'Div': ('Div', (1, 6, 7)),
    'Mul': ('Mul', (1, 6, 7)),
    'MulConstant': ('Mul', (1, 6, 7)),
    'Neg': ('Neg', (1, 6)),
    'PowVarConst': ('Pow', (1, 7)),
    'Sub': ('Sub', (1, 6, 7)),
    'Clip': ('Clip', (1, 6)),
    'Exp': ('Exp', (1, 6)),
    'Identity': ('Identity', (1,)),
    'MatMul': ('Gemm', (1, 6, 7)),
    'Maximum': ('Max', (1, 6, 8)),
    'Minimum': ('Min', (1, 6, 8)),
    'Sqrt': ('Sqrt', (1, 6)),
    'LinearInterpolate': (None, (1, 6, 7)),
    'LogSumExp': ('ReduceLogSumExp', (1,)),
    'Max': ('ReduceMax', (1,)),
    'Mean': ('ReduceMean', (1,)),
    'Min': ('ReduceMin', (1,)),
    'Prod': ('ReduceProd', (1,)),
    'Sum': ('ReduceSum', (1,)),

    # Noise
    'Dropout': ('Dropout', (1, 6, 7)),

    # Pooling
    'AveragePooling2D': ('AveragePool', (1, 7)),
    'AveragePoolingND': ('AveragePool', (1, 7)),
    'MaxPooling2D': ('MaxPool', (1, 8)),
    'MaxPoolingND': ('MaxPool', (1, 8)),

    # Normalization
    'BatchNormalization': ('BatchNormalization', (1, 6, 7)),
    'FixedBatchNormalization': ('BatchNormalization', (1, 6, 7)),
    'LocalResponseNormalization': ('LRN', (1,)),
    'NormalizeL2': ('LpNormalization', (1,)),

}
