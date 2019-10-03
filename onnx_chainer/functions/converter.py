class FunctionConverterParams(object):

    def __init__(
            self, func=None, opset_version=None, input_names=None,
            output_names=None, context=None):
        """Wrapper of converter parameters

        Exporter set this parameters to the target converter's argument.

        >>> def own_converter(params):
        >>>     # params is FunctionConverterParams
        >>>     # so enable to get each attributes:
        >>>     func_name = params.func.__class__.__name__

        Arguments:
            func (~chainer.FunctionNode): Target function.
            opset_version (int): Target opset version.
            input_names (list): List of input names.
            output_names (list): List of ouptut names.
            context (~onnx_chainer.context.Context): Context for Exporting
        """

        self.func = func
        self.opset_version = opset_version
        self.input_names = input_names
        self.output_names = output_names
        self.context = context


class FunctionConverter(object):

    def __init__(self, converter):
        """Wrapper of ONNX-Chainer converter

        Exporter set arguments wrapped by ``FunctionConverterParams``, and
        this class breaks downs to each argument.

        Arguments:
            converter (function): The target converter function.
        """

        self.converter = converter

    def __call__(self, params):
        func = params.func
        opset_version = params.opset_version
        input_names = params.input_names
        output_names = params.output_names
        context = params.context
        return self.converter(
            func, opset_version, input_names, output_names, context)
