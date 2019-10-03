def support(opset_versions):
    """Detect lowest supported version of the target converter

    A simple wrap function for convert functions to detect lowest number of
    supported opset version. For example, the target ONNX operater is added
    from 6 and updated on 8, add this function as decorator like the below.

    >>> @support((6, 8))
    >>> def own_converter(func, opset_version, *args):
    >>>     print(opset_version)
    >>>
    >>> own_converter(None, 6)
    6
    >>> own_converter(None, 7)
    6
    >>> own_converter(None, 8)
    8
    >>> own_converter(None, 9)
    8
    >>> own_converter(None, 5)
    RuntimeError: ONNX-Chainer cannot convert ...(snip)

    Arguments:
        opset_versions (tuple): Tuple of opset versions.

    """

    def _wrapper(func):
        def _func_with_lower_opset_version(*args, **kwargs):
            if opset_versions is None:
                return func(*args, **kwargs)
            opset_version = args[1]
            for opver in sorted(opset_versions, reverse=True):
                if opver <= opset_version:
                    break
            if opver > opset_version:
                func_name = args[0].__class__.__name__
                raise RuntimeError(
                    'ONNX-Chainer cannot convert `{}` of Chainer with ONNX '
                    'opset_version {}'.format(
                        func_name, opset_version))
            opset_version = opver
            return func(args[0], opset_version, *args[2:], **kwargs)
        return _func_with_lower_opset_version
    return _wrapper
