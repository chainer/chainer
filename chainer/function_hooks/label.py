from chainer import function


class LabelHook(function.FunctionHook):

    def __call__(self, function, in_data, out_grad=None):
        print(function.label)
