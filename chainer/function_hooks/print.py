

class PrintHook(FunctionHook):

    def __call__(self, function):
        print('from hook object', function.label)
