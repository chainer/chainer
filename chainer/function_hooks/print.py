class PrintHook(FunctionHook):

    def __call__(self, function, in_data):
        print('from hook object', function.label)
