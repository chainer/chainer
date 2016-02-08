

class TimerHook(FunctionHook):

    def preprocess(self, function):
        self.start = time.time()

    def __call__(self, function):
        self.end = time.time()
        print('{}\t{}'.format(function.label, self.end - self.start))
