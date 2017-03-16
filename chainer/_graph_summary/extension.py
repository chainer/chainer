from chainer.training import extension

from . import _core


class GraphSummary(extension.Extension):
    trigger = lambda a,b: True
    invoke_before_update = True
    invoke_after_update = True

    def __init__(self, graph, keys):
        self.graph = graph
        self.keys = keys
        self._ctx = None
        self.context = _core.GraphContext(graph.tag, graph)

    def __call__(self, trainer):
        if self._ctx is None:
            # Before update
            self._ctx = _core.root_graph([], self.graph, trainer, self.context)
            self._ctx.__enter__()
        else:
            # After update
            outputs = [trainer.observation[_] for _ in self.keys]
            self.context.set_output(outputs)
            self._ctx.__exit__(None, None, None)
            self._ctx = None
