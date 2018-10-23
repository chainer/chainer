import os

from chainer import computational_graph
from chainer import configuration
from chainer.training import extension
from chainer import variable


_var_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}
_func_style = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}


class dump_graph(extension.Extension):

    """Trainer extension to dump a computational graph.

    This extension dumps a computational graph. The graph is output in DOT
    language.

    It only dumps a graph at the first invocation.

    .. note::
       As of v2.0.0, the computational graph is not kept by default. This
       extension changes this behavior until the first invocation. **It is
       strongly recommended to use it with the default trigger setting.**

       The detailed behavior of this extension since v2.0.0 is as follows.

       1. In its initializer, it turns on the
          ``chainer.config.keep_graph_on_report`` flag.
       2. At the first iteration, it dumps the graph using the graph held by
          the reported variable.
       3. After dumping the graph, it turns off the flag (if it was originally
          turned off) so that any variable reported afterward does not hold
          a computational graph.

       When the ``keep_graph_on_report`` flag is turned on, the computational
       graph created by the updater is kept during the invocation of
       extensions. It will cause an unnecessarily large memory consumption
       when an extension also uses a large amount of memory, e.g.
       :class:`~chainer.training.extensions.Evaluator`.

       With the default setting, the ``dump_graph`` extension is called at the
       first iteration. Since :class:`~chainer.training.extensions.Evaluator`
       is not called at the first iteration in most cases, it does not cause
       any memory problem.

    Args:
        root_name (str): Name of the root of the computational graph. The
            root variable is retrieved by this name from the observation
            dictionary of the trainer.
        out_name (str): Output file name.
        variable_style (dict): Dot node style for variables. Each variable is
            rendered by an octagon by default.
        function_style (dict): Dot node style for functions. Each function is
            rendered by a rectangular by default.

    .. seealso::
       See :func:`~chainer.computational_graph.build_computational_graph`
       for the ``variable_style`` and ``function_style`` arguments.

    """

    def __init__(self, root_name, out_name='cg.dot',
                 variable_style=None, function_style=None):
        self._root_name = root_name
        self._out_name = out_name
        if variable_style is None:
            variable_style = _var_style
        self._variable_style = variable_style
        if function_style is None:
            function_style = _func_style
        self._function_style = function_style
        self._original_flag = None
        self._flag_called = False

    def initialize(self, _):
        if not self._flag_called:
            self._original_flag = configuration.config.keep_graph_on_report
            configuration.config.keep_graph_on_report = True

    def trigger(self, _):
        if self._flag_called:
            return False
        self._flag_called = True
        return True

    def __call__(self, trainer):
        try:
            var = trainer.observation[self._root_name]
            if not isinstance(var, variable.Variable):
                raise TypeError('root value is not a Variable')
            cg = computational_graph.build_computational_graph(
                [var],
                variable_style=self._variable_style,
                function_style=self._function_style
            ).dump()

            out_path = os.path.join(trainer.out, self._out_name)
            # TODO(beam2d): support outputting images by the dot command
            with open(out_path, 'w') as f:
                f.write(cg)
        finally:
            configuration.config.keep_graph_on_report = self._original_flag

    def serialize(self, serializer):
        self._original_flag = serializer('_original_flag', self._original_flag)
        self._flag_called = serializer('_flag_called', self._flag_called)
