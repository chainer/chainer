import os
import subprocess

from chainer import computational_graph
from chainer import configuration
from chainer.training import extension
from chainer.utils import argument
from chainer import variable


def is_return_code_zero(args):
    """Return `True` if the return code of the given command
    is zero.

    All the messages sent to stdout or stderr are suppressed.

    Args:
        args (list of str): A command to execute.

    """

    with open(os.devnull, 'wb') as FNULL:
        try:
            subprocess.check_call(args, stdout=FNULL, stderr=FNULL)
        except subprocess.CalledProcessError:
            # The given command returned an error
            return False
        except OSError:
            # The given command was not found
            return False
        return True


def is_graphviz_available():
    """Tell whether graphviz is available or not."""
    return is_return_code_zero(['dot', '-V'])


_var_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}
_func_style = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}


class DumpGraph(extension.Extension):

    """__init__(\
root_name, filename='cg.dot', variable_style=None, function_style=None)

    Trainer extension to dump a computational graph.

    This extension dumps a computational graph. The graph is output in DOT
    language. If graphviz is available, this also renders and saves the image
    of the computational graph.

    It only dumps a graph at the first invocation.

    .. note::
       The computational graph is not kept by default. This
       extension changes this behavior until the first invocation. **It is
       strongly recommended to use it with the default trigger setting.**

       The detailed behavior of this extension is as follows.

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

       With the default setting, the ``DumpGraph`` extension is called at the
       first iteration. Since :class:`~chainer.training.extensions.Evaluator`
       is not called at the first iteration in most cases, it does not cause
       any memory problem.

    Args:
        root_name (str): Name of the root of the computational graph. The
            root variable is retrieved by this name from the observation
            dictionary of the trainer.
        filename (str): Output file name.
            For historical reasons ``out_name`` is also accepted as an alias
            of this argument.
        variable_style (dict): Dot node style for variables. Each variable is
            rendered by an octagon by default.
        function_style (dict): Dot node style for functions. Each function is
            rendered by a rectangular by default.

    .. seealso::
       See :func:`~chainer.computational_graph.build_computational_graph`
       for the ``variable_style`` and ``function_style`` arguments.

    """
    default_name = 'dump_graph'

    def __init__(self, root_name, filename=None,
                 variable_style=None, function_style=None, **kwargs):
        out_name, = argument.parse_kwargs(kwargs, ('out_name', 'cg.dot'))
        if filename is None:
            filename = out_name
        del out_name  # avoid accidental use
        self._root_name = root_name
        self._filename = filename
        if variable_style is None:
            variable_style = _var_style
        self._variable_style = variable_style
        if function_style is None:
            function_style = _func_style
        self._function_style = function_style
        self._original_flag = None
        self._flag_called = False

    def initialize(self, trainer):
        if not self._flag_called:
            self._original_flag = configuration.config.keep_graph_on_report
            configuration.config.keep_graph_on_report = True

    def trigger(self, trainer):
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

            filename = os.path.join(trainer.out, self._filename)
            with open(filename, 'w') as f:
                f.write(cg)
            if is_graphviz_available():
                img_fn = os.path.splitext(self._filename)[0] + '.png'
                image_filename = os.path.join(trainer.out, img_fn)
                subprocess.check_call(
                    ['dot', '-Tpng', filename, '-o', image_filename])
        finally:
            configuration.config.keep_graph_on_report = self._original_flag

    def serialize(self, serializer):
        self._original_flag = serializer('_original_flag', self._original_flag)
        self._flag_called = serializer('_flag_called', self._flag_called)
