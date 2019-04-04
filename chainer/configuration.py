import sys
import threading
import typing as tp  # NOQA

from chainer import types  # NOQA


if types.TYPE_CHECKING:
    import numpy  # NOQA

    from chainer.graph_optimizations import static_graph  # NOQA


class GlobalConfig(object):

    debug = None  # type: bool
    cudnn_deterministic = None  # type: bool
    warn_nondeterministic = None  # type: bool
    enable_backprop = None  # type: bool
    keep_graph_on_report = None  # type: bool
    train = None  # type: bool
    type_check = None  # type: bool
    use_cudnn = None  # type: str
    use_cudnn_tensor_core = None  # type: str
    autotune = None  # type: bool
    schedule_func = None  # type: tp.Optional[static_graph.StaticScheduleFunction] # NOQA
    use_ideep = None  # type: str
    lazy_grad_sum = None  # type: bool
    cudnn_fast_batch_normalization = None  # type: bool
    dtype = None  # type: numpy.dtype
    in_recomputing = None  # type: bool

    """The plain object that represents the global configuration of Chainer."""

    def show(self, file=sys.stdout):
        """show(file=sys.stdout)

        Prints the global config entries.

        The entries are sorted in the lexicographical order of the entry name.

        Args:
            file: Output file-like object.

        """
        keys = sorted(self.__dict__)
        _print_attrs(self, keys, file)


class LocalConfig(object):

    """Thread-local configuration of Chainer.

    This class implements the local configuration. When a value is set to this
    object, the configuration is only updated in the current thread. When a
    user tries to access an attribute and there is no local value, it
    automatically retrieves a value from the global configuration.

    """

    def __init__(self, global_config):
        super(LocalConfig, self).__setattr__('_global', global_config)
        super(LocalConfig, self).__setattr__('_local', threading.local())

    def __delattr__(self, name):
        delattr(self._local, name)

    def __getattr__(self, name):
        dic = self._local.__dict__
        if name in dic:
            return dic[name]
        return getattr(self._global, name)

    def __setattr__(self, name, value):
        setattr(self._local, name, value)

    def show(self, file=sys.stdout):
        """show(file=sys.stdout)

        Prints the config entries.

        The entries are sorted in the lexicographical order of the entry names.

        Args:
            file: Output file-like object.

        .. admonition:: Example

           You can easily print the list of configurations used in
           the current thread.

              >>> chainer.config.show()  # doctest: +SKIP
              debug           False
              enable_backprop True
              train           True
              type_check      True

        """
        keys = sorted(set(self._global.__dict__) | set(self._local.__dict__))
        _print_attrs(self, keys, file)


def _print_attrs(obj, keys, file):
    max_len = max(len(key) for key in keys)
    for key in keys:
        spacer = ' ' * (max_len - len(key))
        file.write(u'{} {}{}\n'.format(key, spacer, getattr(obj, key)))


global_config = GlobalConfig()
'''Global configuration of Chainer.

It is an instance of :class:`chainer.configuration.GlobalConfig`.
See :ref:`configuration` for details.
'''


config = LocalConfig(global_config)
'''Thread-local configuration of Chainer.

It is an instance of :class:`chainer.configuration.LocalConfig`, and is
referring to :data:`~chainer.global_config` as its default configuration.
See :ref:`configuration` for details.
'''


class _ConfigContext(object):

    is_local = False
    old_value = None

    def __init__(self, config, name, value):
        self.config = config
        self.name = name
        self.value = value

    def __enter__(self):
        name = self.name
        value = self.value
        config = self.config
        is_local = hasattr(config._local, name)
        if is_local:
            self.old_value = getattr(config, name)
            self.is_local = is_local

        setattr(config, name, value)

    def __exit__(self, typ, value, traceback):
        if self.is_local:
            setattr(self.config, self.name, self.old_value)
        else:
            delattr(self.config, self.name)


def using_config(name, value, config=config):
    """using_config(name, value, config=chainer.config)

    Context manager to temporarily change the thread-local configuration.

    Args:
        name (str): Name of the configuration to change.
        value: Temporary value of the configuration entry.
        config (~chainer.configuration.LocalConfig): Configuration object.
            Chainer's thread-local configuration is used by default.

    .. seealso::
        :ref:`configuration`

    """
    return _ConfigContext(config, name, value)
