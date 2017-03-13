from __future__ import print_function
import contextlib
import sys
import threading


class GlobalConfig(object):

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
        if hasattr(self._local, name):
            return getattr(self._local, name)
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
        print(u'{} {}{}'.format(key, spacer, getattr(obj, key)), file=file)


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


@contextlib.contextmanager
def using_config(name, value, config=config):
    """using_config(name, value, config=chainer.config)

    Context manager to temporarily change the thread-local configuration.

    Args:
        name (str): Name of the configuration to change.
        value: Temporary value of the configuration entry.
        config (~chainer.configuration.LocalConfig): Configuration object.
            Chainer's thread-local configuration is used by default.

    """
    if hasattr(config._local, name):
        old_value = getattr(config, name)
        setattr(config, name, value)
        try:
            yield
        finally:
            setattr(config, name, old_value)
    else:
        setattr(config, name, value)
        try:
            yield
        finally:
            delattr(config, name)
