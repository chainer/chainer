from __future__ import print_function
import contextlib
import os
import sys
import threading

import six


"""Global and thread-local configuration of Chainer.

Chainer provides some global settings that affect the behavior of some
features. For example, the features with different behaviors on train/test
phases can be configured by setting the ``phase`` configuration.

There are two objects that users mainly deal with: :data:`chainer.config` and
:data:`chainer.global_config`. The ``config`` object configures the thread-
local configuration, while the ``global_config`` object configures the global
configuration shared among all threads.

Each entry of the global configuration is initialized by its default value,
which can be overridden by a value set to the corresponding environment
variable. There is a naming rule of the environment variable: an entry of the
name ``foo_var`` can be configured by the environment variable
``CHAINER_FOO_VAR``.

The following entries are available by default.

- ``debug``
- ``deterministic``
- ``test_mode``
- ``type_check``
- ``use_cudnn``

"""


class GlobalConfig(object):

    """The plain object that represents the global configuration of Chainer."""
    pass


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
        """Prints the config entries.

        The entries are sorted in the lexicographical order of the entry names.

        Args:
            file: Output file like object.

        """
        keys = sorted(set(self._global.__dict__) | set(self._local.__dict__))
        for key in keys:
            print(key, getattr(self, key), sep=':\t', file=file)


global_config = GlobalConfig()
config = LocalConfig(global_config)


@contextlib.contextmanager
def using_config(name, value, config=config):
    """Context manager to temporarily change the thread-local configuration.

    Args:
        name (str): Name of the configuration to change.
        value: Temporary value of the configuration entry.
        config (~chainer.config.LocalConfig): Configuration object. Chainer's
            thread-local configuration is used by default.

    """
    if hasattr(config, name):
        old_value = getattr(config, name)
        setattr(config, name, value)
        yield
        setattr(config, name, old_value)
    else:
        setattr(config, name, value)
        yield
        delattr(config, name)
