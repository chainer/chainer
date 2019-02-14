import collections
import weakref

import six


if six.PY3:
    OrderedDict = collections.OrderedDict
else:
    # Reference counting cannot free keys in old `collections.OrderedDict`,
    # where a doubly linked list is used to maintain the order.
    class OrderedDict(object):
        """Dictionary that remembers insertion order

        This class wraps `collections.OrderedDict` to free keys by reference
        counting.
        """

        def __init__(self):
            self.keys = set()
            self.dict = collections.OrderedDict()

        def __contains__(self, key):
            return weakref.ref(key) in self.dict

        def __setitem__(self, key, value):
            self.keys.add(key)
            self.dict[weakref.ref(key)] = value

        def __getitem__(self, key):
            return self.dict[weakref.ref(key)]

        def items(self):
            return [(k(), v) for k, v in self.dict.items()]

        def values(self):
            return self.dict.values()
