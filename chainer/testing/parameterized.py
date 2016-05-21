import itertools
import random
import sys
import unittest

import six


def _gen_case(base, module, i, param):
    cls_name = '%s_param_%d' % (base.__name__, i)

    def __str__(self):
        name = base.__str__(self)
        return '%s  parameter: %s' % (name, param)

    mb = dict(param)

    mb['__str__'] = __str__
    cls = type(cls_name, (base,), mb)
    setattr(module, cls_name, cls)


def _gen_cases(name, base, params):
    module = sys.modules[name]
    for i, param in enumerate(params):
        _gen_case(base, module, i, param)


def parameterize(*params):
    def f(klass):
        assert issubclass(klass, unittest.TestCase)
        _gen_cases(klass.__module__, klass, params)
        # Remove original base class
        return None
    return f


def product(parameter, max_count=-1):
    keys = sorted(parameter)
    values = [parameter[key] for key in keys]
    values_product = itertools.product(*values)
    ret = [dict(zip(keys, vals)) for vals in values_product]
    if max_count == -1 or len(ret) <= max_count:
        return ret
    random.shuffle(ret)
    return ret[:max_count]


def product_dict(*parameters):
    return [
        {k: v for dic in dicts for k, v in six.iteritems(dic)}
        for dicts in itertools.product(*parameters)]
