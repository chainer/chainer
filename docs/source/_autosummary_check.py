import collections
import inspect
import types

import chainer.functions
import chainer.links


_autodoc_entities = collections.defaultdict(list)


def process(app, what, name, obj, options, lines):
    _autodoc_entities[what].append(name)


def check(app, exception):
    missing_entities = []

    missing_entities += [
        name for name in _list_chainer_functions()
        if name not in _autodoc_entities['function']]

    missing_entities += [
        name for name in _list_chainer_links()
        if name not in _autodoc_entities['class']]

    if len(missing_entities) != 0:
        app.warn('\n'.join([
            'Undocumented entities found.',
            '(Note: be sure to use `make clean html` to run this check):',
            '',
        ] + missing_entities))


def _list_chainer_functions():
    # List exported functions under chainer.functions.
    return ['chainer.functions.{}'.format(name)
            for (name, func) in chainer.functions.__dict__.items()
            if isinstance(func, types.FunctionType)]


def _list_chainer_links():
    # List exported classes under chainer.links.
    return ['chainer.links.{}'.format(name)
            for (name, link) in chainer.links.__dict__.items()
            if inspect.isclass(link)]
