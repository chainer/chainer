import inspect
import os
import types

import chainer.functions
import chainer.links


def _is_rst_exists(entity):
    return os.path.exists('source/reference/generated/{}.rst'.format(entity))


def check(app, exception):
    missing_entities = []

    missing_entities += [
        name for name in _list_chainer_functions()
        if not _is_rst_exists(name)]

    missing_entities += [
        name for name in _list_chainer_links()
        if not _is_rst_exists(name)]

    if missing_entities:
        app.warn('\n'.join([
            'Undocumented entities found.',
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
