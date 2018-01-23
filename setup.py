#!/usr/bin/env python

import imp
import os
import pkg_resources
import sys

from setuptools import setup


if sys.version_info[:3] == (3, 5, 0):
    if not int(os.getenv('CHAINER_PYTHON_350_FORCE', '0')):
        msg = """
Chainer does not work with Python 3.5.0.

We strongly recommend to use another version of Python.
If you want to use Chainer with Python 3.5.0 at your own risk,
set CHAINER_PYTHON_350_FORCE environment variable to 1."""
        print(msg)
        sys.exit(1)


requirements = {
    'install': [
        'filelock',
        'numpy>=1.9.0',
        'protobuf>=3.0.0',
        'six>=1.9.0',
    ],
    'cuda': [
        'cupy==4.0.0b3',
    ],
    'stylecheck': [
        'hacking',
        'autopep8',
    ],
    'test': [
        'pytest',
        'mock',
    ],
    'doctest': [
        'matplotlib',
        'theano',
    ],
    'docs': [
        'sphinx_rtd_theme',
    ],
    'travis': [
        '-r stylecheck',
        '-r test',
        'pytest-timeout',
        'pytest-cov',
        'theano',
        'h5py',
        'pillow',
    ],
    'appveyor': [
        '-r stylecheck',
        '-r test',
        'pytest-timeout',
        'pytest-cov',
    ],
}


def reduce_requirements(key):
    # Resolve recursive requirements notation (-r)
    reqs = requirements[key]
    resolved_reqs = []
    for req in reqs:
        if req.startswith('-r'):
            depend_key = req[2:].lstrip()
            reduce_requirements(depend_key)
            resolved_reqs += requirements[depend_key]
        else:
            resolved_reqs.append(req)
    requirements[key] = resolved_reqs


for k in requirements.keys():
    reduce_requirements(k)


extras_require = {k: v for k, v in requirements.items() if k != 'install'}


setup_requires = []
install_requires = requirements['install']

cupy_pkg = None
try:
    cupy_pkg = pkg_resources.get_distribution('cupy')
except pkg_resources.DistributionNotFound:
    pass

if cupy_pkg is not None:
    install_requires.append(requirements['cuda'])
    print('Use %s' % requirements['cuda'])

here = os.path.abspath(os.path.dirname(__file__))
__version__ = imp.load_source(
    '_version', os.path.join(here, 'chainer', '_version.py')).__version__


setup(
    name='chainer',
    version=__version__,
    description='A flexible framework of neural networks',
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='https://chainer.org/',
    license='MIT License',
    packages=['chainer',
              'chainer.backends',
              'chainer.dataset',
              'chainer.datasets',
              'chainer.functions',
              'chainer.functions.activation',
              'chainer.functions.array',
              'chainer.functions.connection',
              'chainer.functions.evaluation',
              'chainer.functions.loss',
              'chainer.functions.math',
              'chainer.functions.noise',
              'chainer.functions.normalization',
              'chainer.functions.pooling',
              'chainer.functions.theano',
              'chainer.functions.util',
              'chainer.function_hooks',
              'chainer.iterators',
              'chainer.initializers',
              'chainer.links',
              'chainer.links.activation',
              'chainer.links.caffe',
              'chainer.links.caffe.protobuf3',
              'chainer.links.connection',
              'chainer.links.loss',
              'chainer.links.model',
              'chainer.links.model.vision',
              'chainer.links.normalization',
              'chainer.links.theano',
              'chainer.optimizers',
              'chainer.serializers',
              'chainer.testing',
              'chainer.training',
              'chainer.training.extensions',
              'chainer.training.triggers',
              'chainer.training.updaters',
              'chainer.utils'],
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=requirements['test'],
    extras_require=extras_require,
)
