#!/usr/bin/env python

import os
import pkg_resources
import sys

from setuptools import setup

import chainerx_build_helper


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
        'setuptools',
        'filelock',
        'numpy>=1.9.0',
        'protobuf>=3.0.0',
        'six>=1.9.0',
    ],
    'stylecheck': [
        'autopep8>=1.4.1,<1.5',
        'flake8>=3.6,<3.7',
        'pbr==4.0.4',
        'pycodestyle>=2.4,<2.5',
    ],
    'test': [
        'pytest',
        'mock',
    ],
    'doctest': [
        'sphinx==1.7.9',
        'matplotlib',
        'theano',
    ],
    'docs': [
        'sphinx==1.7.9',
        'sphinx_rtd_theme',
    ],
    'travis': [
        '-r stylecheck',
        '-r test',
        '-r docs',
        # pytest-timeout>=1.3.0 requires pytest>=3.6.
        # TODO(niboshi): Consider upgrading pytest to >=3.6
        'pytest-timeout<1.3.0',
        'pytest-cov',
        'theano',
        'h5py',
        'pillow',
    ],
    'appveyor': [
        '-r test',
        # pytest-timeout>=1.3.0 requires pytest>=3.6.
        # TODO(niboshi): Consider upgrading pytest to >=3.6
        'pytest-timeout<1.3.0',
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
tests_require = requirements['test']


def find_any_distribution(pkgs):
    for pkg in pkgs:
        try:
            return pkg_resources.get_distribution(pkg)
        except pkg_resources.DistributionNotFound:
            pass
    return None


mn_pkg = find_any_distribution(['chainermn'])
if mn_pkg is not None:
    msg = """
We detected that ChainerMN is installed in your environment.
ChainerMN has been integrated to Chainer and no separate installation
is necessary. Please uninstall the old ChainerMN in advance.
"""
    print(msg)
    exit(1)

here = os.path.abspath(os.path.dirname(__file__))
# Get __version__ variable
exec(open(os.path.join(here, 'chainer', '_version.py')).read())


setup_kwargs = dict(
    name='chainer',
    version=__version__,  # NOQA
    description='A flexible framework of neural networks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='https://chainer.org/',
    license='MIT License',
    packages=['chainer',
              'chainer.backends',
              'chainer.dataset',
              'chainer.datasets',
              'chainer.distributions',
              'chainer.exporters',
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
              'chainer.graph_optimizations',
              'chainer.optimizers',
              'chainer.optimizer_hooks',
              'chainer.serializers',
              'chainer.testing',
              'chainer.training',
              'chainer.training.extensions',
              'chainer.training.triggers',
              'chainer.training.updaters',
              'chainer.utils',
              'chainermn',
              'chainermn.communicators',
              'chainermn.datasets',
              'chainermn.extensions',
              'chainermn.functions',
              'chainermn.iterators',
              'chainermn.links'],
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,
)


build_chainerx = 0 != int(os.getenv('CHAINER_BUILD_CHAINERX', '0'))
if os.getenv('READTHEDOCS', None) == 'True':
    # TODO(kmaehashi) build ChainerX documentation on RTD
    build_chainerx = False
    # os.environ['MAKEFLAGS'] = '-j4'
    # build_chainerx = True

chainerx_build_helper.config_setup_kwargs(setup_kwargs, build_chainerx)


setup(**setup_kwargs)
