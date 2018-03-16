#!/usr/bin/env python

import argparse
import re
import sys
import subprocess


def _git(*args):
    return subprocess.check_output(('git',) + args)


def get_cupy_commit_for(chainer_commit, cupy_branch, chainer_dir, cupy_dir):
    """Returns CuPy commit required for the given Chainer commit."""

    # Retrieve a commit time of the given Chainer commit.
    commit_time = _git(
        '-C', chainer_dir,
        'show', '--format=%ct', chainer_commit, '--')

    # Retrieve a CuPy commit made just before the commit_time, i.e., HEAD
    # at the time of commit_time.
    cupy_commit = _git(
        '-C', cupy_dir,
        'log', '--merges', '--first-parent', '--max-count', '1',
        '--until', commit_time, '--format=%H', cupy_branch, '--')

    return cupy_commit


def get_cupy_release_for(chainer_version):
    """Returns CuPy version required for the given Chainer version."""

    m = re.search(r'^v(\d)\.(.+)$', chainer_version)
    if m is None:
        raise ValueError(chainer_version)

    chainer_major = int(m.group(1))
    chainer_rest = m.group(2)
    if chainer_major <= 1:
        raise ValueError('Chainer v1 or earlier is unsupported')
    elif 2 <= chainer_major <= 3:
        # Chainer vN requires CuPy v(N-1).
        return 'v{}.{}'.format((chainer_major - 1), chainer_rest)
    else:
        # The same versioning as Chainer.
        return chainer_version


def parse_args(args):
    parser = argparse.ArgumentParser()

    # Find CuPy commit from Chainer commit.
    parser.add_argument(
        '--commit', type=str, default=None,
        help='Chainer commit')
    parser.add_argument(
        '--cupy-branch', type=str,
        help='CuPy branch to find commit')
    parser.add_argument(
        '--chainer', type=str, default='chainer',
        help='Chainer source tree (default: chainer)')
    parser.add_argument(
        '--cupy', type=str, default='cupy',
        help='CuPy source tree (default: cupy)')

    # Find CuPy version (tag) from Chainer version.
    parser.add_argument(
        '--release', type=str, default=None,
        help='Chainer release version')

    return parser.parse_args(args)


def main(args):
    params = parse_args(args[1:])
    if params.commit is not None:
        assert params.cupy_branch is not None
        assert params.release is None
        print(get_cupy_commit_for(
            params.commit, params.cupy_branch, params.chainer, params.cupy))
    elif params.release is not None:
        print(get_cupy_release_for(params.release))
    else:
        print('either --commit nor --release must be specified')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
