# This file includes a derived copy of a part of numpy,
# which is under license bellow.
#
# Copyright (c) 2005-2019, NumPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     * Neither the name of the NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

_released_version = '7.0.0a1'
_tag_name = 'v' + _released_version

_is_released = False


def _local_version():
    # Generate local version using 'git describe' command
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        import os
        import subprocess
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        ).communicate()[0]
        return out

    try:
        import os
        here = os.path.abspath(os.path.dirname(__file__))
        git_dir = os.path.join(here, os.pardir, '.git')
        out = _minimal_ext_cmd(
            ['git', '--git-dir', git_dir, 'describe', '--tags', '--dirty'])
        out = out.decode('ascii')

        if out.startswith(_tag_name + '-'):
            description = out.strip().replace(_tag_name + '-', '')
            return description.replace('-', '.')
        else:
            return 'Unknown'

    except (OSError, ImportError):
        return 'Unknown'


__version__ = _released_version
if not _is_released:
    __version__ += '+' + _local_version()

_optional_dependencies = [
    {
        'name': 'CuPy',
        'packages': [
            'cupy-cuda101',
            'cupy-cuda100',
            'cupy-cuda92',
            'cupy-cuda91',
            'cupy-cuda90',
            'cupy-cuda80',
            'cupy',
        ],
        'specifier': '==7.0.0a1',
        'help': 'https://docs-cupy.chainer.org/en/latest/install.html',
    },
    {
        'name': 'iDeep',
        'packages': [
            'ideep4py',
        ],
        'specifier': '>=2.0.0.post3, <2.1',
        'help': 'https://docs.chainer.org/en/latest/tips.html',
    },
]
