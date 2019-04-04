_released_version = '6.0.0rc1'
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
        'specifier': '==6.0.0rc1',
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
