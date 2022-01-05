__version__ = '7.8.1'


_optional_dependencies = [
    {
        'name': 'CuPy',
        'packages': [
            'cupy-cuda120',
            'cupy-cuda116',
            'cupy-cuda115',
            'cupy-cuda114',
            'cupy-cuda113',
            'cupy-cuda112',
            'cupy-cuda111',
            'cupy-cuda110',
            'cupy-cuda102',
            'cupy-cuda101',
            'cupy-cuda100',
            'cupy-cuda92',
            'cupy-cuda91',
            'cupy-cuda90',
            'cupy-cuda80',
            'cupy',
        ],
        'specifier': '>=7.7.0,<8.0.0',
        'help': 'https://docs.cupy.dev/en/latest/install.html',
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
