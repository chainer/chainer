import chainerx
from chainerx import _docs


def set_docs():
    Context = chainerx.Context

    _docs.set_doc(
        Context,
        """Context()
An isolated execution environment of ChainerX.

In Python binding, a single context is automatically created and set as the
global default context on import. Only advanced users will have to care about
contexts.
""")
