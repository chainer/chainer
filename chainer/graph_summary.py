import threading

from ._graph_summary._core import Graph  # NOQA
from ._graph_summary._core import graph  # NOQA
from ._graph_summary._core import root_graph  # NOQA
from ._graph_summary.http_server import run_server  # NOQA
from ._graph_summary.extension import GraphSummary  # NOQA


def current():
    current_thread = threading.current_thread()
    context = current_thread.__dict__.get('graph_context', None)
    return context


def set_tag(*args, **kwargs):
    context = current()
    if context is not None:
        context.set_tag(*args, **kwargs)
