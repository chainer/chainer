import xchainer


def test_default_graph():
    assert hasattr(xchainer, 'DEFAULT_GRAPH_ID')


def test_anygraph():
    assert hasattr(xchainer, 'anygraph')
