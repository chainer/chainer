import six
from six.moves import urllib
import threading

import numpy
import werkzeug
import werkzeug.serving

from chainer import variable
from chainer import function
from . import _core


server_graph = None


class ErrorResponse(Exception):
    pass


def _do_run_server():
    werkzeug.serving.run_simple('localhost', 6007, graph_app)

def run_server(graph, async=False):
    global server_graph
    server_graph = graph

    if async:
        t = threading.Thread(target=_do_run_server)
        t.start()
    else:
        _do_run_server()

import json

def get_obj(path):
    """
    Returns one of (Graph, Gnode)
    """

    if len(path) == 0:
        path = server_graph.tag

    path_list = path.split('/')

    graph_name = path_list.pop(0)
    graph = server_graph
    if graph_name != graph.tag:
        raise KeyError("Invalid root graph name: {}".format(graph_name))

    while len(path_list) > 0:
        tag = path_list.pop(0)

        # Subgraph?
        graph_ = graph.get_subgraph(tag)
        if graph_ is not None:
            graph = graph_
            continue

        # Node?
        node = graph.get_node(tag)
        if node is not None:
            if len(path_list) > 0:
                raise KeyError("Invalid object path: {}".format(path))
            return node

        raise KeyError("Object not found: {}".format(path))

    return graph


def _ndarray_to_list(data):
    if isinstance(data, numpy.ndarray):
        return data.tolist()
    if isinstance(data, numpy.generic):
        return data.item()
    elif isinstance(data, (tuple,list)):
        return [_ndarray_to_list(_) for _ in data]
    elif isinstance(data, dict):
        return {key: _ndarray_to_list(_) for key,_ in data.items()}
    else:
        assert data is None or isinstance(data, (float,str) + six.integer_types), type(data)
        return data


def api(api_name, path, query, environ):
    method = environ['REQUEST_METHOD']
    if api_name == 'graph' and method == 'GET':
        nodes = []
        edges = []

        if len(path) == 0:
            path = server_graph.tag

        graph = get_obj(path)
        if not isinstance(graph, _core.Graph):
            raise KeyError(
                'No such graph: {}\n'.format(path))

        graph.lock()
        try:
            # TODO: Should not return object id

            for node in graph.nodes:
                if node.is_placeholder:
                    continue

                d_node = {
                    'id': id(node),
                }

                # type
                if node.obj_type in (variable.VariableNode,) + _core._ndarrays:
                    d_node['type'] = 'variable'
                    d_node['shape'] = list(node.shape)
                    d_node['dtype'] = node.dtype.name
                elif node.obj_type is _core.Graph:
                    d_node['type'] = 'subgraph'
                    d_node['path'] = '{}/{}'.format(path, node.tag)
                elif issubclass(node.obj_type, function.Function):
                    d_node['type'] = 'function'
                else:
                    assert False

                # name
                if node.name is not None:
                    d_node['name'] = node.name

                # tag
                if node.tag is not None:
                    d_node['tag'] = node.tag

                # input_index
                try:
                    i = graph.input_nodes.index(node)
                    d_node['input_index'] = i
                except ValueError:
                    pass

                # output_index
                try:
                    i = graph.output_nodes.index(node)
                    d_node['output_index'] = i
                except ValueError:
                    pass

                #
                if node.obj_type is _core.Graph:
                    subgraph = graph.get_subgraph(node.tag)
                    d_node['input_variables'] = [id(_) for _ in subgraph.input_nodes]
                    d_node['output_variables'] = [id(_) for _ in subgraph.output_nodes]

                # data
                if isinstance(node, _core.VariableGnode):
                    summary = node.data_collection.get_summary()
                    if len(summary) > 0:
                        d_node['data_summary'] = summary

                #
                nodes.append(d_node)
            for edge in graph.edges:
                if edge.in_gnode.is_placeholder:
                    continue
                if edge.out_gnode.is_placeholder:
                    continue
                d_edge = {
                    'source': id(edge.in_gnode),
                    'target': id(edge.out_gnode),
                    'arg_index': edge.arg_index,
                    'count': edge.count,
                }

                edges.append(d_edge)
        finally:
            graph.unlock()

        input_nodes = [None if _ is None or _.is_placeholder else id(_)  for _ in graph.input_nodes]
        output_nodes = [None if _ is None or _.is_placeholder else id(_)  for _ in graph.output_nodes]

        data = {
            'tag': graph.tag,
            'path': path,
            'nodes': nodes,
            'edges': edges,
            'input_variables': input_nodes,
            'output_variables': output_nodes,
        }
        json_data = json.dumps(data)
        return 'application/json', json_data

    if api_name == 'data' and method == 'GET':
        required = object()
        def read_query(key, type=str, default=required):
            if key in query:
                return type(query[key][0])
            elif default is required:
                raise ErrorResponse('Required query is missing: {}'.format(key))
            else:
                return default

        """
        data_index:
             None      ... all data
             'current' ... latest data
             int       ... index
        """

        data_name = read_query('name')
        data_index = read_query('index', default=None)
        data_type = read_query('type', default='json')

        # Get data
        try:
            node = get_obj(path)
        except KeyError:
            raise ErrorResponse('Invalid query path: {}'.format(path))

        if data_name not in node.data_collection:
            raise ErrorResponse('Invalid data name: {}'.format(data_name))
        data_series = node.data_collection[data_name]

        if data_index is None:
            # this `epochs` could include `None`
            epochs = data_series.get_iterations()
            epoch_list = []
            data_list = []
            for i, epoch in enumerate(epochs):
                data_index_ = i if epoch is not None else None
                epoch_, data_ = data_series.get_data(data_index_)
                epoch_list.append(epoch_)
                data_list.append(data_)
            data = {
                'epochs': epoch_list,
                'data': data_list,
            }
        else:
            data_index_ = int(data_index) if data_index != 'current' else None

            try:
                _, data = data_series.get_data(data_index_)
            except IndexError as e:
                raise ErrorResponse('Invalid data index: {}'.format(data_index))

        # Encode data
        if data is None:
            raise ErrorResponse('No data is available.')
        elif data_type == 'json':
            response_data = json.dumps(_ndarray_to_list(data))
            content_type = 'application/json'
        elif data_type == 'image':
            assert isinstance(data, numpy.ndarray) and data.ndim == 2
            import matplotlib
            import io
            dpi = 100
            width = 100
            height = 100
            figsize = (width / dpi, height / dpi)
            fig = matplotlib.pyplot.figure(figsize=figsize, frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            ax.pcolormesh(data)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            matplotlib.pyplot.close(fig)
            response_data = buf.getvalue()
            content_type = 'image/png'
        else:
            raise ErrorResponse('Invalid data type: {}'.format(data_type))

        return content_type, response_data

    raise ErrorResponse('Invalid api request: {} method={} path={}'.format(
        api_name, method, path))


def pop_path(path):
    split = path.split('/', 1)
    if len(split) == 1:
        return split[0], ''
    else:
        return split[0], split[1]


def graph_app(environ, start_response):
    status = 200

    path = environ['PATH_INFO']
    path = path[1:] if path.startswith('/') else path
    root_path, path = pop_path(path)
    query = urllib.parse.parse_qs(environ['QUERY_STRING'])

    if root_path == 'api':
        api_name, path = pop_path(path)

        try:
            content_type, data = api(api_name, path, query, environ)
        except ErrorResponse as e:
            content_type = 'application/json'
            data = json.dumps({
                'error': str(e),
            })

        if isinstance(data, str):
            data = data.encode('utf-8')
    else:
        data = b'Error: Invalid request: ' + root_path.encode('utf8')
        content_type = 'text/plain'

    response = werkzeug.wrappers.Response(data)
    response.status_code = status
    response.headers['content-type'] = content_type
    response.headers['access-control-allow-origin'] = '*'
    return response(environ, start_response)
