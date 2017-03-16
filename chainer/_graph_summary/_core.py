import contextlib
import itertools
import sys
import threading
import time
import weakref

import numpy

from chainer import cuda
from chainer import variable
from chainer import function
from chainer.training import util as trigger_util


DEBUG = False

class SpecialObject:
    pass


ROOT_OBJ = SpecialObject()
PLACEHOLDER_OBJ = SpecialObject()


if cuda.available:
    _ndarrays = (numpy.ndarray, cuda.cupy.ndarray)
else:
    _ndarrays = (numpy.ndarray,)


def get_name(obj):
    if isinstance(obj, variable.VariableNode):
        return obj.name
    elif isinstance(obj, _ndarrays):
        return None
    elif isinstance(obj, function.Function):
        return type(obj).__name__
    elif isinstance(obj, (SubgraphObj, Graph)):
        return None
    else:
        assert False


def _get_obj(obj):
    if isinstance(obj, variable.Variable):
        return obj._node
    else:
        return obj


class DataSeriesConfig(object):
    def __init__(self, enable=True, data_reduce=None,
                 preprocess=None, postprocess=None,
                 store_trigger=None, reset_trigger=None):

        if isinstance(data_reduce, DataReduction):
            pass
        elif data_reduce is None or data_reduce == 'overwrite':
            data_reduce = OverwriteReduction()
        elif data_reduce == 'average':
            data_reduce = AverageReduction()
        elif data_reduce == 'mean-std':
            data_reduce = MeanStdReduction()
        elif data_reduce == 'percentile':
            data_reduce = PercentileReduction()
        elif (isinstance(data_reduce, (tuple,list)) and
              len(data_reduce) == 2 and
              callable(data_reduce[0]) and
              callable(data_reduce[1])):
            data_reduce = ReductionByFuncs(
                init_func=data_reduce[0],
                reduce_func=data_reduce[1])
        else:
            raise ValueError("Invalid value for data_reduce.")

        assert isinstance(data_reduce, DataReduction)

        if preprocess is not None:
            assert callable(preprocess)

        if postprocess is not None:
            assert callable(postprocess)

        store_trigger = trigger_util.get_trigger(store_trigger)
        reset_trigger = trigger_util.get_trigger(reset_trigger)

        self.enable = enable
        self.data_reduce = data_reduce
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.store_trigger = store_trigger
        self.reset_trigger = reset_trigger


class GnodeConfig(object):
    def __init__(self, data=None):
        self.data_series_configs = self._make_data_series_configs(data, {})

    def _is_good_config_tuple(self, tup):
        return (len(tup) == 2 and
                isinstance(tup[0], str) and
                isinstance(tup[1], dict))

    def _make_data_series_configs(self, data_spec, configs):
        assert isinstance(configs, dict)

        if isinstance(data_spec, dict):
            data_spec = [('data', data_spec)]

        elif self._is_good_config_tuple(data_spec):
            data_spec = [data_spec]

        if (not isinstance(data_spec, (tuple, list)) or
            not all(len(_) == 2 for _ in data_spec) or
            not all(isinstance(_[0], str) for _ in data_spec) or
            not all(isinstance(_[1], dict) for _ in data_spec)):
            raise ValueError("Invalid data specification.")

        for name, kwargs in data_spec:
            assert isinstance(name, str)
            assert isinstance(kwargs, dict)
            if name not in configs:
                configs[name] = DataSeriesConfig(**kwargs)

        return configs


class DataReduction(object):
    def reset(self):
        pass

    def reduce(self, acc, x, i):
        """Returns new value"""
        pass

    def collect(self, acc, n):
        return acc


class ReductionByFuncs(DataReduction):
    def __init__(self, init_func=None, reduce_func=None):
        assert init_func is not None
        assert reduce_func is not None
        self.init_func = init_func
        self.reduce_func = reduce_func

    def reduce(self, acc, x, i):
        if i == 0:
            return self.init_func(x)
        else:
            return self.reduce_func(acc, x, i)


class OverwriteReduction(DataReduction):
    def reduce(self, acc, x, i):
        return x.copy()


class AverageReduction(DataReduction):
    def reduce(self, acc, x, i):
        if i == 0:
            return x.copy()
        else:
            return acc * (i / float(i+1)) + x / float(i+1)


class MeanStdReduction(DataReduction):
    def __init__(self):
        self._mean = None
        self._mean2 = None

    def reset(self):
        self._mean = None
        self._mean2 = None

    def _std(self):
        return numpy.sqrt(self._mean2 - self._mean * self._mean)

    def reduce(self, acc, x, i):
        if i == 0:
            self._mean = x.copy()
            self._mean2 = x * x
        else:
            self._mean = self._mean * (i / float(i+1)) + x / float(i+1)
            self._mean2 = self._mean2 * (i / float(i+1)) + (x*x) / float(i+1)
        return (self._mean, self._std())


class PercentileReduction(DataReduction):
    def __init__(self, k=10000):
        self._k = k
        self._n_elms = 0

    def reset(self):
        self._n_elms = 0

    def reduce(self, acc, x, i):
        k = self._k
        n_elms = self._n_elms
        reservoir = acc

        if i == 0:
            reservoir = numpy.empty((k,), dtype=x.dtype)

        x_ = x.flat

        # Copy first k elements
        n_copy = max(0, min(x.size, k - n_elms))
        if n_copy > 0:
            reservoir[n_elms:n_elms+n_copy] = x_[0:n_copy]
            n_elms += n_copy

        # Sample remaining elements with probability 1/(n+1)
        #  where n = n_elms, n_elms+1, n_elms+2, ...
        n_sample = x.size - n_copy
        if n_sample > 0:
            j = numpy.random.random((n_sample,)) * numpy.arange(n_elms+1, n_elms+1+n_sample)
            j = j.astype(numpy.int32)
            taken = j < k
            taken_idx = numpy.where(taken)
            reservoir[j[taken_idx]] = x_[taken]
            n_elms += n_sample

        self._n_elms = n_elms

        return reservoir

    def collect(self, acc, n):
        reservoir = acc
        return numpy.percentile(reservoir, numpy.arange(0,110,10))


class DataSeries(object):
    def __init__(self, config):
        self._data_list = []
        self._current_data = None
        self._current_epoch = None
        self._config = config
        self.sample_count = 0
        self._epoch_to_idx = {}

    def get_data(self, index):
        """If index is None, that means the current unfinished data."""

        if index is None:
            # Unfinished data
            epoch = self._current_epoch
            if self._current_data is not None:
                data = self._config.data_reduce.collect(self._current_data, self.sample_count)
                data = self._as_array_recursive(data)
                # Do post-processing on the fly
                if self._config.postprocess:
                    data = self._config.postprocess(data, self.sample_count)
            else:
                data = self._data_list[-1][1]
        else:
            # Finished and stored data
            if not (0 <= index < len(self._data_list)):
                raise IndexError()
            epoch, data = self._data_list[index]
        return epoch, data

    def get_iterations(self):
        epochs = [_[0] for _ in self._data_list]
        # The last `None` represents the latest data
        return epochs + [None]

    def add_sample(self, data, trainer=None):
        assert isinstance(data, _ndarrays)
        config = self._config

        # Preparation
        if config.preprocess:
            data = config.preprocess(data)

        epoch = trainer.updater.epoch_detail

        # Add sample
        assert ((self.sample_count == 0 and self._current_data is None) or
                (self.sample_count > 0 and self._current_data is not None))
        self._current_data = config.data_reduce.reduce(
            self._current_data, data, self.sample_count)
        assert self._current_data is not None
        self._current_epoch = epoch

        self.sample_count += 1

        # Store data
        if config.store_trigger is not None:
            if config.store_trigger(trainer):
                self.store_current(epoch)

        # Reset data
        if config.reset_trigger is not None:
            if config.reset_trigger(trainer):
                self.reset_current()

    def store_current(self, epoch):
        config = self._config
        data = config.data_reduce.collect(self._current_data, self.sample_count)
        data = self._as_array_recursive(data)
        if config.postprocess:
            data = config.postprocess(data, self.sample_count)
        self._data_list.append((epoch, data))

    def reset_current(self):
        self._config.data_reduce.reset()
        self._current_data = None
        self._current_epoch = None
        self.sample_count = 0

    def _as_array_recursive(self, data):
        # TODO: support dict
        if isinstance(data, _ndarrays):
            return self._as_array(data)
        elif isinstance(data, numpy.generic):
            return data
        elif isinstance(data, tuple):
            return tuple([self._as_array_recursive(_) for _ in data])
        elif isinstance(data, list):
            return [self._as_array_recursive(_) for _ in data]
        else:
            assert False, type(data)

    def _as_array(self, data):
        if cuda.available:
            data = cuda.cupy.asnumpy(data)
        assert isinstance(data, numpy.ndarray)
        return data


class DataCollection(object):
    def __init__(self):
        self._data_series_dict = {}

    def __getitem__(self, name):
        return self._data_series_dict[name]

    def __contains__(self, name):
        return name in self._data_series_dict

    def get_names(self):
        return [name for name in self._data_series_dict.keys()
                if self._data_series_dict[name].config.enable]

    def get_summary(self):
        summary = {}
        for name in sorted(self._data_series_dict.keys()):
            data_series = self._data_series_dict[name]
            iter_keys = data_series.get_iterations()
            summary[name] = iter_keys

        return summary


    def add_sample(self, data, trainer=None):
        for name,data_series in self._data_series_dict.items():
            data_series.add_sample(data, trainer)

    def ensure_data_series_prepared(self, name, config):
        assert isinstance(name, str)
        assert isinstance(config, DataSeriesConfig)
        if name not in self._data_series_dict:
            self._data_series_dict[name] = DataSeries(config)


class Gnode(object):
    def __init__(self, obj_type, tag, metatag, name, extra_clue, n_outputs):
        assert tag is None or isinstance(tag, str)

        if obj_type is SpecialObject:
            # Placeholder
            # Only metatag can have a value
            assert tag is not None
            assert name is None
            assert extra_clue is None

            extra_clue = ()
            self.in_edges = None
            self.out_edges = None
            self.kind = None
        else:
            assert isinstance(obj_type, type)

            self.in_edges = set()
            self.out_edges = set()
            self.kind = Gnode._get_kind(obj_type)

        self.tag = tag
        self.metatag = metatag
        self.obj_type = obj_type
        self.extra_clue = extra_clue
        self.n_outputs = n_outputs

        # node_config could either be dict or GnodeConfig.
        self.node_config = None

        # Variable: .name
        # Function: .name
        # Graph: None
        self.name = name

    def set_node_config(self, node_config):
        assert isinstance(node_config, (GnodeConfig, dict))
        if isinstance(node_config, dict):
            assert 'data' not in node_config
        self.node_config = node_config

    @classmethod
    def make_placeholder(cls, metatag=None):
        return cls(SpecialObject, 'PLACEHOLDER', metatag, None, None, None)

    @classmethod
    def make_root(cls):
        return cls(SpecialObject, 'ROOT', None, None, None, None)

    @property
    def is_placeholder(self):
        return self.obj_type is SpecialObject and self.tag == 'PLACEHOLDER'

    @property
    def is_root(self):
        return self.obj_type is SpecialObject and self.tag == 'ROOT'

    @property
    def has_edges(self):
        return ((self.in_edges is not None and len(self.in_edges) > 0) or
                (self.out_edges is not None and len(self.out_edges) > 0))

    @property
    def clue(self):
        return (self.kind, self.tag, self.metatag, self.name) + self.extra_clue

    def __repr__(self):
        if self.is_placeholder:
            return '<Gnode(placeholder) metatag={}>'.format(self.metatag)
        elif self.is_root:
            return '<Gnode(root)>'
        else:
            name = 'Gnode' if self.tag is None else 'Gnode@{}'.format(self.tag)
            return '<{} {:x} in={} out={} type={}>'.format(
                name, id(self), len(self.in_edges), len(self.out_edges), self.obj_type.__name__)

    @classmethod
    def _get_kind(cls, obj_type):
        if obj_type is variable.VariableNode:
            return 'variable'
        if obj_type in _ndarrays:
            return 'variable'
        elif obj_type in (Graph, SubgraphObj):
            return 'subgraph'
        elif issubclass(obj_type, function.Function):
            return 'function'
        else:
            assert False

    @classmethod
    def from_obj(cls, obj, tag, metatag):
        if obj is None:
            # Placeholder gnode
            assert tag is None
            return cls.make_placeholder(metatag)
        elif isinstance(obj, (variable.VariableNode,) + _ndarrays):
            return VariableGnode(obj, tag, metatag)
        elif isinstance(obj, function.Function):
            return FunctionGnode(obj, tag, metatag)
        elif isinstance(obj, SubgraphObj):
            return cls.from_graph(obj.graph, tag, metatag)
        else:
            clue = cls.get_extra_clue(obj, tag)
            return Gnode(
                type(obj), tag, get_name(obj), clue)

    @classmethod
    def get_clue(cls, obj, tag, metatag):
        kind = cls._get_kind(type(obj))
        return (kind, tag, metatag, get_name(obj)) + cls.get_extra_clue(obj, tag)

    @classmethod
    def get_extra_clue(cls, obj, tag):
        if isinstance(obj, (variable.VariableNode,) + _ndarrays):
            extra_clue = VariableGnode.get_extra_clue(obj, tag)
        elif isinstance(obj, function.Function):
            extra_clue = FunctionGnode.get_extra_clue(obj, tag)
        elif isinstance(obj, SubgraphObj):
            extra_clue = cls.get_graph_clue(obj, tag)
        else:
            assert False
        return extra_clue

    def get_compatible_in_gnodes(self, obj, tag, metatag, arg_index):
        if len(self.in_edges) > 0:
            clue = Gnode.get_clue(obj, tag, metatag)
            in_gnodes = []
            for edge in self.in_edges:
                if edge.arg_index != arg_index:
                    continue
                if clue == edge.in_gnode.clue:
                    in_gnodes.append(edge.in_gnode)
            return in_gnodes
        else:
            return []

    def get_out_edge(self, out_gnode):
        for edge in self.out_edges:
            if edge.out_gnode == out_gnode:
                return edge
        return None

    @classmethod
    def from_graph(cls, obj, tag, metatag):
        return Gnode(
            type(obj), tag, metatag, get_name(obj),
            cls.get_graph_clue(obj, tag),
            obj.n_outputs)

    @classmethod
    def get_graph_clue(cls, graph, tag):
        return (graph.tag,)


class VariableGnode(Gnode):
    def __init__(self, var, tag, metatag):
        assert isinstance(var, (variable.VariableNode,) + _ndarrays)
        name = var.name if isinstance(var, variable.VariableNode) else None
        extra_clue = VariableGnode.get_extra_clue(var, tag)
        super(VariableGnode, self).__init__(type(var), tag, metatag, name, extra_clue, None)

        self.shape = var.shape
        self.dtype = var.dtype
        self.name = name

        self.data_collection = DataCollection()

    @classmethod
    def get_extra_clue(cls, var, tag):
        assert isinstance(var, (variable.VariableNode,) + _ndarrays)
        return (var.shape, var.dtype, var.name if isinstance(var, (variable.Variable, variable.VariableNode)) else None)

    def __repr__(self):
        name = self.__class__.__name__
        if self.tag is not None:
            name += '@' + self.tag
        lst = [
            name,
            ('\'' + self.name + '\'') if self.name else None,
            '{:x}'.format(id(self)), self.shape, self.dtype,
            'clue={}'.format(self.clue),
        ]
        return '<{}>'.format(
            ' '.join(str(_) for _ in lst if _ is not None))

    def _prepare_data_collection(self, data_series_configs):
        assert isinstance(data_series_configs, dict)
        for name, config in data_series_configs.items():
            if config.enable:
                self.data_collection.ensure_data_series_prepared(name, config)

    def add_data_sample(self, data, trainer):
        assert isinstance(data, _ndarrays)
        node_config = self.node_config
        if node_config is None:
            return

        self._prepare_data_collection(node_config.data_series_configs)
        self.data_collection.add_sample(data, trainer)


class FunctionGnode(Gnode):
    def __init__(self, func, tag, metatag):
        extra_clue = FunctionGnode.get_extra_clue(func, tag)
        super(FunctionGnode, self).__init__(type(func), tag, metatag, type(func).__name__, extra_clue, len(func.outputs))

    @classmethod
    def get_extra_clue(cls, obj, tag):
        assert isinstance(obj, function.Function)
        inputs = obj.inputs
        outputs = [_() for _ in obj.outputs]
        assert all(_ is not None for _ in inputs)
        assert all(_ is not None for _ in outputs)
        assert all(isinstance(_, variable.VariableNode) for _ in inputs)
        assert all(isinstance(_, variable.VariableNode) for _ in outputs)
        return (
            tuple([(_.shape, _.dtype, _.name) for _ in inputs]),
            tuple([(_.shape, _.dtype, _.name) for _ in outputs]),
        )

    def get_out_edges(self, arg_index):
        return [edge for edge in self.out_edges if edge.arg_index == arg_index]

    def get_in_edges(self, arg_index):
        return [edge for edge in self.in_edges if edge.arg_index == arg_index]


class GraphEdge(object):
    def __init__(self, in_gnode, out_gnode, arg_index):
        """
        arg_index: The index of the argument to which the edge is connected to/from.
        """
        assert in_gnode is None or isinstance(in_gnode, Gnode), type(in_gnode)
        assert isinstance(out_gnode, Gnode)
        assert isinstance(arg_index, int)
        self.in_gnode = in_gnode
        self.out_gnode = out_gnode
        self.hash_key = (in_gnode, out_gnode, arg_index)

        self.count = 0
        self.arg_index = arg_index

        self.data_sum = None

    def __hash__(self):
        return hash(self.hash_key)

    def __eq__(self, other):
        return self.hash_key == other.hash_key

    def __repr__(self):
        return '<GraphEdge {:x} i={} from={} to={}>'.format(
            id(self), self.arg_index, self.in_gnode, self.out_gnode)


class Graph(object):
    def __init__(self, tag, inherited_node_configs=None):
        self.tag = tag
        self.nodes = set()
        self.node_map = {} # tag -> node
        self.subgraphs = {}
        self.subgraph_inout_gnodes = {} # tag -> (input nodes, output nodes)

        self.subgraph_output_tag_map = {} # output variable tag -> (subgraph tag, arg_index)

        self.input_nodes = None
        self.output_nodes = None
        self._lock = threading.Lock()

        self._inherited_node_configs = inherited_node_configs
        self._node_configs = None

        self._last_context = None

        self.root_gnode = Gnode.make_root()

    def lock(self):
        self._lock.acquire()

    def unlock(self):
        self._lock.release()

    @property
    def n_outputs(self):
        return len(self.output_nodes)

    def config_node(self, tag_path, **kwargs):
        tag_path = tag_path.split('/')

        # Make hierarchical config
        d = self._node_configs
        if d is None:
            d = self._node_configs = {}

        for tag in tag_path[:-1]:
            subd = d.get(tag)
            if subd is None:
                subd = d[tag] = {}
            d = subd

        d[tag_path[-1]] = GnodeConfig(**kwargs)

    def get_node_config(self, tag):
        """Returns a GnodeConfig (leaf) or a dict (subgraph) or None (not found)"""

        if self._node_configs is not None:
            config = self._node_configs.get(tag)
            if config is not None:
                return config

        if self._inherited_node_configs is not None:
            config = self._inherited_node_configs.get(tag)
            if config is not None:
                return config

        return None

    @property
    def is_empty(self):
        return len(self.nodes) == 0

    @property
    def has_edges(self):
        """Test if the graph has any edges.

        In most cases this is as fast as N(1).
        """
        return any(node.has_edges for node in self.nodes)

    @property
    def edges(self):
        edges = set()
        for node in self.nodes:
            if node.in_edges:
                edges.update(node.in_edges)
            if node.out_edges:
                edges.update(node.out_edges)
        return edges

    def set_tag(self, node, tag):
        assert tag not in self.node_map
        assert node in self.nodes
        self.node_map[tag] = node

    def find_node(self, tag):
        assert isinstance(tag, str)
        return self.node_map.get(tag)

    def submit_outputs(self, output_variables):
        assert any(_ is not None for _ in output_variables)
        self.debug('submit_outputs: {}'.format([type(_) for _ in output_variables]))

        if self.output_nodes is None:
            self.output_nodes = [None] * len(output_variables)

        for i,(obj, tag, metatag) in enumerate(output_variables):
            gnode = self.output_nodes[i]
            if gnode is not None and not gnode.is_placeholder:
                if gnode.tag != tag:
                    raise RuntimeError("Inconsistent tag of output variable ({} -> {}) between passes".format(gnode.tag, tag))
                continue

            obj_ = _get_obj(obj)
            gnode = None if tag is None else self.find_node(tag)
            if gnode is None:
                gnode = Gnode.from_obj(obj_, tag, metatag)
                self._add_node(gnode)

            self.output_nodes[i] = gnode

        assert len(self.output_nodes) == len(output_variables)

    def submit_inputs(self, input_tuples):
        if self.input_nodes is None:
            self.input_nodes = [None] * len(input_tuples)

        for i,(obj,tag,metatag) in enumerate(input_tuples):
            if tag is not None and self.find_node(tag) is not None:
                continue
            if (self.input_nodes[i] is not None and
                not self.input_nodes[i].is_placeholder):
                continue

            obj_ = _get_obj(obj)
            gnode = Gnode.from_obj(obj_, tag, metatag)
            self._add_node(gnode)
            self.input_nodes[i] = gnode
            self.debug('input gnode: {}'.format(gnode))

        assert len(self.input_nodes) == len(input_tuples)

    def get_node(self, tag):
        return self.node_map.get(tag)

    def get_subgraph(self, tag, create=False):
        assert isinstance(tag, str)
        subgraph = self.subgraphs.get(tag)
        if subgraph is None:
            if not create:
                return None

            subconfigs = self.get_node_config(tag)
            subgraph = Graph(tag, subconfigs)
            self.subgraphs[tag] = subgraph
        return subgraph

    def set_subgraph(self, tag, subgraph):
        assert tag not in self.subgraphs
        self.subgraphs[tag] = subgraph

    def create_subgraph_node(self, tag):
        subgraph = self.subgraphs[tag]
        node = self.node_map.get(tag)
        if node is None:
            node = self._add_graph_node(subgraph, tag, None)
        return node

    def get_compatible_nodes(self, obj, tag, metatag):
        clue = Gnode.get_clue(obj, tag, metatag)
        nodes = []
        for node in self.nodes:
            if node.clue == clue:
                nodes.append(node)
        return nodes

    def _add_graph_node(self, obj, tag, metatag):
        assert isinstance(obj, Graph)
        assert isinstance(tag, str)
        node = self.node_map.get(tag, None)
        if node is None:
            node = Gnode.from_graph(obj, tag, metatag)
            self._add_node(node)
        return node

    def _add_node(self, node):
        if node not in self.nodes:
            if not node.is_placeholder:
                if node.tag is not None:
                    if node.tag in self.node_map:
                        raise RuntimeError('Duplicate tag is detected (\'{}\')'.format(node.tag))

                    # Store node config if any
                    node_config = self.get_node_config(node.tag)
                    if node_config is not None:
                        node.set_node_config(node_config)

                self.node_map[node.tag] = node

            self.nodes.add(node)
            self.debug("{}: Gnode added: {}".format(self.tag, node))

    def add_node(self, obj, tag):
        gnode = self.node_map.get(tag)
        if gnode is None:
            gnode = Gnode.from_obj(obj, tag)
            self._add_node(gnode)
        return gnode

    def assert_consistent(self):
        # Test
        for tag,node in self.node_map.items():
            assert node in self.nodes

        for node in self.nodes:
            if not node.is_placeholder and node.tag is not None:
                assert self.node_map[node.tag] == node

        # Test that node tags do not conflict among nodes
        node_tags = [node.tag for node in self.nodes if not node.is_placeholder and node.tag is not None]
        assert len(node_tags) == len(set(node_tags))

        # No node should be isolated
        for node in self.nodes:
            assert (node.is_placeholder or
                    len(node.in_edges) > 0 or len(node.out_edges) > 0), node

        # Test node connections
        for node in self.nodes:
            if node.is_placeholder:
                continue
            for in_edge in node.in_edges:
                assert in_edge.out_gnode == node
                in_gnode = in_edge.in_gnode
                if in_gnode is not None:
                    assert in_gnode in self.nodes
                    assert any(_.out_gnode == node for _ in in_gnode.out_edges)

            for out_edge in node.out_edges:
                assert out_edge.in_gnode == node

                out_gnode = out_edge.out_gnode
                if out_gnode is not None:
                    assert out_gnode in self.nodes
                    assert any(_.in_gnode == node for _ in out_gnode.in_edges)

    def debug(self, s):
        if DEBUG:
            print('{}: {}'.format(self.tag, s))

    def debug_dump(self, out=None):
        if out is None:
            out = sys.stderr

        def putline(s):
            out.write(s)
            out.write('\n')

        def debug(s):
            self.debug("  " + s)

        debug("---")
        putline('digraph graphname {rankdir=TB;')
        visited_nodes = set()
        written_nodes = set()

        nodes = set([(_, 0) for _ in self.nodes if len(_.out_edges) == 0])
        while len(nodes) > 0:
            node, depth = nodes.pop()
            print_prefix = "Dump {}".format("  " * depth)
            debug("{}{}".format(print_prefix, node))
            if id(node) in visited_nodes:
                continue
            visited_nodes.add(id(node))


            if id(node) not in written_nodes:
                if node.obj_type is variable.VariableNode:
                    shape = 'oval'
                    label = str(node)
                elif node.obj_type in _ndarrays:
                    shape = 'oval'
                    label = 'ndarray'
                elif issubclass(node.obj_type, function.Function):
                    shape = 'box'
                    label = "{}\\n{}".format(node, node.clue)
                elif node.obj_type is Graph:
                    shape = 'doubleoctagon'
                    label = str(node)
                else:
                    assert False
                putline('{} [label="{}", shape="{}"];'.format(
                    id(node),
                    label, shape))
                written_nodes.add(id(node))

            debug("{}In edges: {}".format(print_prefix, node.in_edges))

            for in_edge in sorted(node.in_edges, key=lambda edge: edge.arg_index):
                debug("{}In edge: {}".format(print_prefix, in_edge))
                debug("{}In edge@: {}".format(print_prefix, in_edge.in_gnode.out_edges))
                in_gnode = in_edge.in_gnode
                assert in_gnode is not None

                if id(in_gnode) not in visited_nodes:
                    nodes.add((in_gnode, depth+1))

                putline('{} -> {} [label="i={} n={}"];'.format(
                    id(in_gnode), id(node), in_edge.arg_index, in_edge.count))

        putline('}')


class SubgraphObj:
    def __init__(self, graph, input_variables, output_variables, metatag):
        self.graph = graph
        self.input_objs = [_get_obj(_) for _ in input_variables]
        self.output_objs = output_variables
        self.tag = graph.tag
        self.metatag = metatag


class MatchConfig:
    def __init__(self, can_create_edge=False, can_create_node=False, max_create=0):
        self.can_create_edge = can_create_edge
        self.can_create_node = can_create_node
        self.max_create = max_create


class MatchNode:
    def __init__(self, obj, gnode, prev_mnode, arg_index):
        assert gnode is None or isinstance(gnode, Gnode)
        assert prev_mnode is None or isinstance(prev_mnode, MatchNode)
        self.obj = obj
        self.gnode = gnode
        self.prev_mnode = prev_mnode
        self.arg_index = arg_index

        self.submitted = False

    def __repr__(self):
        return '<MatchNode gnode={}>'.format(self.gnode)

    @property
    def out_edge(self):
        if self.prev_mnode is None:
            return None
        if self.gnode is None:
            return None
        prev_gnode = self.prev_mnode.gnode
        out_edge = self.gnode.get_out_edge(prev_gnode)
        return out_edge


class MatchStateChain:
    """Keeps track of created nodes/edges"""

    def __init__(self, search, prev_chain, edge, created_gnode, created_obj=None):
        assert isinstance(search, MatchSearch)
        assert prev_chain is None or isinstance(prev_chain, MatchStateChain)
        assert isinstance(edge, tuple)
        assert len(edge) == 3
        assert isinstance(created_gnode, bool)
        if created_gnode:
            assert created_obj is not None

        if prev_chain is not None:
            assert prev_chain.next_chain is None

        self.search = search
        self.created_gnode = created_gnode
        self.prev_chain = prev_chain
        self.next_chain = None
        self.edge = edge
        self.abandoned = False
        self.created_gnodes = created_gnode
        self.created_obj = created_obj
        self.visited_objs = []
        if prev_chain is not None:
            prev_chain.next_chain = self
            self.n_created_gnodes = prev_chain.n_created_gnodes
            self.n_created_edges = prev_chain.n_created_edges + 1
        else:
            self.n_created_gnodes = 0
            self.n_created_edges = 1

        if self.created_gnode:
            self.n_created_gnodes += 1

        self.search.obj_to_gnode[id(_get_obj(created_obj))] = edge[1] # gnode
        self.search.push_edge(self.edge)
        if self.created_gnode:
            self.search.push_created_gnode(self.edge[0])

        self.search.debug("New state: {}".format(self))

    def __repr__(self):
        return '<MatchStateChain depth={} created_gnode={}, edge={}>'.format(
            self.depth,
            self.created_gnode,
            self.edge)

    @property
    def depth(self):
        if self.prev_chain is None:
            return 0
        else:
            return self.prev_chain.depth + 1

    def visit(self, gnode):
        self.visited_objs.append(gnode)
        self.search.visit(gnode)

    def abandon(self):
        if self.abandoned:
            return

        self.search.debug("Abandon: {}".format(self))
        if self.next_chain is not None:
            self.next_chain.abandon()
            self.next_chain = None

        self.search.pop_edge(self.edge)
        if self.created_gnode:
            self.search.pop_created_gnode(self.edge[0])
            del self.search.obj_to_gnode[id(_get_obj(self.created_obj))]

        for obj in self.visited_objs:
            self.search.unvisit(obj)

        if self.prev_chain is not None:
            assert self.prev_chain.next_chain is self
            self.prev_chain.next_chain = None

        self.abandoned = True


class MatchStackFrame:
    """Represents the graph traversal"""

    """Just for tracking back?"""

    def __init__(self, search, prev_frame, gnode, arg_index, obj):
        self.search = search
        self.prev_frame = prev_frame
        self.gnode = gnode
        self.arg_index = arg_index
        if prev_frame is None:
            self.depth = 0
        else:
            assert prev_frame.good
            self.depth = prev_frame.depth + 1

        self.obj = obj
        self.good = True

    @property
    def prev_mnode(self):
        if self.prev_frame is None:
            return None
        else:
            return self.prev_frame.mnode


class MatchSolution:
    def __init__(self, input_nodes, floating_nodes, state_frame):
        self.input_nodes = input_nodes or []
        self.floating_nodes = floating_nodes or []
        self.state_frame = state_frame

    @classmethod
    def incr(cls, base_sol, input_nodes, floating_nodes, state_frame):
        return MatchSolution(
            base_sol.input_nodes + input_nodes,
            base_sol.floating_nodes + floating_nodes,
            state_frame)

    def __repr__(self):
        return '<MatchSolution {} {}>'.format(
            len(self.input_nodes), len(self.floating_nodes))

    def empty(self):
        return len(self.input_nodes) == 0 and len(self.floating_nodes) == 0

    def leaf_nodes(self):
        return set(self.input_nodes + self.floating_nodes)


class MatchSearch:
    def __init__(self, config, context):
        self.config = config
        self.context = context

        self.head_frame = None
        self.head_state = None

        # (Gnode, bool_in) -> (Gnode_in, Gnode_out, arg_index)
        self.edges = {}

        self.created_gnodes = set()
        self.visited_objs = set()
        self.obj_to_gnode = {}

    def is_edge_filled(self, gnode, is_input, arg_index):
        s = self.edges.get((gnode, is_input))
        if s:
            return any(_[2] == arg_index for _ in s)

    def is_out_edges_filled(self, gnode):
        n_outputs = gnode.n_outputs
        if n_outputs is None or n_outputs == 0:
            return True
        else:
            each_arg_filled = [False] * gnode.n_outputs
            d = self.edges.get((gnode, True))
            if d is None:
                return True
            for edge,count in d.items():
                each_arg_filled[edge[2]] = True

            return all(each_arg_filled)

    def is_all_gnodes_filled(self):
        # Check output of every gnodes are filled
        gnodes = set(_[0] for _ in self.edges.keys() if _[1] == False)
        return all(self.is_out_edges_filled(_) for _ in  gnodes)

    def is_visited(self, gnode):
        return gnode in self.visited_objs

    def visit(self, gnode):
        assert not self.is_visited(gnode)
        self.visited_objs.add(gnode)

    def unvisit(self, gnode):
        self.visited_objs.remove(gnode)

    def push_edge(self, tup):
        in_gnode, out_gnode, arg_index = tup

        self.debug("push_edge: {}".format(tup))

        def add_edge(key, tup):
            d_ = self.edges.get(key)
            if d_ is None:
                d_ = self.edges[key] = {}
            if tup not in d_:
                d_[tup] = 1
            else:
                d_[tup] += 1

        add_edge((in_gnode, True), tup)
        add_edge((out_gnode, False), tup)

    def pop_edge(self, tup):
        in_gnode, out_gnode, arg_index = tup
        # Remove edges.
        # If a node attached to the edge is one of the created node,
        # and it is no longer used, remove it from the created nodes.

        d_ = self.edges[(in_gnode, True)]
        d_[tup] -= 1
        if d_[tup] == 0:
            del d_[tup]

        d_ = self.edges[(out_gnode, False)]
        d_[tup] -= 1
        if d_[tup] == 0:
            del d_[tup]

    def push_created_gnode(self, gnode):
        self.debug("Push created gnode: {}".format(gnode))
        assert gnode not in self.created_gnodes
        self.created_gnodes.add(gnode)

    def pop_created_gnode(self, gnode):
        self.debug("Pop created gnode: {}".format(gnode))
        self.created_gnodes.remove(gnode)

    def run(self):
        frame = MatchStackFrame(self, None, self.context.graph.root_gnode, None, ROOT_OBJ)
        sol_ = MatchSolution([], [], frame)
        state = None
        try:
            sol, st = next(self._find_matches_partial_tree(sol_, None, frame, state))
        except StopIteration:
            sol, st = None, None

        if sol:
            # Solution found
            self.debug("Solution found: {}".format(sol))
            self.debug("  input_nodes: {}".format(sol.input_nodes))
            self.debug("  floating_nodes: {}".format(sol.floating_nodes))
            return sol

        else:
            # No solution
            return None

    def debug(self, s):
        if DEBUG:
            print("{} {}: {}".format(self.context.tag, len(self.created_gnodes), s))

    def _find_matches_partial_tree(self, prev_sol, prev_mnode, frame, state):
        """ Yields solution
        """

        assert state is None or isinstance(state, MatchStateChain)

        def debug(s):
            self.debug("{}{}".format("        " * frame.depth, s))

        config = self.config
        context = self.context

        obj = frame.obj
        gnode = frame.gnode
        arg_index = frame.arg_index

        if not self.is_out_edges_filled(gnode):
            debug("Not filled yet. returning")
            yield prev_sol, state
            return

        #if self.is_visited(gnode):
        #    debug("Gnode is already visited, returning")
        #    yield prev_sol, state
        #    return

        #if state is not None:
        #    state.visit(gnode)

        debug("Obj = {!r}".format(obj))
        debug("gnode = {}".format(gnode))

        assert isinstance(gnode, Gnode)
        assert prev_mnode is None or isinstance(prev_mnode, MatchNode)

        if prev_mnode is not None:
            if (config.max_create is not None and
                state is not None and
                state.n_created_gnodes > config.max_create):
                # Failure: Maximum created edges/nodes reached
                debug("Exceed: {} + {} > {}".format(
                    state.n_created_edges, state.n_created_gnodes, config.max_create))
                return

            if self.context._is_input_variable(obj):
                # Success: Reached an input variable
                mnode = MatchNode(obj, gnode, prev_mnode, arg_index)
                debug("Reached input variable: {}".format(obj.shape))
                yield MatchSolution(prev_sol.input_nodes + [mnode], prev_sol.floating_nodes, frame), state
                return

            if id(obj) in context.last_output_variables:
                # Success: Reached an output variable of last pass
                mnode = MatchNode(obj, gnode, prev_mnode, arg_index)
                debug("Reached last output variable: {}".format(id(obj)))
                yield MatchSolution(prev_sol.input_nodes + [mnode], prev_sol.floating_nodes, frame), state
                return

            if ((isinstance(obj, variable.VariableNode) and obj.creator is None) or
                isinstance(obj, _ndarrays)):
                # Success: Reached a root variable
                mnode = MatchNode(obj, gnode, prev_mnode, arg_index)
                debug("Reached floating variable: {}".format(obj.shape))
                yield MatchSolution(prev_sol.input_nodes, prev_sol.floating_nodes + [mnode], frame), state
                return

        # From here, we need recursive search

        mnode = MatchNode(obj, gnode, prev_mnode, arg_index)
        debug('** Examining node obj: {}'.format(type(obj)))

        if obj is ROOT_OBJ:
            # Root node

            assert len(context.output_variables) == len(context.graph.output_nodes)
            inputs = [_get_obj(_[0]) for _ in context.output_variables]
            arg_indices = range(len(inputs))

        elif isinstance(obj, variable.VariableNode):
            func = obj.creator

            subgraph_tuple = context.subgraph_output_map.get(id(obj))
            if subgraph_tuple is not None:
                # obj is an output variable of a subgraph
                subgraph, index = subgraph_tuple
                inputs = [subgraph]
                arg_indices = [index]

            else:
                # obj is a non-root variable
                inputs = [func]
                arg_indices = [i for i,_ in enumerate(func.outputs) if _() is obj]

        elif isinstance(obj, function.Function):
            # obj is a function
            inputs = obj.inputs
            arg_indices = list(range(len(inputs)))

        elif isinstance(obj, SubgraphObj):
            # obj is a subgraph
            subgraph_tag = obj.tag
            assert isinstance(subgraph_tag, str)
            inputs = obj.input_objs
            debug("AAA: {}".format(self.context.graph.input_nodes))

            arg_indices = list(range(len(inputs)))
        else:
            assert False

        assert all(_ is None or isinstance(_, (variable.VariableNode, function.Function, SubgraphObj) + _ndarrays) for _ in inputs)
        assert len(arg_indices) == len(inputs)

        # Find solutions for each input argument
        debug("Input args: {}".format(len(arg_indices)))

        def recurse_next_input(prev_gen, in_obj, in_arg_index):
            for sol_, st_ in prev_gen:
                if sol_ is None: continue

                next_solution_gen = self._recurse_single_input(
                    mnode, frame, sol_, in_obj, in_arg_index, st_)

                for sol, st  in next_solution_gen:
                    yield sol, st

        # solution, state
        gen = [(prev_sol, state)]

        for in_obj, in_arg_index in zip(inputs, arg_indices):
            gen = recurse_next_input(gen, in_obj, in_arg_index)

        for sol, st in gen:
            yield sol, st

    def get_compatible_in_gnodes(self, gnode, in_obj, in_tag, in_metatag, in_arg_index):
        in_gnode = self.obj_to_gnode.get(id(_get_obj(in_obj)))
        if in_gnode is not None:
            return [in_gnode]

        return gnode.get_compatible_in_gnodes(in_obj, in_tag, in_metatag, in_arg_index)

    def _recurse_single_input(self, mnode, frame, prev_sol, in_obj, in_arg_index, state):
        assert prev_sol is None or isinstance(prev_sol, MatchSolution)
        def debug(s):
            self.debug("{}{}".format("        " * frame.depth, s))

        context = self.context
        config = self.config
        gnode = frame.gnode

        if in_obj is None:  # TODO: PLACEHOLDER_OBJ?
            yield prev_sol, state
            return

        if state is None:
            assert len(self.created_gnodes) == 0
            assert len(self.edges) == 0

        if gnode.is_root:
            # This is root gnode: Recurse into an output variable
            assert in_obj is _get_obj(context.output_variables[in_arg_index][0])
            in_gnode = context.graph.output_nodes[in_arg_index]

            frame = MatchStackFrame(self, frame, in_gnode, in_arg_index, in_obj)
            if frame.good:
                for sol, st in self._find_matches_partial_tree(prev_sol, mnode, frame, state):
                    yield sol, st
            return


        debug('- input: {} id:{}'.format(type(in_obj), id(in_obj)))
        if isinstance(in_obj, (variable.VariableNode,) + _ndarrays):
            in_tag, in_metatag = context._get_variable_tags(in_obj)
        elif isinstance(in_obj, SubgraphObj):
            in_tag = in_obj.tag
            in_metatag = in_obj.metatag
        else:
            in_tag = None
            in_metatag = None

        debug('in_tag={} in_metatag={} in_arg_index={}'.format(in_tag, in_metatag, in_arg_index))

        # (1) Examine known input gnodes
        debug('(1)')
        for in_gnode in self.get_compatible_in_gnodes(gnode, in_obj, in_tag, in_metatag, in_arg_index):
            debug('Compatible node: {}'.format(in_gnode))
            frame = MatchStackFrame(self, frame, in_gnode, in_arg_index, in_obj)
            if frame.good:
                new_state = MatchStateChain(self, state, (in_gnode, gnode, in_arg_index), False)
                for sol, st in self._find_matches_partial_tree(prev_sol, mnode, frame, new_state):
                    yield sol, st
                else:
                    debug("no solution with known inputs: abandon: {} {}".format(gnode, in_gnode))
                    new_state.abandon()
        else:
            debug('  No known path')

        # (2) Examine compatible gnodes
        to_create_edge = config.can_create_edge

        if to_create_edge:
            debug('(2)')
            for in_gnode in self.get_compatible_gnodes(in_obj, in_tag, in_metatag):
                debug('Compatible gnode: {}'.format(in_gnode))
                frame = MatchStackFrame(self, frame, in_gnode, in_arg_index, in_obj)
                if frame.good:
                    new_state = MatchStateChain(self, state, (in_gnode, gnode, in_arg_index), False)
                    for sol, st in self._find_matches_partial_tree(prev_sol, mnode, frame, new_state):
                        yield sol, st
                    else:
                        debug("no solution with compatible gnodes: abandon: {} {}".format(gnode, in_gnode))
                        new_state.abandon()
            else:
                debug('  No compatible gnode')


        # (3) Examine creating new gnode
        to_create_node = config.can_create_node

        if to_create_node:
            debug('(3)')
            in_gnode = Gnode.from_obj(in_obj, in_tag, in_metatag)
            debug("create: tag={} in_obj={}".format(in_tag, in_obj))
            frame = MatchStackFrame(self, frame, in_gnode, in_arg_index, in_obj)
            if frame.good:
                new_state = MatchStateChain(self, state, (in_gnode, gnode, in_arg_index), True, in_obj)
                for sol, st in self._find_matches_partial_tree(prev_sol, mnode, frame, new_state):
                    yield sol, st
                else:
                    debug("no solution with creating gnode: abandon: {} {}".format(gnode, in_gnode))
                    new_state.abandon()

    def get_compatible_gnodes(self, obj, tag, metatag):
        #if isinstance(obj, variable.VariableNode) and tag is None and metatag is None:
        if tag is None and metatag is None:
            return []
        clue = Gnode.get_clue(obj, tag, metatag)
        gnodes = self.context.graph.get_compatible_nodes(obj, tag, metatag)
        for gnode in self.created_gnodes:
            if gnode.clue == clue:
                gnodes.append(gnode)
        return gnodes


class GraphContext(object):
    def __init__(self, tag, graph):
        self.tag = tag
        self.graph = graph
        self._closed = False

        self.n_passes = 0
        self._output_variables_of_last_pass = []

        graph._last_context = self

    def start_pass(self, inputs, trainer=None):
        self.debug("Starting context pass: inputs={}".format([type(_) for _ in inputs]))
        # See: comment of end_pass()
        last_outputs = [_() for _ in self._output_variables_of_last_pass]
        last_outputs = [_ for _ in last_outputs if _ is not None]
        self._init_pass(inputs, last_outputs, trainer)

    def end_pass(self):
        # Weakly keep the output variables of this pass.
        # They are used to identify the input variables of the next pass, if omitted.
        # If a weak reference is invalidated, that means the variable is no longer used
        # and cannot be a part of the input variables.
        self._output_variables_of_last_pass = [
            weakref.ref(v) for v,_,_ in self.output_variables if v is not None]

        self._cleanup_pass()
        self.n_passes += 1

    def _init_pass(self, inputs, last_outputs, trainer):
        self._closed = False
        self.nodes = set()
        self._var_unnamed_tag_counter = 0
        self.buffered_node_configs = None
        self.trainer = trainer
        self.temp_subgraphs = {}

        # chainer.Variable
        self.output_variables = None
        self.input_variables = inputs
        self.last_output_variables = {id(_._node) for _ in last_outputs}

        self.input_variable_set = {id(_get_obj(_)) for _ in inputs if _ is not None}

        # chainer.VariableNode -> tag, metatag
        self.variable_map = {}
        for i,v in enumerate(inputs):
            obj_ = _get_obj(v)
            self.variable_map[id(obj_)] = (None, None)

        # tag -> list of chainer.VariableNode
        self.variable_map2 = {}

        # chainer.VariableNode -> chainer.Variable
        self.variable_node_map = {}

        # id(ndarray) => ndarray
        self.root_variable_map = {}

        self.subgraph_output_map = {} # output variable -> (subgraph, index, gnode)

        # Submit input variables to the graph
        for v in inputs:
            if isinstance(v, variable.Variable):
                self._memorize_variable(v)
            elif isinstance(v, _ndarrays):
                self.root_variable_map[id(v)] = v

    def _cleanup_pass(self):
        # TODO: These variables and related operations could be capsulated into
        #       a separate class.
        del self.nodes
        del self.buffered_node_configs
        del self.output_variables
        del self.input_variables
        del self.input_variable_set
        del self._var_unnamed_tag_counter
        del self.variable_map
        del self.variable_map2
        del self.variable_node_map
        del self.subgraph_output_map
        del self.trainer
        del self.temp_subgraphs

    def set_output(self, outputs):
        self.debug("set_output: {}".format([type(_) for _ in outputs]))
        assert not self._closed
        assert any(_ is not None for _ in outputs)
        output_variables = []

        for i,var in enumerate(outputs):
            obj_ = _get_obj(var)
            tag, metatag = self._get_variable_tags(var)

            # If the output variable is not tagged,
            # put a metatag to avoid creating loop connection to this variable.
            # Note that `outputs` can have multiple duplicate variables,
            # in which case metatag can be non-None for latter variables.
            if tag is None and metatag is None:
                metatag = '$o{}'.format(i)

            self.variable_map[id(obj_)] = (tag, metatag)
            output_variables.append((var, tag, metatag))

        self.output_variables = output_variables
        self._closed = True

        #
        self.debug("subgraph_output_map: {}".format(self.subgraph_output_map))
        # Create output variables to the graph
        self.graph.submit_outputs(self.output_variables)

    def _memorize_variable(self, obj):
        """
        Memorize mapping from variable.VariableNode to variable.Variable
        and prevent automatic data release
        """
        assert isinstance(obj, variable.Variable)
        obj_ = _get_obj(obj)
        self.variable_node_map[id(obj_)] = obj

    def _is_input_variable(self, obj):
        if id(obj) in self.input_variable_set:
            return True
        if isinstance(obj, variable.VariableNode):
            return id(obj.data) in self.input_variable_set
        return False

    def _get_variable_tags(self, obj):
        if obj is None:
            return None, None
        obj_ = _get_obj(obj)
        tup = self.variable_map.get(id(obj_), None)

        # xp.ndarray will be converted to Variable (and VariableNode) by a Function.
        # Original xp.ndarray can be found in VariableNode.data.
        if tup is None and isinstance(obj, variable.VariableNode):
            obj_ = obj.data
            tup = self.variable_map.get(id(obj_), None)

        if tup is None:
            return None, None
        else:
            assert len(tup) == 2
            return tup

    def set_tag(self, obj, tag):
        assert obj is not None
        assert isinstance(obj, (variable.Variable,) + _ndarrays)
        assert isinstance(tag, str)
        #assert tag not in self.variable_map2
        if isinstance(obj, variable.Variable):
            self._memorize_variable(obj)

        tag_, metatag_ = self._get_variable_tags(obj)
        if tag_ is not None:
            raise RuntimeError("Variable is already tagged as '{}'".format(
                tag_))

        obj_ = _get_obj(obj)
        self.variable_map[id(obj_)] = (tag, metatag_)
        self.variable_map2.setdefault(tag, []).append(obj_)
        self.debug("set_tag({})".format(tag))
        self.debug("{}, {}".format(id(self), self.variable_map2.keys()))

    def _ensure_variable_has_tag(self, obj):
        tag, metatag = self._get_variable_tags(obj)
        if tag is None:
            tag = 'var{}'.format(self._var_unnamed_tag_counter)
            self._var_unnamed_tag_counter += 1
            self.set_tag(obj, tag)

        return tag

    def config_node(self, target, **kwargs):
        # If target is tag (=str), just store the tag for future statistics collection.
        # If it's a variable (ndarray or Variable), attach a tag (if not yet) to it
        # and use that tag.

        if isinstance(target, str):
            tag = target
        elif isinstance(target, (variable.Variable,) + _ndarrays):
            tag = self._ensure_variable_has_tag(target)
        else:
            assert False

        #
        self.debug("config_node: {}".format(tag))
        if self.buffered_node_configs is None:
            self.buffered_node_configs = []
        self.buffered_node_configs.append((
            tag, GnodeConfig(**kwargs)
        ))

    def debug(self, s):
        if DEBUG:
            print("{}: {}".format(self.tag, s))

    def get_subgraph(self, tag, create=False):
        assert isinstance(tag, str)

        subgraph = self.graph.get_subgraph(tag, False)
        if subgraph is not None:
            # If the graph has the subgraph, return it
            return subgraph
        else:
            # If not, create it within the graph context
            if not create:
                return None
            subconfigs = self.graph.get_node_config(tag)
            subgraph = Graph(tag, subconfigs)
            self.temp_subgraphs[tag] = subgraph
            return subgraph

    def submit_subgraph(self, subgraph, input_variables, output_variables, metatag=None):
        self.debug("submit_subgraph: {}".format(subgraph.tag))

        obj = SubgraphObj(subgraph, input_variables, output_variables, metatag)

        subgraph_tag = subgraph.tag
        assert isinstance(subgraph_tag, str)

        for i,out_var in enumerate(output_variables):
            self.subgraph_output_map[id(_get_obj(out_var))] = (obj, i)

    def submit_graph(self):
        # Submit input variables to the graph
        input_tuples = []
        for i,var in enumerate(self.input_variables):
            tag, metatag = self._get_variable_tags(var)

            # If the input variable is not tagged,
            # put a metatag to avoid creating loop connection to this variable.
            if metatag is None and tag is None:
                metatag = '$i{}'.format(i)

            obj = _get_obj(var)
            self.variable_map[id(obj)] = (tag, metatag)
            input_tuples.append((var, tag, metatag))

        self.graph.submit_inputs(input_tuples)

        # Submit subgraphs
        for tag, subgraph in self.temp_subgraphs.items():
            self.graph.set_subgraph(tag, subgraph)
        self.temp_subgraphs = None

        # Find the best match
        solution = self._find_best_match_tree()
        assert not solution.empty()

        # Create graph nodes and edges according to the match tree
        mnodes = self._submit_match_tree(solution)

        # Submit buffered node configs
        self._submit_buffered_node_configs()

        # Submit data statistics
        self._submit_data_statistics(mnodes)

        # Debug
        self.graph.assert_consistent()

    def _find_best_match_tree(self):
        self.debug("Finding the best match tree.")
        if not self.graph.has_edges:
            #configs = (MatchConfig(can_create_edge=True, can_create_node=True, max_create=_) for _ in itertools.count(1))
            configs = [MatchConfig(can_create_edge=True, can_create_node=True, max_create=None)]
        else:
            configs = itertools.chain(
                [
                    MatchConfig(can_create_edge=False, can_create_node=False, max_create=0),
                ],
                (MatchConfig(can_create_edge=True, can_create_node=True, max_create=_) for _ in itertools.count(1)))

        time_start = time.time()
        for i_try, config in enumerate(configs):
            self.debug("")
            self.debug("Matching trial {}: config={}".format(i_try, config.__dict__))
            search = MatchSearch(config, self)
            solution = search.run()

            if solution is not None and search.is_all_gnodes_filled():
                break

        if solution.empty():
            self.debug("Solution not found at pass {}".format(self.n_passes))
            assert False

        time_end = time.time()
        print(self.tag, i_try, "  time: {} sec".format(time_end - time_start))

        return solution

    def _submit_match_tree(self, match_solution):
        # Traverses the tree from input nodes

        front = set(match_solution.leaf_nodes())
        submitted_mnodes = set()

        while len(front) > 0:
            mnode = front.pop()
            if mnode.submitted:
                continue

            mnode.submitted = True
            submitted_mnodes.add(mnode)

            gnode = mnode.gnode
            self.graph._add_node(gnode)

            prev_mnode = mnode.prev_mnode
            if prev_mnode is not None:
                prev_gnode = prev_mnode.gnode
                if prev_gnode.is_root:
                    continue

                out_edge = GraphEdge(gnode, prev_gnode, mnode.arg_index)
                prev_gnode.in_edges.add(out_edge)
                gnode.out_edges.add(out_edge)

                # Edge statistics
                out_edge.count += 1

                front.add(prev_mnode)

        return submitted_mnodes

    def _submit_buffered_node_configs(self):
        if self.buffered_node_configs is None:
            return
        for tag, node_config in self.buffered_node_configs:
            gnode = self.graph.node_map[tag]
            if not isinstance(gnode, VariableGnode):
                raise NotImplementedError('Currently only variable nodes have data statistics')
            gnode.set_node_config(node_config)

        self.buffered_node_configs = None

    def _submit_data_statistics(self, mnodes):
        # Node statistics
        for mnode in mnodes:
            gnode = mnode.gnode

            if isinstance(mnode.obj, (variable.VariableNode,) + _ndarrays):
                if gnode.node_config is not None:
                    if isinstance(mnode.obj, variable.VariableNode):
                        var = self.variable_node_map.get(id(mnode.obj))
                        if var is not None:
                            data = var.data
                        else:
                            data = self.root_variable_map[id(mnode.obj.data)]
                    else:
                        data = mnode.obj
                    self.debug('Add data sample: {}'.format(gnode.tag))
                    gnode.add_data_sample(data, self.trainer)


class DummyGraphContext(object):
    def __init__(self):
        pass

    def set_output(self, outputs):
        pass

    def config_node(self, *args, **kwargs):
        pass

    def set_tag(self, *args, **kwargs):
        pass


@contextlib.contextmanager
def root_graph(input_variables, graph, trainer=None, context=None):
    assert context is None or isinstance(context, GraphContext)
    current_thread = threading.current_thread()

    graph.lock()

    if context is None:
        context = GraphContext(graph.tag, graph)
    current_thread.__dict__['graph_context'] = context
    context.start_pass(input_variables, trainer=trainer)
    time1 = time.time()

    try:
        yield context
    finally:
        context.submit_graph()
        context.end_pass()
        time2 = time.time()
        print("root_graph: {} sec".format(time2 - time1))
        current_thread.__dict__['graph_context'] = None
        graph.unlock()


class graph(object):
    def __init__(self, input_variables, tag, metatag=None, enable=True):
        assert isinstance(tag, str)
        self.input_variables = input_variables
        self.tag = tag
        self.metatag = metatag
        self.enable = enable

    def cleanup(self):
        self.input_variables = None
        self.graph = None

    def __enter__(self):
        if not self.enable:
            context = DummyGraphContext()
        else:
            current_thread = threading.current_thread()
            outer_context = current_thread.__dict__.get('graph_context', None)

            input_variables = self.input_variables
            tag = self.tag

            if outer_context is None:
                # There's no root graph.
                self.enable = False
                return DummyGraphContext()

            # Take the graph from the outer context
            graph = outer_context.get_subgraph(tag, create=True)

            context = graph._last_context
            if context is None:
                context = GraphContext(tag, graph)

            current_thread.__dict__['graph_context'] = context

            context.start_pass(input_variables, trainer=outer_context.trainer)

            self.outer_context = outer_context

        self.context = context
        return context

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enable:
            outer_context = self.outer_context

            if exc_type is None:
                context = self.context
                input_variables = self.input_variables
                graph = context.graph
                context.submit_graph()

                outer_context.submit_subgraph(
                    graph,
                    input_variables,
                    [v for v,_,_ in context.output_variables],
                    self.metatag)

                if not context._closed:
                    raise RuntimeError(
                        "You must call GraphContext.set_output()"
                        " at the end of the graph context.")

                context.end_pass()

            current_thread = threading.current_thread()
            current_thread.__dict__['graph_context'] = outer_context

        self.cleanup()
