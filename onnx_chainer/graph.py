import collections
from collections import OrderedDict
import heapq

import chainer

from onnx_chainer.functions.converter import FunctionConverterParams
from onnx_chainer import onnx_helper


class Graph(object):

    def __init__(self, context, converters, opset_version,
                 explicit_input_names, network_outputs):
        self.context = context
        self.converters = converters

        self.graph = []
        self.func_name_counts = collections.defaultdict(int)
        self.outputs = set()  # Output variable names
        self.specified_opset_version = opset_version
        self.explicit_input_names = explicit_input_names
        self.network_outputs = network_outputs

        self.function_nodes = self._build_computational_graph(
            network_outputs.values())

    def _build_computational_graph(self, outputs):
        cands = []
        function_nodes = OrderedDict()
        push_count = [0]

        def add_cand(cand):
            heapq.heappush(cands, (-cand.rank, push_count[0], cand))
            push_count[0] += 1

        for o in outputs:
            if isinstance(o, chainer.Variable):
                o = o.node
            add_cand(o)

        while cands:
            _, _, cand = heapq.heappop(cands)
            if not isinstance(cand, chainer.variable.VariableNode):
                raise NotImplementedError(
                    'ONNX-Chainer does not support node type {}'.format(
                        type(cand)))
            creator = cand.creator_node
            if creator is None:
                continue
            assert isinstance(creator, chainer.FunctionNode)
            creator_id = id(creator)
            if creator_id in function_nodes:
                continue
            function_nodes[creator_id] = creator

            for input_ in creator.inputs:
                add_cand(input_)

        return reversed(function_nodes.values())

    def create_node(
            self, func_name, func, input_names, output_names):
        converter = self.converters.get(func_name, None)
        if converter is None:
            raise ValueError('{} is not supported'.format(func_name))
        params = FunctionConverterParams(
            func, self.specified_opset_version, input_names, output_names,
            self.context)
        nodes = converter(params)
        return list(nodes)

    def convert_to_onnx_node(self, function):
        if isinstance(function, chainer.function.FunctionAdapter):
            function = function.function
        func_name = getattr(
            function, 'custom_function_node_name', function.__class__.__name__)
        base_func_name = '{}_{}'.format(
            func_name, self.func_name_counts[func_name])
        self.func_name_counts[func_name] += 1

        input_names = []
        for input_var in function.inputs:
            # 'input_var' is a VariableNode,
            # so check if it has a Variable/Parameter
            var = input_var.get_variable_or_none()
            if var is None:  # No reference to Variable/Parameter
                # Use VariableNode as is
                input_name = self.context.get_name(input_var)
            else:  # It is a parameter inside a Link or network input
                input_name = self.context.get_name(var)
                if (input_name not in self.explicit_input_names and
                        input_name not in self.outputs):
                    # register input variables to check implicit inputs
                    self.context.implicit_inputs[input_name] = var
            input_names.append(input_name)

        # This is to get corresponding VariableNode id from the output
        # Variable of the network
        output_names = []
        for i, output_ref in enumerate(function.outputs):
            if output_ref() is None:
                var = output_ref
            else:
                var = output_ref().get_variable_or_none()
                if var is None:
                    var = output_ref()
            output_name = self.context.get_name(var)

            # The context sets unique names on node and param, like "v1".
            # To be more understandable, change the names like function name
            # + number like "FuncitionName_0"
            if not self.context.is_pinned(var):
                if len(function.outputs) == 1:
                    new_name = base_func_name
                else:
                    new_name = '{}_{}'.format(base_func_name, i)
                if output_name in self.network_outputs:
                    del self.network_outputs[output_name]
                    self.network_outputs[new_name] = var
                self.context.set_name(var, new_name)
                output_name = new_name
            self.outputs.add(output_name)

            output_names.append(output_name)

        onnx_helper.set_func_name(base_func_name)
        nodes = self.create_node(
            func_name, function, input_names, output_names)
        # Insert constants before computation nodes.
        self.graph.extend(self.context.constants)
        self.context.constants.clear()
        self.graph.extend(nodes)

    def to_onnx_graph(self):
        for node in self.function_nodes:
            self.convert_to_onnx_node(node)
