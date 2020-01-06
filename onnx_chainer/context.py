import chainer

import onnx
from onnx import numpy_helper

from onnx_chainer import onnx_helper


def _tensor_from_array_for_constant(array, name):
    tensor = numpy_helper.from_array(array, name=name)
    # Avoid `raw_data` for better debuggability. This would be OK
    # since constants are usually small.
    field_name = onnx.mapping.STORAGE_TENSOR_TYPE_TO_FIELD.get(
        tensor.data_type, None)
    if field_name is not None:
        tensor.ClearField('raw_data')
        getattr(tensor, field_name)[:] = array.flatten().tolist()
    return tensor


class Context(object):
    """Context of converter

    This context shares names during exporting.

    Attributes:
        name_list (dict): list of being exported as ONNX node name with pinned
            or not, keyed by instance ID. When the target variable is
            ``chainer.Variable`` or ``chainer.Parameter``, instance ID of
            ``ndarray`` held by the variable is also put as key, because some
            functions like ``F.where`` internally unwrap variable.

    """

    def __init__(self, model):
        self.name_list = dict()
        self.parameters = []
        self.constants = []
        self.implicit_inputs = dict()  # inputs which not connect to output
        namedlink = {n: l for n, l in model.namedlinks()}
        self.param_to_link = {}
        for name, param in model.namedparams():
            owned_link_name = name[:name.rindex('/')]
            if owned_link_name in namedlink:
                onnx_owned_link_name = onnx_helper.cleanse_param_name(
                    owned_link_name)
                self.param_to_link[id(param)] = (
                    onnx_owned_link_name, namedlink[owned_link_name])
            onnx_name = onnx_helper.cleanse_param_name(name)
            self.set_name(param, onnx_name)

    def get_name(self, variable):
        str_id = id(variable)
        if str_id in self.name_list:
            return self.name_list[str_id][0]
        else:
            new_name = 'v{}'.format(len(self.name_list))
            self.set_name(variable, new_name)
            return new_name

    def set_name(self, variable, name, pinned=False):
        """Set ONNX node name

        Arguments:
            variable (var): target variable
            name (str): name to be exported as ONNX node name
            pinned (bool): if ``True``, the name will not be overwritten in
                subsequence process.
        """

        str_id = id(variable)
        assert str_id not in self.name_list or not self.name_list[str_id][1]
        self.name_list[str_id] = (name, pinned)
        if isinstance(variable, (chainer.Variable, chainer.Parameter)):
            array_id = id(variable.array)
            self.name_list[array_id] = (name, pinned)

    def is_pinned(self, variable):
        str_id = id(variable)
        if str_id not in self.name_list:
            return False
        return self.name_list[str_id][1]

    def add_param(self, array, name, use_original_name=False):
        """Add a parameter array as an ONNX initializer.

        Returns:
            str: registered name.
        """
        if use_original_name:
            onnx_name = name
        else:
            if not (name.startswith('/') or name.startswith('_')):
                name = '/' + name
            onnx_name = '{}_{}'.format(
                onnx_helper.get_func_name(),
                onnx_helper.cleanse_param_name(name))
        self.set_name(array, onnx_name)
        self.parameters.append(array)
        return onnx_name

    def add_const(self, array, name):
        """Add a constant array as an ONNX Constant node.

        Returns:
            str: registered name.
        """
        assert '/' not in name
        onnx_name = '{}_const_{}'.format(onnx_helper.get_func_name(), name)
        self.set_name(array, onnx_name)
        tensor = _tensor_from_array_for_constant(array, name=onnx_name)
        const_node = onnx_helper.make_node(
            'Constant', [], [onnx_name], value=tensor)
        self.constants.append(const_node)
        return onnx_name

    def get_link(self, param):
        """Return link with name which has the param.

        Arguments:
            param(chainer.Parameter): the target param.

        Returns:
            tuple: name and link. returns ``None`` when not found.
        """
        return self.param_to_link.get(id(param), None)
