import six

import chainer
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer import link


class Cell(link.ChainList):

    """The generic Recurrent Neural Network link (RNN) cell.

    This is an implementation of a generic
    Recurrent Neural Network Cell.
    The underlying idea is to simply stack multiple
    Stateful RNN cells where the RNN cell at the bottom takes
    the regular input, and the cells after that simply take
    the outputs (represented by h) of the previous GRUs as inputs.

    Args:
          in_size (int)- The size of embeddings of the inputs
          out_size (int)- The size of the hidden layer
                    representation of each GRU unit
          cell_type (~chainer.Chain or ~chainer.ChainList)-
          The type of RNN cell which can be GRU or LSTM
          or anything in general
          num_layers (int)- The number of RNN layers
    Attributes:
          num_layers (int): Indicates the number of cell layers
          cell_type (~chainer.Chain or ~chainer.ChainList):
          Indicates the RNN cell type
    User Defined Methods:
    """

    def __init__(self, in_size, out_size, cell_type, num_layers=1):
        super(Cell, self).__init__()
        assert num_layers >= 1
        assert cell_type is not None
        self.add_link(cell_type(in_size, out_size))
        for i in range(1, num_layers):
            self.add_link(cell_type(out_size, out_size))
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.reset_state()

    def to_cpu(self):
        if 'to_cpu' in dir(self[0]):
            for layer in self:
                layer.to_cpu()

    def to_gpu(self, device=None):
        if 'to_gpu' in dir(self[0]):
            for layer in self:
                layer.to_gpu(device)

    def set_state(self, h):
        if 'set_state' in dir(self[0]):
            h = split_axis.split_axis(h, self.num_layers, 1, True)
            for layer, h in six.moves.zip(self, h):
                assert isinstance(h, chainer.Variable)
                layer.set_state(h)

    def reset_state(self):
        if 'reset_state' in dir(self[0]):
            for layer in self:
                layer.reset_state()

    def __call__(self, x, h=None, top_n=None):
        """Updates the internal state and returns the Cell outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.
            h (~chainer.Variable): The batched form of the previous state.
            Make sure that you pass the previous state if you
            use stateless RNN cells
            top_n (int): The number of cells from the top whose outputs
            you want (default: outputs of all GRUs are returned)
            When using stateless cells the states of all cells will
            be returned as they will be needed for the next step

        Returns:
            ~chainer.Variable: A concatenation of the outputs (h)
            of the updated cell units over the top N layers;
            by default all layers are considered.

        """
        if top_n is None:
            top_n = self.num_layers
        if h is not None:
            assert 'reset_state' not in self[0]
            assert top_n is self.num_layers
            h = split_axis.split_axis(h, self.num_layers, 1, True)
        h_list = []
        h_curr = x
        for layer_id, layer in enumerate(self):
            if h is None:
                h_curr = layer(h_curr)
            else:
                h_curr = layer(h_curr, h[layer_id])
            h_list.append(h_curr)
        return concat.concat(h_list[-top_n:], 1)
