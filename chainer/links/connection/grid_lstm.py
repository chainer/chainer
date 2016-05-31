from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer import link
from chainer.links.connection import linear
from chainer import variable
from chainer.links.connection import lstm

class GridCell(link.ChainList):

    """The generic Grid Memory Cell.

    This is an implementation of a Grid Memory Cell.
    It is composed of a number of memory cells.
    Their weights may be shared and the individual cells
    may be of different types.
    We can have LSTM and GRU cells in the same N-Grid cell.
    The underlying idea is based on the following paper:
    http://arxiv.org/pdf/1507.01526v3.pdf

    Args:
          in_size (list of ints)- The sizes of the inputs
          of the individual cells in the N-Grid.
          out_size (list of ints)- The sizes of the
          outputs of the individual cells in the N-Grid.
          cell_types (list of
          ~chainer.Chain or ~chainer.ChainList)- The type
		  of memory cells which can be GRU or LSTM
          or anything in general.
          sharing_dimensions (list of list of ints)-
          The number of inner lists indicate the total number
          of memory cells and the inner lists themselves
          indicate the dimensions that share said cells.
          dimensionality (int)- The dimensionality of the Grid
    Attributes:
          dimensionality (int): Indicates the dimensionality
          of the N-Grid cell
          cell_types (list of
          ~chainer.Chain or ~chainer.ChainList): The type
		  of memory cells which can be GRU or LSTM
          or anything in general.
    User Defined Methods:
    """

    def __init__(self, in_size, out_size, cell_types, sharing_dimensions, dimensionality=1):
        super(GridCell, self).__init__()
        assert dimensionality >= 1
        assert cell_types is not None
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
        """Updates the internal state and returns the GRU outputs.

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
            of the updated GRU units over the top N layers;
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
