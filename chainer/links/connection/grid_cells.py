import six

from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import variable

class GridLSTMBase(link.Chain):

    def __init__(self, in_size, out_size,
                 lateral_init=None):
        super(GridLSTMBase, self).__init__(
            lateral=linear.Linear(in_size, 4 * out_size,
                                  initialW=0, nobias=True),
        )
        self.state_size = out_size

        for i in six.moves.range(0, 4 * out_size, out_size):
            initializers.init_weight(
                self.lateral.W.data[i:i + out_size, :], lateral_init)

class GridGRUBase(link.Chain):

    def __init__(self, n_inputs, n_units, init=None,
                 inner_init=None, bias_init=0):
        if n_inputs is None:
            n_inputs = n_units
        super(GRUBase, self).__init__(
            U_r=linear.Linear(n_units, n_units,
                              initialW=inner_init, initial_bias=bias_init),
            U_z=linear.Linear(n_units, n_units,
                              initialW=inner_init, initial_bias=bias_init),
            U=linear.Linear(n_units, n_units,
                            initialW=inner_init, initial_bias=bias_init),
        )


class StatelessGridGRUbase(GridGRUBase):

    """Stateless Gated Recurrent Unit function (GRU).
       This is the same as a regular GRU except that there is no input
       except the initial hidden state which is a transform of the actual
       input.
    Args:
        n_inputs(int): Dimension of input vector :math:`x`. If ``None``,
            it is set to the same value as ``n_units``.
        n_units(int): Dimension of hidden vector :math:`h`.
        
    See:
        - `On the Properties of Neural Machine Translation: Encoder-Decoder
          Approaches <http://www.aclweb.org/anthology/W14-4012>`_
          [Cho+, SSST2014].
        - `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
          Modeling <http://arxiv.org/abs/1412.3555>`_
          [Chung+NIPS2014 DLWorkshop].

    """

    def __call__(self, h):
        r = sigmoid.sigmoid(self.U_r(h))
        z = sigmoid.sigmoid(self.U_z(h))
        h_bar = tanh.tanh(self.U(r * h))
        h_new = (1 - z) * h + z * h_bar
        return h_new

class StatelessGridLSTMbase(GridLSTMBase):

    """Stateless LSTM cell.

    This is a fully-connected LSTM cell as a chain to be used in a grid.

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        upward (chainer.links.Linear): Linear layer of upward connections.
        lateral (chainer.links.Linear): Linear layer of lateral connections.

    """

    def __call__(self, c, h):
        """Returns new cell state and updated output of LSTM.

        Args:
            c (~chainer.Variable): Cell states of LSTM units.
            h (~chainer.Variable): Output at the previous timestep.
            For a grid LSTM the initial h and c should be a transform
            of the input x
            For a grid LSTM h acts as the input at each step which itself
            is a transform of the original input for the first step
            
        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
                ``c_new`` represents new cell state, and ``h_new`` is updated
                output of LSTM units.

        """
        assert c is not None
        assert h is not None
        lstm_in = self.lateral(h)
        return lstm.lstm(c, lstm_in)


class GridLSTMCell(link.ChainList):

    """The Grid LSTM Memory Cell.

    This is an implementation of a Grid LSTM Cell.
    It is composed of a number of memory cells.
    Their weights may be shared and the individual cells
    may be of different types.
    The underlying idea is based on the following paper:
    http://arxiv.org/pdf/1507.01526v3.pdf

    Args:
          cell_sizes (list of ints)- The sizes of the cells
          of the individual cells in the N-Grid.
          sharing_dimensions (list of list of ints)-
          The number of inner lists indicate the total number
          of memory cells and the inner lists themselves
          indicate the dimensions that share said cells.
          dimensionality (int)- The dimensionality of the Grid
    Attributes:
    	  in_indices (list of ints)- The indices indicating the boundaries
          of the individual inputs in the N-Grid.
          dimensionality (int): Indicates the dimensionality
          of the N-Grid cell
          sharing_dimensions (list of list of ints)-
          The number of inner lists indicate the total number
          of memory cells and the inner lists themselves
          indicate the dimensions that share said cells.
    Notes:
    	  Following are some ways to create a Grid memory cell.
    	  1. grid_cell = GridLSTMCell([200, 300, 400],
    	  	 [[1,3], [2,4], [5]], 5)
    	     This creates a 5-Grid cell where there are 3 memory cells
    	     The first cell is a LSTM and is shared between dimensions
    	     1 and 3. The second is a LSTM and is shared between 2 and 4.
    	     The last one is also a LSTM and is a standalone cell.
    	     The first cell takes batched input of size (batch_size, 200)
    	     and produces a batched output of size (batch_size, 300).
    	     The sizes of the other cells should be similarly interpreted.
    User Defined Methods:
    """

    def __init__(self, cell_sizes, 
    	sharing_dimensions, dimensionality=1):
        super(GridLSTMCell, self).__init__()
        assert dimensionality >= 1
        dim = sum([sum(shares) for shares in sharing_dimensions])
        assert dim == dimensionality
        comb_in = sum(cell_sizes)
        for i in range(len(sharing_dimensions)):
        	self.add_link(StatelessGridLSTMbase(comb_in, cell_sizes[i]]))
        self.in_indices = [x+y for x,y in zip(cell_sizes,[0]+cell_sizes[:-1])]
        self.dimensionality = dimensionality
        self.sharing_dimensions = sharing_dimensions
        

    def __call__(self, c, h):
        """Updates the internal state and returns the Cell outputs.

        Args:
            c (~chainer.Variable): The previous memory of the Grid cell.
            h (~chainer.Variable): The batched form of the previous state.
            
        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
            ``c_new`` represents new cell state, and ``h_new`` is updated
            output of LSTM units.

        """
        assert h is not None
        assert c is not None
        c = split_axis.split_axis(c, self.out_indices, 1, True)
        h_list = []
        h_curr = None
        for layer_id, layer in enumerate(self):
        	h_curr = layer(c[layer_id], h)
            h_list.append(h_curr)
        h_new = concat.concat([h[1] for h in h_list], 1)
        c_new = concat.concat([h[0] for h in h_list], 1)
        return c_new, h_new


class GridGRUCell(link.ChainList):

    """The Grid GRU Cell.

    This is an implementation of a Grid GRU Cell.
    It is composed of a number of memory cells.
    Their weights may be shared and the individual cells
    may be of different types.
    The underlying idea is based on the following paper:
    http://arxiv.org/pdf/1507.01526v3.pdf

    Args:
          cell_sizes (list of ints)- The sizes of the cells
          of the individual cells in the N-Grid.
          sharing_dimensions (list of list of ints)-
          The number of inner lists indicate the total number
          of memory cells and the inner lists themselves
          indicate the dimensions that share said cells.
          dimensionality (int)- The dimensionality of the Grid
    Attributes:
    	  in_indices (list of ints)- The indices indicating the boundaries
          of the individual inputs in the N-Grid.
          dimensionality (int): Indicates the dimensionality
          of the N-Grid cell
          sharing_dimensions (list of list of ints)-
          The number of inner lists indicate the total number
          of memory cells and the inner lists themselves
          indicate the dimensions that share said cells.
    Notes:
    	  Following are some ways to create a Grid memory cell.
    	  1. grid_cell = GridGRUCell([200, 300, 400],
    	  	 [[1,3], [2,4], [5]], 5)
    	     This creates a 5-Grid cell where there are 3 memory cells
    	     The first cell is a GRU and is shared between dimensions
    	     1 and 3. The second is a GRU and is shared between 2 and 4.
    	     The last one is also a GRU and is a standalone cell.
    	     The first cell takes batched input of size (batch_size, 200)
    	     and produces a batched output of size (batch_size, 300).
    	     The sizes of the other cells should be similarly interpreted.
    User Defined Methods:
    """

    def __init__(self, cell_sizes, 
    	sharing_dimensions, dimensionality=1):
        super(GridLSTMCell, self).__init__()
        assert dimensionality >= 1
        dim = sum([sum(shares) for shares in sharing_dimensions])
        assert dim == dimensionality
        comb_in = sum(cell_sizes)
        for i in range(len(cell_types)):
        	self.add_link(StatelessGridGRUbase(comb_in, cell_sizes[i]]))
        self.in_indices = [x+y for x,y in zip(cell_sizes,[0]+cell_sizes[:-1])]
        self.dimensionality = dimensionality
        self.sharing_dimensions = sharing_dimensions
        

    def __call__(self, h):
        """Updates the internal state and returns the Cell outputs.

        Args:
            h (~chainer.Variable): The batched form of the previous state.
            
        Returns:
            ~chainer.Variable: Returns h_new), where
            ``h_new`` is updated output of LSTM units.

        """
        assert h is not None
        h_list = []
        h_curr = None
        for layer in self:
        	h_curr = layer(h)
            h_list.append(h_curr)
        h_new = concat.concat(h_list, 1)
        return h_new

memory_cell_dictionary = {'GRU': StatelessGridGRUbase, 
						  'LSTM': StatelessGridLSTMbase}

class GridCell(link.ChainList):

    """The generic Grid Memory Cell.

    This is an implementation of a Grid Memory Cell.
    It is composed of a number of memory cells.
    Their weights may be shared and the individual cells
    may be of different types.
    We can have LSTM and GRU cells in the same N-Grid cell.
    The underlying idea is based on the following paper:
    http://arxiv.org/pdf/1507.01526v3.pdf
    This implementation however allows for mixing GRUs and LSTMs.
    When using this cell one must treat it as if there are only LSTMS.

    Args:
          in_sizes (list of ints)- The sizes of the inputs
          of the individual cells in the N-Grid.
          out_sizes (list of ints)- The sizes of the
          hidden outputs of the individual cells in the N-Grid.
          cell_types (list of strings)- The type
          of memory cells which can be GRU or LSTM
          (for now)). The StatelessGridLSTMbase and/or
		  StatelessGridGRUbase will be selected accordingly.
		  sharing_dimensions (list of list of ints)-
          The number of inner lists indicate the total number
          of memory cells and the inner lists themselves
          indicate the dimensions that share said cells.
          dimensionality (int)- The dimensionality of the Grid
    Attributes:
          in_indices (list of ints)- The indices indicating the boundaries
          of the individual inputs in the N-Grid.
          dimensionality (int): Indicates the dimensionality
          of the N-Grid cell
          cell_types (strings): The type of memory cells
          which can be GRU or LSTM.
          sharing_dimensions (list of list of ints)-
          The number of inner lists indicate the total number
          of memory cells and the inner lists themselves
          indicate the dimensions that share said cells.
    Notes:
          Following are some ways to create a Grid memory cell.
          1. grid_cell = GridCell([200, 300, 400],[300, 400, 500],
             ['GRU', 'LSTM', 'LSTM'], [[1,3], [2,4], [5]], 5)
             This creates a 5-Grid cell where there are 3 memory cells
             The first cell is a GRU and is shared between dimensions
             1 and 3. The second is a LSTM and is shared between 2 and 4.
             The last one is also a LSTM and is a standalone cell.
             The first cell takes batched input of size (batch_size, 200)
             and produces a batched output of size (batch_size, 300).
             The sizes of the other cells should be similarly interpreted.
    User Defined Methods:
    """

    def __init__(self, in_sizes, out_sizes, cell_types, 
        sharing_dimensions, dimensionality=1):
        super(GridCell, self).__init__()
        assert dimensionality >= 1
        assert cell_types is not None
        assert len(sharing_dimensions) == len(cell_types)
        dim = sum([sum(shares) for shares in sharing_dimensions])
        assert dim == dimensionality
        combined_input_size = sum(in_sizes)
        for i in range(len(cell_types)):
            in_sizes_curr, out_sizes_curr = combined_input_size, out_sizes[i]
            cell = memory_cell_dictionary[cell_types[i]]
            self.add_link(cell(in_sizes_curr, out_sizes_curr))
        self.in_indices = [x+y for x,y in zip(in_sizes,[0]+in_sizes[:-1])]
        self.cell_types = cell_types
        self.dimensionality = dimensionality
        self.sharing_dimensions = sharing_dimensions
        
    def __call__(self, c, h):
        """Updates the internal state and returns the Cell outputs.
           Remember to treat this Grid cell as if its an LSTM and pass
           ``c`` as well as ``h``. Only parts of ``c`` will be used
           depending on whether there is a LSTM or not.

        Args:
            c (~chainer.Variable): The previous memory information.
            h (~chainer.Variable): The previous state information.
            
        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
            ``c_new`` represents new cell state, and ``h_new`` is updated
            output. Parts of ``c_new`` will be useless.

        """
        assert h is not None
        assert c is not None
        c = split_axis.split_axis(c, self.out_indices, 1, True)
        h_list = []
        h_curr = None
        for layer_id, layer in enumerate(self):
        	layer_params = inspect.getargspec(layer)[0]
        	if 'c' in layer_params:
        		h_curr = layer(c[layer_id], h)
        	else:
        		h_curr = (c[layer_id], layer(h))
            h_list.append(h_curr)
        h_new = concat.concat([h[1] for h in h_list], 1)
        c_new = concat.concat([h[0] for h in h_list], 1)
        return c_new, h_new
