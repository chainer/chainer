import chainer
from chainer import cuda
from chainer import function
from chainer.functions.array import split_axis
from chainer import link
from chainer.links.connection import linear
from chainer.utils import type_check
import numpy


class SigmoidAPLusBMultipliedByC(function.Function):
    """SigmoidAPLusBMultipliedByC function. Computes sigmoid(a + b) * c.

    This is faster than doing the same thing by composing chainer operations.
    Used to optimize GRUs.
    """

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(in_types[0].dtype == numpy.float32)
        type_check.expect(in_types[1].dtype == numpy.float32)
        type_check.expect(in_types[2].dtype == numpy.float32)

    def forward_cpu(self, x):
        self.sigma_a_plus_b = (numpy.tanh((x[0] + x[1]) * 0.5) * 0.5 + 0.5)
        return x[2] * self.sigma_a_plus_b,

    def forward_gpu(self, x):
        self.sigma_a_plus_b, y = cuda.elementwise(
            'T x1, T x2, T x3', 'T sigma_a_plus_b, T y',
            '''
                sigma_a_plus_b = tanh((x1 + x2) * 0.5) * 0.5 + 0.5;
                y = x3 * sigma_a_plus_b;
                ''',
            'sigmoid_a_plus_b_by_c_fwd')(x[0], x[1], x[2])
        return y,

    def backward_cpu(self, x, gy):
        gy_by_sigma_a_plus_b = gy[0] * self.sigma_a_plus_b
        deriv1 = x[2] * gy_by_sigma_a_plus_b * (1 - self.sigma_a_plus_b)
        return deriv1, deriv1, gy_by_sigma_a_plus_b

    def backward_gpu(self, x, gy):
        gx1, gx2, gx3 = cuda.elementwise(
            'T sigma_a_plus_b, T h, T gy', 'T gx1, T gx2, T gx3',
            '''
            gx3 = gy * sigma_a_plus_b;
            gx1 = h * gx3 * (1-sigma_a_plus_b);
            gx2 = gx1;
            ''',
            'sigmoid_a_plus_b_by_c_bwd')(self.sigma_a_plus_b, x[2], gy[0])
        return gx1, gx2, gx3,


def sigmoid_a_plus_b_multiplied_by_c(a, b, c):
    """Compute sigmoid(a + b) * c"""
    return SigmoidAPLusBMultipliedByC()(a, b, c)


class ComputeOutputGRU(function.Function):

    """Function used to compute the output of a GRU.

    More precisely, given the five inputs z_x, z_h, h_x, h and hh, it computes:
            z = sigmoid(z_x + z_h)
            h_bar = tanh(h_x + hh)
            h_new = (1 - z) * h + z * h_bar
            return h_new
    """

#     def __init__(self):
#         self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 5)
        type_check.expect(in_types[0].dtype == numpy.float32)
        type_check.expect(in_types[1].dtype == numpy.float32)
        type_check.expect(in_types[2].dtype == numpy.float32)
        type_check.expect(in_types[3].dtype == numpy.float32)
        type_check.expect(in_types[4].dtype == numpy.float32)

    def forward_cpu(self, x):
        z_x, z_h, h_x, h, hh = x
        z = 1.0 / (1 + numpy.exp(- (z_x + z_h)))
        h_bar = numpy.tanh(h_x + hh)
        h_new = (1 - z) * h + z * h_bar
        self.z = z
        self.h_bar = h_bar
        return h_new,

    def forward_gpu(self, x):
        z_x, z_h, h_x, h, hh = x
        self.z, self.h_bar, h_new = cuda.elementwise(
            'T z_x, T z_h, T h_x, T h, T hh',
            'T z, T h_bar, T h_new',
            '''
                z = tanh((z_x + z_h) * 0.5) * 0.5 + 0.5;
                h_bar = tanh(h_x + hh);
                h_new = (1 - z) * h + z * h_bar;
                ''',
            'compute_output_gru_fwd')(z_x, z_h, h_x, h, hh)
        return h_new,

    def backward_cpu(self, x, gy):
        z_x, z_h, h_x, h, hh = x
        g_h = (1 - self.z) * gy[0]
        g_hh = self.z * (1 - self.h_bar * self.h_bar) * gy[0]
        g_h_x = g_hh
        g_z_x = g_h * self.z * (self.h_bar - h)
        g_z_h = g_z_x
        return g_z_x, g_z_h, g_h_x, g_h, g_hh

    def backward_gpu(self, x, gy):
        z_x, z_h, h_x, h, hh = x
        g_z_x, g_z_h, g_h_x, g_h, g_hh = cuda.elementwise(
            'T z, T h_bar, T h, T gy', 'T g_z_x, T g_z_h, T g_h_x, T g_h, T g_hh', '''
            g_h = (1 - z) * gy;
            g_hh = z * (1 - h_bar * h_bar) * gy;
            g_h_x = g_hh;
            g_z_x = g_h * z * (h_bar - h);
            g_z_h = g_z_x;
            ''', 'compute_output_gru_bwd')(
            self.z, self.h_bar, h, gy[0])
        return g_z_x, g_z_h, g_h_x, g_h, g_hh,


def compute_output_GRU(z_x, z_h, h_x, h, hh):
    """Function used to compute the output of a GRU.

    More precisely, given the five inputs z_x, z_h, h_x, h and hh, it computes:
            z = sigmoid(z_x + z_h)
            h_bar = tanh(h_x + hh)
            h_new = (1 - z) * h + z * h_bar
            return h_new
    """
    return ComputeOutputGRU()(z_x, z_h, h_x, h, hh)


class GRUBase(link.Chain):
    """GRUBase instantiate the linear links that parameterize a GRU.
    
    Contrarily to gru.GRUBase, fast_gru.GRUBase only instantiate 3 linear links (vs 6 for gru.GRUBase)
        W_r_z_h merge the links W_r, W_z and W_h of gru.GRUBase
        U_r_z merge the links U_r and U_z of gru.GRUBase. In addition, these links do not have biases, 
        as it would be redundant with the biases of W_r_z_h.
        U is equivalent to the link U of gru.GRUBase (minus the redundant bias).
    """
    def __init__(self, n_units, n_inputs=None):
        if n_inputs is None:
            n_inputs = n_units
        super(GRUBase, self).__init__(
            W_r_z_h=linear.Linear(n_inputs, n_units * 3),
            U_r_z=linear.Linear(n_units, n_units * 2, nobias=True),
            U=linear.Linear(n_units, n_units, nobias=True),
        )
        self.n_units = n_units
        self.n_inputs = n_inputs

    def initialize_with_classic_implementation(self, gru):
        """Initialize the parameters from a gru.GRUBase object.
        
        self and gru should be on the same device.
        """
        assert isinstance(gru, chainer.links.connection.gru.GRUBase)
        # input matrices
        self.W_r_z_h.W.data[:self.n_units] = gru.W_r.W.data
        self.W_r_z_h.W.data[self.n_units: 2*self.n_units] = gru.W_z.W.data
        self.W_r_z_h.W.data[2 * self.n_units:] = gru.W.W.data
        
        # biases
        self.W_r_z_h.b.data[:self.n_units] = gru.W_r.b.data + gru.U_r.b.data
        self.W_r_z_h.b.data[self.n_units: 2*self.n_units] = gru.W_z.b.data + gru.U_z.b.data
        self.W_r_z_h.b.data[2 * self.n_units:] = gru.W.b.data + gru.U.b.data  
        
        # hidden state matrices
        self.U_r_z.W.data[:self.n_units] = gru.U_r.W.data
        self.U_r_z.W.data[self.n_units:] = gru.U_z.W.data

        self.U.W.data[...] = gru.U.W.data
        
    @classmethod
    def from_classic_GRU(cls, gru):
        """Creates a FastGRU object from a GRU object.
        """
        assert isinstance(gru, chainer.links.connection.gru.GRUBase)
        n_units, n_inputs = gru.W_r.W.data.shape
        fast_gru = cls(n_units, n_inputs)
        
        # move to the correct device
        if not isinstance(gru.W_r.W.data, numpy.ndarray):
            fast_gru = fast_gru.to_gpu(chainer.cuda.get_device(gru.W_r.W.data))
            
        fast_gru.initialize_with_classic_implementation(gru)
        return fast_gru
    
    def to_classic_GRU(self):
        """Convert to an object of type gru.GRU
        
        Useful for testing / converting models.
        """
        gru = chainer.links.connection.gru.GRU(self.n_units, self.n_inputs)
        
        # move to the correct device
        if not isinstance(self.W_r_z_h.W.data, numpy.ndarray):
            gru = gru.to_gpu(chainer.cuda.get_device(self.W_r_z_h.W.data))
        
        # input matrices
        gru.W_r.W.data[...] = self.W_r_z_h.W.data[:self.n_units]
        gru.W_z.W.data[...] = self.W_r_z_h.W.data[self.n_units: 2*self.n_units]
        gru.W.W.data[...] = self.W_r_z_h.W.data[2 * self.n_units:]
        
        # biases
        gru.W_r.b.data[...] = self.W_r_z_h.b.data[:self.n_units]
        gru.W_z.b.data[...] = self.W_r_z_h.b.data[self.n_units: 2*self.n_units]
        gru.W.b.data[...] = self.W_r_z_h.b.data[2 * self.n_units:]
        
        gru.U_r.b.data[...] = 0
        gru.U_z.b.data[...] = 0
        gru.U.b.data[...] = 0
        
        # hidden state matrices
        gru.U_r.W.data[...] = self.U_r_z.W.data[:self.n_units]
        gru.U_z.W.data[...] = self.U_r_z.W.data[self.n_units:]

        gru.U.W.data[...] = self.U.W.data[...]
        
        return gru        
        

class GRU(GRUBase):

    """Stateless Gated Recurrent Unit function (GRU).

    GRU function has six parameters :math:`W_r`, :math:`W_z`, :math:`W`,
    :math:`U_r`, :math:`U_z`, and :math:`U`. All these parameters are
    :math:`n \\times n` matrices, where :math:`n` is the dimension of
    hidden vectors.

    Given two inputs a previous hidden vector :math:`h` and an input vector
    :math:`x`, GRU returns the next hidden vector :math:`h'` defined as

    .. math::

       r &=& \\sigma(W_r x + U_r h), \\\\
       z &=& \\sigma(W_z x + U_z h), \\\\
       \\bar{h} &=& \\tanh(W x + U (r \\odot h)), \\\\
       h' &=& (1 - z) \\odot h + z \\odot \\bar{h},

    where :math:`\\sigma` is the sigmoid function, and :math:`\\odot` is the
    element-wise product.

    :class:`~chainer.links.GRU` does not hold the value of
    hidden vector :math:`h`. So this is *stateless*.
    Use :class:`~chainer.links.StatefulGRU` as a *stateful* GRU.

    Args:
        n_units(int): Dimension of hidden vector :math:`h`.
        n_inputs(int): Dimension of input vector :math:`x`. If ``None``,
            it is set to the same value as ``n_units``.

    See:
        - `On the Properties of Neural Machine Translation: Encoder-Decoder
          Approaches <http://www.aclweb.org/anthology/W14-4012>`_
          [Cho+, SSST2014].
        - `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
          Modeling <http://arxiv.org/abs/1412.3555>`_
          [Chung+NIPS2014 DLWorkshop].


    .. seealso:: :class:`~chainer.links.StatefulGRU`
    """

    def __call__(self, h, x):
        # We compute r_x, z_x and h_x simultaneously
        r_z_h_x = self.W_r_z_h(x)
        r_x, z_x, h_x = split_axis.split_axis(
            r_z_h_x, (self.n_units, self.n_units * 2), axis=1)

        # We compute r_h and z_h simultaneously
        r_z_h = self.U_r_z(h)
        r_h, z_h = split_axis.split_axis(r_z_h, (self.n_units,), axis=1)

        # finally we compute the output using the optimized functions
        return compute_output_GRU(
            z_x,
            z_h,
            h_x,
            h,
            self.U(sigmoid_a_plus_b_multiplied_by_c(r_x, r_h, h))
        )
