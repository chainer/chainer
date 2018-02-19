from chainer import cuda
from chainer import function


_preamble = '''
template <typename T> __device__ T sigmoid(T x) {
    const T half = 0.5;
    return tanh(x * half) * half + half;
}
template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }
'''


class SRUFunction(function.Function):

    def forward(self, inputs):
        c_prev, x, u = inputs
        n_unit = x.shape[1]
        x_bar = u[:, 0:n_unit]
        f_in = u[:, n_unit:n_unit*2]
        r_in = u[:, n_unit*2:]
        c, h = cuda.elementwise(
            'T c_prev, T x, T x_bar, T f_in, T r_in, int32 n_unit',
            'T c, T h',
            '''
            float f = sigmoid(f_in);
            float r = sigmoid(r_in);
            c = f * c_prev + (1 - f) * x_bar;
            h = r * tanh(c) + (1 - r) * x;
            ''', 'sru_forward', preamble=_preamble)(
                c_prev, x, x_bar, f_in, r_in, n_unit)
        self.c = c
        return c, h

    def backward(self, inputs, grads):
        c_prev, x, u = inputs
        gc, gh = grads
        if gc is None:
            gc = 0
        if gh is None:
            gh = 0
        n_unit = x.shape[1]
        x_bar = u[:, 0:n_unit]
        f_in = u[:, n_unit:n_unit*2]
        r_in = u[:, n_unit*2:]
        gc_prev = cuda.cupy.empty_like(c_prev)
        gx = cuda.cupy.empty_like(x)
        gu = cuda.cupy.empty_like(u)
        gx_bar = gu[:, 0:n_unit]
        gf_in = gu[:, n_unit:n_unit*2]
        gr_in = gu[:, n_unit*2:]
        cuda.elementwise(
            '''T c_prev, T x, T x_bar, T f_in, T r_in, int32 n_unit,
            T gc, T gh, T c''',
            'T gc_prev, T gx, T gx_bar, T gf_in, T gr_in',
            '''
            float f = sigmoid(f_in);
            float r = sigmoid(r_in);
            gx = gh * (1 - r);
            float tanh_c = tanh(c);
            float g = gh * r * grad_tanh(tanh_c) + gc;
            gc_prev = g * f;
            gx_bar = g * (1 - f);
            gf_in = g * grad_sigmoid(f) * (c_prev - x_bar);
            gr_in = gh * grad_sigmoid(r) * (tanh_c - x);
            ''', 'sru_backward', preamble=_preamble)(
                c_prev, x, x_bar, f_in, r_in, n_unit, gc, gh, self.c,
                gc_prev, gx, gx_bar, gf_in, gr_in)
        return gc_prev, gx, gu


def sru(c, x, u):
    return SRUFunction()(c, x, u)
