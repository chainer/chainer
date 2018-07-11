import functools

import six

from chainer.backends import cuda


def mulexp(xs, init=None):
    if init is not None:
        return functools.reduce('{} * {}'.format, xs, init)
    else:
        return functools.reduce('{} * {}'.format, xs)


def andexp(xs, init=None):
    if init is not None:
        return functools.reduce('{} && {}'.format, xs, init)
    else:
        return functools.reduce('{} && {}'.format, xs)


def muladdexp(xs, ys, init=None):
    def aux(exp, arg):
        x, y = arg
        return '({} + {} * {})'.format(y, x, exp)
    if init is not None:
        return functools.reduce(aux, six.moves.zip(xs, ys), init)
    else:
        return functools.reduce(aux, six.moves.zip(xs, ys))


def map_(fn, *lst):
    # For py2/py3 compatibility.
    return list(map(fn, *lst))


def succ_sublists(xs):
    # Returns successive sublists of xs.
    return [xs[i:] for i in six.moves.range(len(xs))]


def vars(prefix, n):
    return ['{}_{}'.format(prefix, i) for i in six.moves.range(n)]


class Writer(object):

    def __init__(self):
        self._indent = 0
        self._lines = []

    def write(self, line, indent=None):
        if indent == 'dec' or indent == 'decinc':
            self._indent -= 1
        self._lines.append('  ' * self._indent + line)
        if indent == 'inc' or indent == 'decinc':
            self._indent += 1

    def get(self):
        return '\n'.join(self._lines)


#
# im2col

class Im2colNDKernel(object):

    def _in_params(self, ds, outs, ks, ss, ps, dilate):
        # 2D: raw T img, int32 d_0, int32 d_1, int32 out_0, int32 out_1,
        #     int32 k_0, int32 k_1, int32 s_0, int32 s_1, int32 p_0, int32 p_1,
        #     int32 di_0, int32 di_1
        def aux(x):
            return 'int32 {}'.format(x)
        return ', '.join(
            ['raw T img'] + map_(aux, ds + outs + ks + ss + ps + dilate))

    def _out_params(self):
        return 'T col'

    def _compile_c0(self, outs, ks):
        # 2D: int c0 = i / (k_0 * k_1 * out_0 * out_1)
        return ['int c0 = i / ({});'.format(mulexp(ks + outs))]

    def _compile_kx(self, ndim, outs, ks):
        # 2D: int kx_0 = i / (k_1 * out_0 * out_1) % k_0;
        #     int kx_1 = i / (out_0 * out_1) % k_1;
        def aux(kx, xs):
            head = xs[0]
            tail = xs[1:] + outs
            if tail:
                return 'int {} = i / ({}) % {};'.format(kx, mulexp(tail), head)
            else:
                return 'int {} = i % {};'.format(kx, head)
        kxs = vars('kx', ndim)
        kx_decls = map_(aux, kxs, succ_sublists(ks))
        return kx_decls, kxs

    def _compile_out_x(self, ndim, outs):
        # 2D: int out_x0 = i / (out_1) % out_0;
        #     int out_x1 = i % out_1;
        def aux(out_x, xs):
            head = xs[0]
            tail = xs[1:]
            if tail:
                return 'int {} = i / ({}) % {};'.format(
                    out_x, mulexp(tail), head)
            else:
                return 'int {} = i % {};'.format(out_x, head)
        out_xs = vars('out_x', ndim)
        out_x_decls = map_(aux, out_xs, succ_sublists(outs))
        return out_x_decls, out_xs

    def _compile_main(self, ndim, ds, ks, ss, ps, dilate, kxs, out_xs):
        # 2D: int in_0 = kx_0 * di_0 + out_x_0 * s_0 - p_0;
        #     int in_1 = kx_1 * di_1 + out_x_1 * s_1 - p_1;
        #     if (0 <= in_0 && in_0 < d_0 && 0 <= in_1 && in_1 < d_1) {
        #       int idx_0 = in_0 + d_0 * c0;
        #       int idx_1 = in_1 + d_1 * idx_0;
        #       col = img[idx_1];
        #     } else {
        #       col = (T)0;
        #     }
        w = Writer()

        ins = vars('in', ndim)
        for _in, kx, out_x, s, p, di in six.moves.zip(ins, kxs, out_xs,
                                                      ss, ps, dilate):
            target = 'int {} = {} * {} + {} * {} - {};'
            w.write(target.format(_in, kx, di, out_x, s, p))

        def rel_aux(_in, d):
            return '0 <= {} && {} < {}'.format(_in, _in, d)
        w.write(
            'if ({}) {{'.format(andexp(map_(rel_aux, ins, ds))), indent='inc')

        idxs = vars('idx', ndim)
        idx0s = ['c0'] + idxs[:-1]
        for idx, _in, d, idx0 in six.moves.zip(idxs, ins, ds, idx0s):
            w.write('int {} = {} + {} * {};'.format(idx, _in, d, idx0))

        w.write('col = img[{}];'.format(idxs[-1]))
        w.write('} else {', indent='decinc')
        w.write('col = (T)0;')
        w.write('}', indent='dec')

        return [w.get()]

    def _operation(self, ndim, ds, outs, ks, ss, ps, dilate):
        c0 = self._compile_c0(outs, ks)
        kx, kxs = self._compile_kx(ndim, outs, ks)
        out_x, out_xs = self._compile_out_x(ndim, outs)
        main = self._compile_main(ndim, ds, ks, ss, ps, dilate, kxs, out_xs)
        return '\n'.join(c0 + kx + out_x + main)

    def _generate(self, ndim):
        ds = vars('d', ndim)
        outs = vars('out', ndim)
        ks = vars('k', ndim)
        ss = vars('s', ndim)
        ps = vars('p', ndim)
        dilate = vars('di', ndim)

        in_params = self._in_params(ds, outs, ks, ss, ps, dilate)
        out_params = self._out_params()
        operation = self._operation(ndim, ds, outs, ks, ss, ps, dilate)
        name = name = 'im2col_{}d'.format(ndim)
        return in_params, out_params, operation, name

    @staticmethod
    @cuda.memoize()
    def generate(ndim):
        return _im2col_nd_kernel._generate(ndim)


_im2col_nd_kernel = Im2colNDKernel()


#
# col2im

class Col2imNDKernel(object):

    def _in_params(self, ds, outs, ks, ss, ps, dilate):
        # 2D: raw T col, int32 d_0, int32 d_1, int32 out_0, int32 out_1,
        #     int32 k_0, int32 k_1, int32 s_0, int32 s_1, int32 p_0, int32 p_1,
        #     int32 di_0, int32 di_1
        def aux(x):
            return 'int32 {}'.format(x)
        return ', '.join(
            ['raw T col'] + map_(aux, ds + outs + ks + ss + ps + dilate))

    def _out_params(self):
        return 'T img'

    def _compile_c0(self, ds):
        # 2D: int c0 = i / (d_0 * d_1);
        return ['int c0 = i / ({});'.format(mulexp(ds))]

    def _compile_x(self, ndim, ds):
        # 2D: int x_0 = i / (d_1) % d_0;
        #     int x_1 = i % d_1;
        def aux(x, ds):
            head = ds[0]
            tail = ds[1:]
            if tail:
                return 'int {} = i / ({}) % {};'.format(
                    x, mulexp(tail), head)
            else:
                return 'int {} = i % {};'.format(x, head)
        xs = vars('x', ndim)
        x_decls = map_(aux, xs, succ_sublists(ds))
        return x_decls, xs

    def _compile_loop(self, ndim, outs, ks, ss, ps, xs, dilate):
        # 2D: for (int kx_0 = 0; kx_0 < k_0; ++kx_0) {
        #     int out_x_0 = x_0 + p_0 - kx_0 * di_0;
        #     if (0 > out_x_0 || out_x_0 >= out_0 * s_0) continue;
        #     if (out_x_0 % s_0 != 0) continue;
        #     out_x_0 /= s_0;
        #     for (int kx_1 = 0; kx_1 < k_1; ++kx_1) {
        #       int out_x_1 = x_1 + p_1 - kx_1 * di_1;
        #       if (0 > out_x_1 || out_x_1 >= out_1 * s_1) continue;
        #       if (out_x_1 % s_1 != 0) continue;
        #       out_x_1 /= s_1;
        #       ... Main-part here ...
        #     }
        #   }
        #   ... After-part here ...
        def _loop_main(main, ndim, ks, ss):
            w = Writer()

            # Loop openings.
            out_xs = vars('out_x', ndim)
            kxs = vars('kx', ndim)
            for out, out_x, kx, s, p, x, k, di in six.moves.zip(
                    outs, out_xs, kxs, ss, ps, xs, ks, dilate):
                w.write('for (int {} = 0; {} < {}; ++{}) {{'.format(
                    kx, kx, k, kx), indent='inc')
                w.write('int {} = {} + {} - {} * {};'.format(
                    out_x, x, p, kx, di))
                w.write('if (0 > {} || {} >= {} * {}) continue;'.format(
                    out_x, out_x, out, s))
                w.write('if ({} % {} != 0) continue;'.format(out_x, s))
                w.write('{} /= {};'.format(out_x, s))

            # Main-part.
            for l in main(ks, kxs, out_xs).split('\n'):
                w.write(l)

            # Loop closings.
            for _ in out_xs:
                w.write('}', indent='dec')

            return [w.get()]

        return _loop_main

    def _compile_procedure(self, outs, xs):
        # 2D: val = val + col[
        #     (out_x_1 + out_1 * (out_x_0 + out_0 *
        #     (kx_1 + k_1 * (kx_0 + k_0 * c0))))];
        def _main(ks, kxs, out_xs):
            index = muladdexp(outs, out_xs, muladdexp(ks, kxs, 'c0'))
            return 'val = val + col[{}];'.format(index)
        before = ['T val = 0;']
        after = ['img = val;']
        return before, _main, after

    def _operation(self, ndim, ds, outs, ks, ss, ps, dilate):
        c0 = self._compile_c0(ds)
        x, xs = self._compile_x(ndim, ds)
        loop_main = self._compile_loop(ndim, outs, ks, ss, ps, xs, dilate)
        before, main, after = self._compile_procedure(outs, xs)
        return '\n'.join(
            c0 + x + before + loop_main(main, ndim, ks, ss) + after)

    def _generate(self, ndim):
        ds = vars('d', ndim)
        outs = vars('out', ndim)
        ks = vars('k', ndim)
        ss = vars('s', ndim)
        ps = vars('p', ndim)
        dilate = vars('di', ndim)

        in_params = self._in_params(ds, outs, ks, ss, ps, dilate)
        out_params = self._out_params()
        operation = self._operation(ndim, ds, outs, ks, ss, ps, dilate)
        name = 'col2im_{}d'.format(ndim)
        return in_params, out_params, operation, name

    @staticmethod
    @cuda.memoize()
    def generate(ndim):
        return _col2im_nd_kernel._generate(ndim)


_col2im_nd_kernel = Col2imNDKernel()
