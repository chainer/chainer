#include "chainerx/routines/loss.h"

#include "chainerx/array.h"
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/explog.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/logic.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/reduction.h"
#include "chainerx/scalar.h"

namespace chainerx {

Array AbsoluteError(const Array& x1, const Array& x2) { return Absolute(x1 - x2); }

Array SquaredError(const Array& x1, const Array& x2) { return Square(x1 - x2); }

Array GaussianKLDivergence(const Array& mean, const Array& ln_var) { return (Square(mean) + Exp(ln_var) - ln_var - 1) * 0.5; }

Array HuberLoss(const Array& x1, const Array& x2, Scalar delta) {
    Array a = x1 - x2;
    Array abs_a = Absolute(a);
    Array delta_array = chainerx::FullLike(a, delta, a.device());

    // TODO(kshitij12345) : use Array < Scalar when implemented.
    return Where(abs_a < delta_array, 0.5 * Square(a), delta * (abs_a - Scalar{0.5} * delta));
}

Array SigmoidCrossEntropy(const Array& x1, const Array& x2) {
    Array ignore_label = -OnesLike(x2, x2.device());
    Array ignore_mask = NotEqual(x2, ignore_label);
    return -(ignore_mask * (x1 * (x2 - (GreaterEqual(x1, ZerosLike(x1, x1.device()))).AsType(x1.dtype())) - Log1p(Exp(-Absolute(x1)))));
}

Array SoftmaxCrossEntropy(const Array& x1, const Array& x2) {
    if (x1.ndim() != 2) {
        throw DimensionError{"Input array must be 2 dimensional."};
    }
    if (x2.ndim() != 1) {
        throw DimensionError{"Target array must be 1 dimensional."};
    }
    if (x1.shape()[0] != x2.shape()[0]) {
        throw DimensionError{"x1.shape[0] must be equal to x2.shape[0]"};
    }
    Array score = LogSoftmax(x1, 1);
    Array mask = (x2.At({Slice{}, NewAxis{}}) == Arange(score.shape()[1], x2.dtype(), x1.device())).AsType(score.dtype());
    return -(score * mask).Sum({1});
}

Array Hinge(const Array& x, const Array& t, double norm) {
    if (x.ndim() != 2) {
        throw DimensionError{"Input array must be 2 dimensional."};
    }
    if (t.ndim() != 1) {
        throw DimensionError{"Target array must be 1 dimensional."};
    }
    if (x.shape()[0] != t.shape()[0]) {
        throw DimensionError{"x.shape[0] must be equal to t.shape[0]"};
    }

    int64_t num = x.shape()[1];
    Array one_minus_diff = Where(ExpandDims(t, 1) == Arange(num), 1 - x, 1 + x);
    Array bottom_diff = Maximum(0, one_minus_diff);

    return Power(bottom_diff, Scalar{norm});
}

}  // namespace chainerx
