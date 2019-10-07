#include "chainerx/routines/activation.h"

#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/enum.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/explog.h"
#include "chainerx/routines/hyperbolic.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {

namespace {

std::vector<Array> ExtractGates(const Array& x, int64_t n_splits, int64_t axis) {
    StackVector<int64_t, kMaxNdim> shape_vec;
    shape_vec.emplace_back(x.shape()[0]);
    if (axis == 1) {
        shape_vec.emplace_back(n_splits);
        shape_vec.emplace_back(static_cast<int64_t>(x.shape()[1] / n_splits));
    } else {
        shape_vec.emplace_back(static_cast<int64_t>(x.shape()[1] / n_splits));
        shape_vec.emplace_back(n_splits);
    }
    for (int64_t i = 2; i < x.ndim(); i++) {
        shape_vec.emplace_back(x.shape()[i]);
    }
    Shape shape{shape_vec};
    Array x_r = Reshape(x, shape);
    std::vector<Array> x_split = Split(x_r, n_splits, axis);
    return x_split;
}

}  // namespace

Array ClippedRelu(const Array& x, Scalar z) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    return Minimum(Maximum(0, x_cast), z);
}

Array CRelu(const Array& x, int8_t axis) {
    // TODO(aksub99): Optimize implementation to use a single memory allocation.
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    std::vector<Array> c{x_cast, Negative(x_cast)};
    Array concat = Concatenate(c, axis);
    return Relu(concat);
}

Array Elu(const Array& x, double alpha) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    // TODO(aksub99): Replace x > zero with x > 0 when operator > supports scalars.
    Array zero = Zeros({}, x_cast.dtype(), x_cast.device());
    return Where(x_cast > zero, x_cast, alpha * Expm1(x_cast));
}

Array Sigmoid(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    return Reciprocal(1 + Exp(-x_cast));
}

Array Relu(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    return Maximum(0, x_cast);
}

Array LeakyRelu(const Array& x, Scalar slope) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    // TODO(hamaji): Replace x >= zero with x >= 0 when operator >= supports scalars.
    Array zero = Zeros({}, x_cast.dtype(), x_cast.device());
    return Where(x_cast >= zero, x_cast, slope * x_cast);
}

std::vector<Array> TreeLstm(std::vector<Array> arrays) {
    size_t n_ary = arrays.size() - 1;
    std::vector<Array> gates = ExtractGates(arrays[arrays.size() - 1], 3 + n_ary, 1);

    Array a = Squeeze(gates[0]);
    Array i = Squeeze(gates[1]);
    Array o = Squeeze(gates[2]);
    std::vector<Array> fs;
    for (size_t i = 3; i < gates.size(); i++) {
        fs.emplace_back(Squeeze(gates[i]));
    }
    Array a_ = Tanh(a);
    Array i_ = Sigmoid(i);
    Array o_ = Sigmoid(o);
    std::vector<Array> fs_s{};
    fs_s.reserve(fs.size());
    for (const auto& f : fs) {
        fs_s.emplace_back(Sigmoid(f));
    }
    std::vector<Array> sum;
    sum.emplace_back(arrays[0] * fs_s[0]);
    for (size_t i = 1; i < arrays.size() - 1; i++) {
        sum.emplace_back(sum[i - 1] + arrays[i] * fs_s[i]);
    }
    Array c = a_ * i_ + sum[sum.size() - 1];
    Array h = o_ * Tanh(c);
    return {c, h};
}

std::vector<Array> SLstm(const Array& c_prev1, const Array& c_prev2, const Array& x1, const Array& x2) {
    std::vector<Array> x1_gates = ExtractGates(x1, 4, 2);
    std::vector<Array> x2_gates = ExtractGates(x2, 4, 2);
    Array a1 = Squeeze(x1_gates[0]);
    Array i1 = Squeeze(x1_gates[1]);
    Array f1 = Squeeze(x1_gates[2]);
    Array o1 = Squeeze(x1_gates[3]);

    Array a2 = Squeeze(x2_gates[0]);
    Array i2 = Squeeze(x2_gates[1]);
    Array f2 = Squeeze(x2_gates[2]);
    Array o2 = Squeeze(x2_gates[3]);

    a1 = Tanh(a1);
    i1 = Sigmoid(i1);
    f1 = Sigmoid(f1);

    a2 = Tanh(a2);
    i2 = Sigmoid(i2);
    f2 = Sigmoid(f2);

    Array o = Sigmoid(o1 + o2);
    Array c = a1 * i1 + a2 * i2 + f1 * c_prev1 + f2 * c_prev2;
    Array h = o * Tanh(c);

    return {c, h};
}

Array Softplus(const Array& x, double beta) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    double beta_inv = 1.0 / beta;
    Array bx = beta * x_cast;
    Array y = (Maximum(bx, 0) + Log1p(Exp(-Fabs(bx)))) * beta_inv;
    return y;
}

}  // namespace chainerx
