#include "chainerx/routines/activation.h"

#include <cmath>
#include <numeric>
#include <utility>
#include <iostream>
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
    Array zero = ZerosLike(x_cast, x_cast.device());
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
    Array zero = ZerosLike(x_cast, x_cast.device());
    return Where(x_cast >= zero, x_cast, slope * x_cast);
}

std::vector<Array> _extract_gates(Array x, int64_t n_splits) {
    StackVector<int64_t, kMaxNdim> shape_vec;
    shape_vec.push_back(x.shape()[0]);
    shape_vec.push_back(n_splits);
    shape_vec.push_back(static_cast<int>(x.shape()[1] / n_splits));
    for (int i = 2; i < x.ndim(); i++) {
        shape_vec.push_back(x.shape()[i]);
    }
    Shape shape{shape_vec};
    Array x_r = Reshape(x, shape);
    std::vector<Array> x_split = Split(x_r, n_splits, 1);
    return x_split;
}

std::vector<Array> TreeLstm(std::vector<Array> arrays) {
    uint n_ary = arrays.size() - 1;
    std::vector<Array> gates = _extract_gates(arrays[arrays.size() - 1], 3 + n_ary);

    Array a = Squeeze(gates[0]);
    Array i = Squeeze(gates[1]);
    Array o = Squeeze(gates[2]);
    std::vector<Array> fs;
    for (uint i = 3; i < gates.size(); i++) {
        fs.push_back(Squeeze(gates[i]));
    }
    Array a_ = Tanh(a);
    Array i_ = Sigmoid(i);
    Array o_ = Sigmoid(o);
    std::vector<Array> fs_s{};
    for (uint i = 0; i < fs.size(); i++) {
        fs_s.push_back(Sigmoid(fs[i]));
    }
    std::vector<Array> sum;
    sum.push_back(arrays[0] * fs_s[0]);
    for (uint i = 1; i < arrays.size() - 1; i++) {
        sum.push_back(sum[i-1] + arrays[i] * fs_s[i]);
    }
    Array c = a_ * i_ + sum[sum.size() - 1];
    Array h = o_ * Tanh(c);
    return {c, h};
}

}  // namespace chainerx
