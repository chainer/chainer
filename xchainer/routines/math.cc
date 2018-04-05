#include "xchainer/routines/math.h"

#include <cstdint>
#include <numeric>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/util.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace {

void AddImpl(const Array& x1, const Array& x2, const Array& out) {
    // TODO(sonots): dtype conversion
    CheckEqual(x1.dtype(), x2.dtype());
    CheckEqual(x1.shape(), x2.shape());

    auto x1_backward_function = [](const Array& gout, const std::vector<GraphId>&) -> Array { return gout; };
    auto rhs_backward_function = x1_backward_function;
    internal::SetUpOpNodes("add", {x1, x2}, out, {x1_backward_function, rhs_backward_function});

    x1.device().Add(x1, x2, out);
}

template <typename ArrayType>
ArrayType& AddAssignImpl(ArrayType& self, const Array& x2) {
    auto func = [](ArrayType& x1, const Array& x2) -> ArrayType& {
        AddImpl(x1, x2, x1);
        return x1;
    };

    if (self.shape() == x2.shape()) {
        return func(self, x2);
    }
    Array rhs_broadcasted = x2.BroadcastTo(self.shape());
    return func(self, rhs_broadcasted);
}

}  // namespace

namespace internal {

Array& IAdd(Array& x1, const Array& x2) { return AddAssignImpl(x1, x2); }

const Array& IAdd(const Array& x1, const Array& x2) { return AddAssignImpl(x1, x2); }

}  // namespace internal

Array Add(const Array& x1, const Array& x2) {
    auto func = [](const Array& x1, const Array& x2) {
        Array out = Array::EmptyLike(x1, x1.device());
        AddImpl(x1, x2, out);
        return out;
    };

    if (x1.shape() == x2.shape()) {
        return func(x1, x2);
    }
    Shape result_shape = internal::BroadcastShapes(x1.shape(), x2.shape());
    if (x1.shape() == result_shape) {
        return func(x1, x2.BroadcastTo(result_shape));
    }
    if (x2.shape() == result_shape) {
        return func(x1.BroadcastTo(result_shape), x2);
    }
    return func(x1.BroadcastTo(result_shape), x2.BroadcastTo(result_shape));
}

namespace {

void SubtractImpl(const Array& x1, const Array& x2, const Array& out) {
    // TODO(niboshi): dtype conversion
    CheckEqual(x1.dtype(), x2.dtype());
    CheckEqual(x1.shape(), x2.shape());

    auto x1_backward_function = [](const Array& gout, const std::vector<GraphId>&) -> Array { return gout; };
    auto rhs_backward_function = [](const Array& gout, const std::vector<GraphId>&) -> Array {
        // TODO(niboshi): Use unary negate
        return -1 * gout;
    };
    internal::SetUpOpNodes("subtract", {x1, x2}, out, {x1_backward_function, rhs_backward_function});

    x1.device().Subtract(x1, x2, out);
}

template <typename ArrayType>
ArrayType& SubtractAssignImpl(ArrayType& self, const Array& x2) {
    auto func = [](ArrayType& x1, const Array& x2) -> ArrayType& {
        SubtractImpl(x1, x2, x1);
        return x1;
    };

    if (self.shape() == x2.shape()) {
        return func(self, x2);
    }
    Array rhs_broadcasted = x2.BroadcastTo(self.shape());
    return func(self, rhs_broadcasted);
}

}  // namespace

namespace internal {

Array& ISubtract(Array& x1, const Array& x2) { return SubtractAssignImpl(x1, x2); }

const Array& ISubtract(const Array& x1, const Array& x2) { return SubtractAssignImpl(x1, x2); }

}  // namespace internal

Array Subtract(const Array& x1, const Array& x2) {
    auto func = [](const Array& x1, const Array& x2) {
        Array out = Array::EmptyLike(x1, x1.device());
        SubtractImpl(x1, x2, out);
        return out;
    };

    if (x1.shape() == x2.shape()) {
        return func(x1, x2);
    }
    Shape result_shape = internal::BroadcastShapes(x1.shape(), x2.shape());
    if (x1.shape() == result_shape) {
        return func(x1, x2.BroadcastTo(result_shape));
    }
    if (x2.shape() == result_shape) {
        return func(x1.BroadcastTo(result_shape), x2);
    }
    return func(x1.BroadcastTo(result_shape), x2.BroadcastTo(result_shape));
}

namespace {

void MultiplyImpl(const Array& x1, const Array& x2, const Array& out) {
    // TODO(sonots): dtype conversion
    CheckEqual(x1.dtype(), x2.dtype());
    CheckEqual(x1.shape(), x2.shape());

    auto x1_backward_function = [other = x2](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) {
        return gout * other.AsConstant(graph_ids_to_stop_gradient);
    };
    auto rhs_backward_function = [other = x1](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) {
        return gout * other.AsConstant(graph_ids_to_stop_gradient);
    };
    internal::SetUpOpNodes("mul", {x1, x2}, out, {x1_backward_function, rhs_backward_function});

    x1.device().Multiply(x1, x2, out);
}

template <typename ArrayType>
ArrayType& MultiplyAssignImpl(ArrayType& self, const Array& x2) {
    auto func = [](ArrayType& x1, const Array& x2) -> ArrayType& {
        MultiplyImpl(x1, x2, x1);
        return x1;
    };

    if (self.shape() == x2.shape()) {
        return func(self, x2);
    }
    Array rhs_broadcasted = x2.BroadcastTo(self.shape());
    return func(self, rhs_broadcasted);
}

}  // namespace

namespace internal {

Array& IMultiply(Array& x1, const Array& x2) { return MultiplyAssignImpl(x1, x2); }

const Array& IMultiply(const Array& x1, const Array& x2) { return MultiplyAssignImpl(x1, x2); }

}  // namespace internal

Array Multiply(const Array& x1, const Array& x2) {
    auto func = [](const Array& x1, const Array& x2) {
        Array out = Array::EmptyLike(x1, x1.device());
        MultiplyImpl(x1, x2, out);
        return out;
    };

    if (x1.shape() == x2.shape()) {
        return func(x1, x2);
    }
    Shape result_shape = internal::BroadcastShapes(x1.shape(), x2.shape());
    if (x1.shape() == result_shape) {
        return func(x1, x2.BroadcastTo(result_shape));
    }
    if (x2.shape() == result_shape) {
        return func(x1.BroadcastTo(result_shape), x2);
    }
    return func(x1.BroadcastTo(result_shape), x2.BroadcastTo(result_shape));
}

Array Multiply(const Array& x1, Scalar x2) {
    Array out = Array::EmptyLike(x1, x1.device());
    x1.device().Multiply(x1, x2, out);

    auto backward_function = [x2](const Array& gout, const std::vector<GraphId>&) { return gout * x2; };
    internal::SetUpOpNodes("mul_scalar", {x1}, out, {backward_function});

    return out;
}

Array Multiply(Scalar x1, const Array& x2) { return Multiply(x2, x1); }

Array Sum(const Array& a, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
    std::vector<int8_t> sorted_axis;
    if (axis.has_value()) {
        sorted_axis = internal::GetSortedAxes(*axis, a.ndim());
    } else {
        // Fill with all axes
        sorted_axis.resize(a.ndim());
        std::iota(sorted_axis.begin(), sorted_axis.end(), int8_t{0});
    }

    // Calculate output shape
    std::vector<int64_t> out_shape_vec;
    out_shape_vec.reserve(a.ndim());
    int8_t i_axis = 0;
    for (int8_t i = 0; i < a.ndim(); ++i) {
        if (i_axis < static_cast<int8_t>(sorted_axis.size()) && i == sorted_axis[i_axis]) {
            ++i_axis;
            if (keepdims) {
                out_shape_vec.push_back(int64_t{1});
            }
        } else {
            out_shape_vec.push_back(a.shape()[i]);
        }
    }

    Array out;
    if (keepdims) {
        // Set reduced strides of the output array to 0
        Shape out_shape{out_shape_vec.begin(), out_shape_vec.end()};
        Strides contiguous_strides{out_shape, a.dtype()};
        std::vector<int64_t> out_strides_vec(contiguous_strides.begin(), contiguous_strides.end());
        for (int8_t i_axis : sorted_axis) {
            out_strides_vec[i_axis] = 0;
        }
        out = internal::Empty(out_shape, a.dtype(), Strides{out_strides_vec.begin(), out_strides_vec.end()}, a.device());
    } else {
        out = Empty({out_shape_vec.begin(), out_shape_vec.end()}, a.dtype(), a.device());
    }
    a.device().Sum(a, sorted_axis, out);

    auto backward_function = [ sorted_axis, in_shape = a.shape(), keepdims ](const Array& gout, const std::vector<GraphId>&) {
        assert(std::is_sorted(sorted_axis.begin(), sorted_axis.end()));

        if (!(in_shape.ndim() == 0 || sorted_axis.empty() || keepdims)) {
            std::vector<int64_t> out_shape_broadcastable{gout.shape().begin(), gout.shape().end()};
            for (auto axis : sorted_axis) {
                out_shape_broadcastable.insert(out_shape_broadcastable.begin() + axis, 1);
            }
            return gout.Reshape({out_shape_broadcastable.begin(), out_shape_broadcastable.end()}).BroadcastTo(in_shape);
        }
        return gout.BroadcastTo(in_shape);
    };
    internal::SetUpOpNodes("sum", {a}, out, {backward_function});

    return out;
}

namespace {

// Calculates: x1 < x2 ? pos : neg
// Can only differentiate with respect to neg.
Array IfLessElse(const Array& x1, Scalar x2, Scalar pos, const Array& neg) {
    Array out = Array::EmptyLike(x1, x1.device());
    x1.device().IfLessElse(x1, x2, pos, neg, out);

    auto backward_function = [x1, x2](const Array& gout, const std::vector<GraphId>&) {
        return IfLessElse(x1, x2, Scalar{0, gout.dtype()}, gout);
    };
    internal::SetUpOpNodes("if-less-else", {neg}, out, {backward_function});

    return out;
}

}  // namespace

Array Maximum(const Array& x1, Scalar x2) {
    return IfLessElse(x1, x2, x2, x1);  // x1 < x2 ? x2 : x1
}

Array Maximum(Scalar x1, const Array& x2) { return Maximum(x2, x1); }

Array Log(const Array& x) {
    Array out = Array::EmptyLike(x, x.device());
    x.device().Log(x, out);
    // TODO(niboshi): Implement backward
    return out;
}

}  // namespace xchainer
