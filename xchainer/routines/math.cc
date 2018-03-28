#include "xchainer/routines/math.h"

#include "xchainer/routines/creation.h"
#include "xchainer/routines/util.h"

namespace xchainer {
namespace {

void AddImpl(const Array& lhs, const Array& rhs, const Array& out) {
    // TODO(sonots): dtype conversion
    CheckEqual(lhs.dtype(), rhs.dtype());
    CheckEqual(lhs.shape(), rhs.shape());

    auto lhs_backward_function = [](const Array& gout, const std::vector<GraphId>&) -> Array { return gout; };
    auto rhs_backward_function = lhs_backward_function;
    internal::SetUpOpNodes("add", {lhs, rhs}, out, {lhs_backward_function, rhs_backward_function});

    lhs.device().Add(lhs, rhs, out);
}

template <typename ArrayType>
ArrayType& AddAssignImpl(ArrayType& self, const Array& rhs) {
    auto func = [](ArrayType& lhs, const Array& rhs) -> ArrayType& {
        AddImpl(lhs, rhs, lhs);
        return lhs;
    };

    if (self.shape() == rhs.shape()) {
        return func(self, rhs);
    }
    Array rhs_broadcasted = rhs.BroadcastTo(self.shape());
    return func(self, rhs_broadcasted);
}

}  // namespace

namespace internal {

Array& IAdd(Array& lhs, const Array& rhs) { return AddAssignImpl(lhs, rhs); }

const Array& IAdd(const Array& lhs, const Array& rhs) { return AddAssignImpl(lhs, rhs); }

}  // namespace internal

Array Add(const Array& lhs, const Array& rhs) {
    auto func = [](const Array& lhs, const Array& rhs) {
        Array out = Array::EmptyLike(lhs, lhs.device());
        AddImpl(lhs, rhs, out);
        return out;
    };

    if (lhs.shape() == rhs.shape()) {
        return func(lhs, rhs);
    }
    Shape result_shape = internal::BroadcastShapes(lhs.shape(), rhs.shape());
    if (lhs.shape() == result_shape) {
        return func(lhs, rhs.BroadcastTo(result_shape));
    }
    if (rhs.shape() == result_shape) {
        return func(lhs.BroadcastTo(result_shape), rhs);
    }
    return func(lhs.BroadcastTo(result_shape), rhs.BroadcastTo(result_shape));
}

namespace {

void MulImpl(const Array& lhs, const Array& rhs, const Array& out) {
    // TODO(sonots): dtype conversion
    CheckEqual(lhs.dtype(), rhs.dtype());
    CheckEqual(lhs.shape(), rhs.shape());

    auto lhs_backward_function = [other = rhs](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) {
        return gout * other.AsConstant(graph_ids_to_stop_gradient);
    };
    auto rhs_backward_function = [other = lhs](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) {
        return gout * other.AsConstant(graph_ids_to_stop_gradient);
    };
    internal::SetUpOpNodes("mul", {lhs, rhs}, out, {lhs_backward_function, rhs_backward_function});

    lhs.device().Mul(lhs, rhs, out);
}

template <typename ArrayType>
ArrayType& MulAssignImpl(ArrayType& self, const Array& rhs) {
    auto func = [](ArrayType& lhs, const Array& rhs) -> ArrayType& {
        MulImpl(lhs, rhs, lhs);
        return lhs;
    };

    if (self.shape() == rhs.shape()) {
        return func(self, rhs);
    }
    Array rhs_broadcasted = rhs.BroadcastTo(self.shape());
    return func(self, rhs_broadcasted);
}

}  // namespace

namespace internal {

Array& IMultiply(Array& lhs, const Array& rhs) { return MulAssignImpl(lhs, rhs); }

const Array& IMultiply(const Array& lhs, const Array& rhs) { return MulAssignImpl(lhs, rhs); }

}  // namespace internal

Array Multiply(const Array& lhs, const Array& rhs) {
    auto func = [](const Array& lhs, const Array& rhs) {
        Array out = Array::EmptyLike(lhs, lhs.device());
        MulImpl(lhs, rhs, out);
        return out;
    };

    if (lhs.shape() == rhs.shape()) {
        return func(lhs, rhs);
    }
    Shape result_shape = internal::BroadcastShapes(lhs.shape(), rhs.shape());
    if (lhs.shape() == result_shape) {
        return func(lhs, rhs.BroadcastTo(result_shape));
    }
    if (rhs.shape() == result_shape) {
        return func(lhs.BroadcastTo(result_shape), rhs);
    }
    return func(lhs.BroadcastTo(result_shape), rhs.BroadcastTo(result_shape));
}

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
    std::vector<int8_t> out_axis;
    int8_t i_axis = 0;
    for (int8_t i = 0; i < a.ndim(); ++i) {
        if (i_axis < static_cast<int8_t>(sorted_axis.size()) && i == sorted_axis[i_axis]) {
            ++i_axis;
            if (keepdims) {
                out_shape_vec.push_back(int64_t{1});
            }
        } else {
            out_shape_vec.push_back(a.shape()[i]);
            out_axis.push_back(i);
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

}  // namespace xchainer
