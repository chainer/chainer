#include "xchainer/routines/math.h"

namespace xchainer {
namespace routines {
namespace {

void AddImpl(const Array& lhs, const Array& rhs, const Array& out) {
    // TODO(sonots): dtype conversion
    CheckEqual(lhs.dtype(), rhs.dtype());
    CheckEqual(lhs.shape(), rhs.shape());

    auto lhs_backward_function = [](const Array& gout, const std::vector<GraphId>&) -> Array { return gout; };
    auto rhs_backward_function = lhs_backward_function;
    xchainer::internal::SetUpOpNodes("add", {lhs, rhs}, out, {lhs_backward_function, rhs_backward_function});

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

Array& IAdd(Array& lhs, const Array& rhs) { return AddAssignImpl(lhs, rhs); }

const Array& IAdd(const Array& lhs, const Array& rhs) { return AddAssignImpl(lhs, rhs); }

Array Add(const Array& lhs, const Array& rhs) {
    auto func = [](const Array& lhs, const Array& rhs) {
        Array out = Array::EmptyLike(lhs, lhs.device());
        AddImpl(lhs, rhs, out);
        return out;
    };

    if (lhs.shape() == rhs.shape()) {
        return func(lhs, rhs);
    }
    Shape result_shape = xchainer::internal::BroadcastShapes(lhs.shape(), rhs.shape());
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

Array& IMul(Array& lhs, const Array& rhs) { return MulAssignImpl(lhs, rhs); }

const Array& IMul(const Array& lhs, const Array& rhs) { return MulAssignImpl(lhs, rhs); }

Array Mul(const Array& lhs, const Array& rhs) {
    auto func = [](const Array& lhs, const Array& rhs) {
        Array out = Array::EmptyLike(lhs, lhs.device());
        MulImpl(lhs, rhs, out);
        return out;
    };

    if (lhs.shape() == rhs.shape()) {
        return func(lhs, rhs);
    }
    Shape result_shape = xchainer::internal::BroadcastShapes(lhs.shape(), rhs.shape());
    if (lhs.shape() == result_shape) {
        return func(lhs, rhs.BroadcastTo(result_shape));
    }
    if (rhs.shape() == result_shape) {
        return func(lhs.BroadcastTo(result_shape), rhs);
    }
    return func(lhs.BroadcastTo(result_shape), rhs.BroadcastTo(result_shape));
}

}  // namespace routines
}  // namespace xchainer
