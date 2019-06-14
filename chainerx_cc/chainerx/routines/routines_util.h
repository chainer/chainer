#pragma once

#include <functional>
#include <initializer_list>
#include <memory>

#include "chainerx/array.h"
#include "chainerx/error.h"
#include "chainerx/routines/creation.h"

namespace chainerx {
namespace internal {

// Checks for unsafe inplace operation on arrays.
// This functions throws ChainerxError if at least one of the following conditions is met.
// * The output array has any array nodes. Rewriting its data may affect other arrays in the graph.
// * At least one input array is identical to the output array, and the input array is to be backpropped in the current no/force backprop
// mode. In this case, if the backward computation of the operation retains and uses the input array, the retained input will not reflect
// the original nodes. There will be no problem if the input array is not to be retained, but it's difficult to distinguish such operations.
// Currently we take the safer strategy to forbid this case in any operations.
// TODO(niboshi): Flag the data of retained input/output arrays and detect in-place operation using the flag. If that will be implemented,
// the check in this function will not be needed.
inline void CheckNoUnsafeInplace(const Array& out, std::initializer_list<std::reference_wrapper<const Array>> inputs) {
    const std::shared_ptr<ArrayBody>& out_body = internal::GetArrayBody(out);
    if (!out_body->nodes().empty()) {
        throw ChainerxError{"In-place assignment to output array requiring grad is not allowed."};
    }

    bool any_input_grad_required = false;
    bool any_inplace = false;
    for (const Array& input : inputs) {
        any_input_grad_required |= input.IsBackpropRequired(AnyGraph{});
        any_inplace |= out_body == internal::GetArrayBody(input);
    }

    if (any_input_grad_required && any_inplace) {
        throw ChainerxError{"In-place assignment that involves input arrays requiring grad is not allowed."};
    }
}

// Makes view of output arrays of ForwardBackward implementations to avoid cyclic references since ForwardBackward may internally capture
// the output arrays.
template <size_t N>
void MakeViewForForwardBackwardOutput(std::array<Array, N>& outputs) {
    for (Array& output : outputs) {
        output = output.MakeView();
    }
}

inline void MakeViewForForwardBackwardOutput(Array& output) { output = output.MakeView(); }

// Called from Add, Subtract, Multiply, Divide, etc. to handle broadcasting.
template <typename Impl>
Array BroadcastBinary(Impl&& impl, const Array& x1, const Array& x2, Dtype dtype) {
    auto func = [&impl, dtype](const Array& x1, const Array& x2) -> Array {
        Array out = Empty(x1.shape(), dtype, x1.device());
        impl(x1, x2, out);
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

// Called from IAdd, ISubtract, IMultiply, IDivide, etc. to handle broadcasting.
template <typename Impl>
void BroadcastBinaryInplace(Impl&& impl, const Array& x1, const Array& x2) {
    internal::CheckNoUnsafeInplace(x1, {x1, x2});
    if (x1.shape() == x2.shape()) {
        impl(x1, x2, x1);
    } else {
        impl(x1, x2.BroadcastTo(x1.shape()), x1);
    }
}

template <typename Impl>
Array Binary(Impl&& impl, const Array& x1, Scalar x2, Dtype dtype) {
    Array out = Empty(x1.shape(), dtype, x1.device());
    impl(x1, x2, out);
    return out;
}

template <typename Impl>
Array Binary(Impl&& impl, Scalar x1, const Array& x2, Dtype dtype) {
    Array out = Empty(x2.shape(), dtype, x2.device());
    impl(x1, x2, out);
    return out;
}

template <typename Impl>
void BinaryInplace(Impl&& impl, const Array& x1, Scalar x2) {
    internal::CheckNoUnsafeInplace(x1, {x1});
    impl(x1, x2, x1);
}

}  // namespace internal
}  // namespace chainerx
