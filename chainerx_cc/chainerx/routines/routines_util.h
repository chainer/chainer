#pragma once

#include <functional>
#include <initializer_list>
#include <memory>

#include "chainerx/array.h"
#include "chainerx/error.h"

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

}  // namespace internal
}  // namespace chainerx
