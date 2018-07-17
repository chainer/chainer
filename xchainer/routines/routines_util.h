#pragma once

#include <functional>
#include <initializer_list>

#include "xchainer/array.h"
#include "xchainer/error.h"

namespace xchainer {
namespace internal {

inline void CheckNoInplaceWithRequiredGrad(const Array& out, std::initializer_list<std::reference_wrapper<const Array>> inputs) {
    if (out.IsGradRequired(AnyGraph{})) {
        throw XchainerError{"In-place assignment to output array requiring grad is not allowed."};
    }

    bool any_grad_required = false;
    bool any_inplace = false;
    for (const Array& input : inputs) {
        any_grad_required |= input.IsGradRequired(AnyGraph{});
        any_inplace |= (out.body() == input.body());
    }

    if (any_grad_required && any_inplace) {
        throw XchainerError{"In-place assignment that involves input arrays requiring grad is not allowed."};
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
}  // namespace xchainer
