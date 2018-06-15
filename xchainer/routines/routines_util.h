#pragma once

#include <functional>
#include <initializer_list>

#include "xchainer/array.h"
#include "xchainer/error.h"

namespace xchainer {
namespace internal {

void CheckNoInplaceWithRequiredGrad(const Array& out, std::initializer_list<std::reference_wrapper<const Array>> inputs) {
    if (!out.IsConstant()) {
        throw XchainerError{"In-place assignment to non-constant output array is not allowed."};
    }

    bool any_non_const = false;
    bool any_inplace = false;
    for (const Array& input : inputs) {
        any_non_const |= !input.IsConstant();
        any_inplace |= (out.body() == input.body());
    }

    if (any_non_const && any_inplace) {
        throw XchainerError{"In-place assignment that involves non-constant input arrays is not allowed."};
    }
}

}  // namespace internal
}  // namespace xchainer
