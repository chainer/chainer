#include "xchainer/axis.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace xchainer {
namespace internal {

bool IsAxesPermutation(const std::vector<int8_t>& axes, int8_t ndim) {
    assert(ndim >= 0);
    if (axes.size() != static_cast<size_t>(ndim)) {
        return false;
    }

    std::vector<int8_t> sorted_axes = axes;
    std::sort(sorted_axes.begin(), sorted_axes.end());
    for (int8_t i = 0; i < ndim; ++i) {
        if (sorted_axes[i] != i) {
            return false;
        }
    }
    return true;
}

}  // namespace internal
}  // namespace xchainer
