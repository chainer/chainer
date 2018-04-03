#include "xchainer/routines/sorting.h"

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"

namespace xchainer {

Array ArgMax(const Array& a, const nonstd::optional<std::vector<int8_t>>& axis) {
    (void)axis;  // unused
    return a;
}

}  // namespace xchainer
