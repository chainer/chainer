#pragma once

#include <cstdint>
#include <vector>

namespace xchainer {
namespace internal {

bool IsAxesPermutation(const std::vector<int8_t>& axes, int8_t ndim);

}  // namespace internal
}  // namespace xchainer
