#pragma once

#include <cstdint>
#include <vector>

namespace xchainer {
namespace internal {

std::vector<int8_t> GetSortedAxes(const std::vector<int8_t>& axis, int8_t ndim);

}  // namespace internal
}  // namespace xchainer
