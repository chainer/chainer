#pragma once

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

namespace xchainer {
namespace internal {

std::vector<int8_t> GetSortedAxes(const std::vector<int8_t>& axis, int8_t ndim);
std::vector<int8_t> GetSortedAxesOrAll(const nonstd::optional<std::vector<int8_t>>& axis, int8_t ndim);

}  // namespace internal
}  // namespace xchainer
