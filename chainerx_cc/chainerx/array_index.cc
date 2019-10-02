#include "chainerx/array_index.h"

#include <algorithm>
#include <vector>

#include "chainerx/error.h"

namespace chainerx {
namespace internal {

std::vector<ArrayIndex> GetNormalizedArrayIndices(const std::vector<ArrayIndex>& indices, int8_t ndim) {
    auto it = std::find_if(indices.begin(), indices.end(), [](const ArrayIndex& index) { return index.tag() == ArrayIndexTag::kEllipsis; });
    if (it == indices.end()) {
        return indices;
    }
    if (std::find_if(std::next(it), indices.end(), [](const ArrayIndex& index) { return index.tag() == ArrayIndexTag::kEllipsis; }) !=
        indices.end()) {
        throw IndexError{"Indices can only have a single ellipsis."};
    }
    std::vector<ArrayIndex> result;
    int8_t ellipsized_ndim =
            ndim - static_cast<int8_t>(indices.size()) + 1 +
            std::count_if(indices.begin(), indices.end(), [](const ArrayIndex& index) { return index.tag() == ArrayIndexTag::kNewAxis; });
    if (ellipsized_ndim < 0) {
        throw IndexError{"Too many indices for array."};
    }
    result.reserve(ndim);
    std::copy(indices.begin(), it, std::back_inserter(result));
    std::fill_n(std::back_inserter(result), ellipsized_ndim, ArrayIndex{Slice{}});
    std::copy(std::next(it), indices.end(), std::back_inserter(result));
    return result;
}

}  // namespace internal
}  // namespace chainerx
