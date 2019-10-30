#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "chainerx/error.h"
#include "chainerx/slice.h"

namespace chainerx {

enum class ArrayIndexTag {
    kSingleElement = 1,
    kSlice,
    kNewAxis,
    kEllipsis,
};

// Index out-of-bounds handling modes for
// numpy compatible take, choose, put routines
enum class IndexBoundsMode {
    kDefault,  // Default (raise for native, wrap for cuda)
    kRaise,  // Raise exception on OOB
    kWrap,  // Use the index modulo size of the dimension
    kClip  // Clip the index, negative values are always 0
};

class NewAxis {};

class Ellipsis {};

class ArrayIndex {
public:
    ArrayIndex(int64_t index) : tag_{ArrayIndexTag::kSingleElement}, index_{index} {}  // NOLINT
    ArrayIndex(Slice slice) : tag_{ArrayIndexTag::kSlice}, slice_{std::move(slice)} {}  // NOLINT
    ArrayIndex(NewAxis /*new_axis*/) : tag_{ArrayIndexTag::kNewAxis} {}  // NOLINT
    ArrayIndex(Ellipsis /*ellipsis*/) : tag_{ArrayIndexTag::kEllipsis} {}  // NOLINT

    ArrayIndexTag tag() const { return tag_; }

    int64_t index() const {
        if (tag_ != ArrayIndexTag::kSingleElement) {
            throw ChainerxError{"Array index is not a single element."};
        }
        return index_;
    }

    Slice slice() const {
        if (tag_ != ArrayIndexTag::kSlice) {
            throw ChainerxError{"Array index is not a slice."};
        }
        return slice_;
    }

private:
    ArrayIndexTag tag_;
    int64_t index_{};
    Slice slice_;
};

namespace internal {

std::vector<ArrayIndex> GetNormalizedArrayIndices(const std::vector<ArrayIndex>& indices, int8_t ndim);

}  // namespace internal

}  // namespace chainerx
