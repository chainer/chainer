#pragma once

#include <cstdint>
#include <utility>

#include "chainerx/error.h"
#include "chainerx/slice.h"

namespace chainerx {

enum class ArrayIndexTag {
    kSingleElement = 1,
    kSlice,
    kNewAxis,
};

class NewAxis {};

class ArrayIndex {
public:
    ArrayIndex(int64_t index) : tag_{ArrayIndexTag::kSingleElement}, index_{index} {}  // NOLINT
    ArrayIndex(Slice slice) : tag_{ArrayIndexTag::kSlice}, slice_{std::move(slice)} {}  // NOLINT
    ArrayIndex(NewAxis /*new_axis*/) : tag_{ArrayIndexTag::kNewAxis} {}  // NOLINT

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

}  // namespace chainerx
