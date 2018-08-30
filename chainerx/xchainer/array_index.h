#pragma once

#include <cstdint>

#include "xchainer/error.h"
#include "xchainer/slice.h"

namespace xchainer {

enum class ArrayIndexTag {
    kSingleElement = 1,
    kSlice,
    kNewAxis,
};

class NewAxis {};

class ArrayIndex {
public:
    ArrayIndex(int64_t index) : tag_{ArrayIndexTag::kSingleElement}, index_{index} {}  // NOLINT(runtime/explicit)
    ArrayIndex(Slice slice) : tag_{ArrayIndexTag::kSlice}, slice_{slice} {}  // NOLINT(runtime/explicit)
    ArrayIndex(NewAxis) : tag_{ArrayIndexTag::kNewAxis} {}  // NOLINT(runtime/explicit)

    ArrayIndexTag tag() const { return tag_; }

    int64_t index() const {
        if (tag_ != ArrayIndexTag::kSingleElement) {
            throw XchainerError{"Array index is not a single element."};
        }
        return index_;
    }

    Slice slice() const {
        if (tag_ != ArrayIndexTag::kSlice) {
            throw XchainerError{"Array index is not a slice."};
        }
        return slice_;
    }

private:
    ArrayIndexTag tag_;
    int64_t index_{};
    Slice slice_;
};

}  // namespace xchainer
