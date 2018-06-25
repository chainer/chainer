#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

namespace xchainer {

class ArrayNode;

namespace internal {

// Data holder of Array.
//
// C++ Array and Python bindings both share ArrayBody through shared_ptr. C++ Array provides the value-based semantics of Array in C++,
// while Python Array provides the reference-based semantics, which is more natural in Python.
//
// The current design requires a subtle overhead on converting between C++ Array and Python Array (due to reference counting), which is
// currently considered to be ignorable compared to other Python operations.
//
// NOTE: This class should not be instantiated by any functions except those defined in array.cc. This class is still defined here so that
// the code is made simple and we can use inline access to each member from member accessor functions of Array.
class ArrayBody {
public:
    ArrayBody(
            const Shape& shape,
            const Strides& strides,
            Dtype dtype,
            Device& device,
            std::shared_ptr<void> data,
            int64_t offset,
            std::vector<std::shared_ptr<ArrayNode>> nodes = std::vector<std::shared_ptr<ArrayNode>>())
        : shape_{shape},
          strides_{strides},
          dtype_{dtype},
          device_{device},
          data_{std::move(data)},
          offset_{offset},
          nodes_{std::move(nodes)} {}

    ArrayBody(const ArrayBody&) = delete;
    ArrayBody& operator=(const ArrayBody&) = delete;

private:
    friend class ::xchainer::Array;

    Shape shape_;
    Strides strides_;
    Dtype dtype_;
    Device& device_;
    std::shared_ptr<void> data_;
    int64_t offset_;  // in bytes
    std::vector<std::shared_ptr<ArrayNode>> nodes_;
};

}  // namespace internal
}  // namespace xchainer
