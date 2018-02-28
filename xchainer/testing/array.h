#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <memory>
#include <utility>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace testing {

template <typename T, size_t N>
Array MakeArray(const Shape& shape, const std::array<T, N>& data) {
    assert(static_cast<size_t>(shape.GetTotalSize()) == N);
    auto a = std::make_unique<T[]>(N);
    std::copy(data.begin(), data.end(), a.get());
    return Array::FromBuffer(shape, TypeToDtype<T>, std::move(a));
}

template <typename T>
Array MakeArray(const Shape& shape, std::initializer_list<T> data) {
    assert(static_cast<size_t>(shape.GetTotalSize()) == data.size());
    auto a = std::make_unique<T[]>(data.size());
    std::copy(data.begin(), data.end(), a.get());
    return Array::FromBuffer(shape, TypeToDtype<T>, std::move(a));
}

}  // namespace testing
}  // namespace xchainer
