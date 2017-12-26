#include "check_backward.h"

#include <functional>
#include <iostream>

#include <gtest/gtest.h>

#include "dtype.h"
#include "error.h"
#include "shape.h"

namespace xchainer {
namespace {

template <typename T>
Array MakeArray(std::initializer_list<int64_t> shape, std::shared_ptr<void> data) {
    return {shape, TypeToDtype<T>, data};
}

template <typename T>
Array MakeArray(std::initializer_list<int64_t> shape, std::initializer_list<T> data) {
    auto a = std::make_unique<T[]>(data.size());
    std::copy(data.begin(), data.end(), a.get());
    return {shape, TypeToDtype<T>, std::move(a)};
}

  /*
TEST(CheckBackward, CorretcGradients) {
    Array x1 = MakeArray<float>({1}, {1});
    Array x2 = MakeArray<float>({1}, {1.5});
    Array gy = MakeArray<float>({1}, {0.2});
    Array e1 = MakeArray<float>({1}, {1e-3});
    Array e2 = MakeArray<float>({1}, {1e-3});

    auto func = [](const std::vector<Array>& inputs) -> std::vector<Array> {
        const Array& a1 = inputs[0];
        const Array& a2 = inputs[1];
        Array out = a1 + a2;
        return {out};
    };

    float atol = 1e-5;
    float rtol = 1e-4;

    EXPECT_NO_THROW(CheckBackwardComputation(func, {x1, x2}, {gy}, {e1, e2}, atol, rtol));
}

TEST(CheckBackward, IncorrectGradients) {
    Array x1 = MakeArray<float>({1}, {1});
    Array x2 = MakeArray<float>({1}, {1.5});
    Array gy = MakeArray<float>({1}, {0.2});
    Array e1 = MakeArray<float>({1}, {1e-3});
    Array e2 = MakeArray<float>({1}, {1e-3});

    auto func = [](const std::vector<Array>& inputs) -> std::vector<Array> {
        const Array& a1 = inputs[0];
        const Array& a2 = inputs[1];

        // TODO(hvy): Define an op that calculates incorrect gradients
        Array out = a1 + a2;
        return {out};
    };

    float atol = 1e-5;
    float rtol = 1e-4;

    EXPECT_THROW(CheckBackwardComputation(func, {x1, x2}, {gy}, {e1, e2}, atol, rtol), AssertionError);
}
*/

}  // namespace
}  // namespace xchainer
