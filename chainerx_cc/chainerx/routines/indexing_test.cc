#include "chainerx/routines/indexing.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/check_backward.h"
#include "chainerx/error.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/array_check.h"
#include "chainerx/testing/device_session.h"
#include "chainerx/testing/routines.h"
#include "chainerx/testing/threading.h"

namespace chainerx {
namespace {

class IndexingTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_THREAD_SAFE_P(IndexingTest, At) {
    using T = int32_t;
    Shape input_shape{2, 3, 1};
    Shape output_shape{1, 2, 1};
    std::vector<ArrayIndex> indices{-1, NewAxis{}, Slice{1, 3}};
    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    Array e = testing::BuildArray(output_shape).WithData<T>({4, 5});

    Run([&]() {
        testing::CheckForward(
                [&indices](const std::vector<Array>& xs) {
                    Array y = internal::At(xs[0], indices);
                    // Check if strides are 0 for newaxis.
                    EXPECT_EQ(0, y.strides()[0]);
                    EXPECT_NE(0, y.strides()[1]);
                    EXPECT_NE(0, y.strides()[2]);
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

// Index out of bounds
TEST_P(IndexingTest, InvalidAt1) {
    using T = int32_t;
    Shape input_shape{2, 3};
    std::vector<ArrayIndex> indices{0, 0, 0};
    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(internal::At(a, indices), DimensionError);
}

// Too large dimension
TEST_P(IndexingTest, InvalidAt2) {
    using T = int32_t;
    Shape input_shape{2, 3};
    std::vector<ArrayIndex> indices{2};
    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(internal::At(a, indices), DimensionError);
}

TEST_P(IndexingTest, AtBackward) {
    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                std::vector<ArrayIndex> indices{1, NewAxis{}, Slice{1, 3}};
                return {internal::At(xs[0], indices)};
            },
            {(*testing::BuildArray({2, 3}).WithData<float>({1.f, -1.f, 2.f, -2.f, 3.f, -3.f})).RequireGrad()},
            {Ones({1, 2}, Dtype::kFloat32)},
            {Full({2, 3}, 1e-3f)});
}

TEST_P(IndexingTest, AtDoubleBackward) {
    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                std::vector<ArrayIndex> indices{0, NewAxis{}, Slice{1, 3}};
                auto y = internal::At(xs[0], indices);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::BuildArray({2, 3}).WithData<float>({1.f, -1.f, 2.f, -2.f, 3.f, -3.f})).RequireGrad()},
            {Ones({1, 2}, Dtype::kFloat32).RequireGrad()},
            {Ones({2, 3}, Dtype::kFloat32)},
            {Full({2, 3}, 1e-3f), Full({1, 2}, 1e-3f)});
}

TEST_THREAD_SAFE_P(IndexingTest, TakeUInt8Indices) {
    using T = int8_t;
    Shape input_shape{2, 4};
    Shape indices_shape{2, 3};
    Shape output_shape{2, 2, 3};
    int8_t axis = 1;
    Array a = testing::BuildArray(input_shape).WithLinearData<T>().WithPadding(1);
    Array indices = testing::BuildArray(indices_shape).WithData<uint8_t>({0, 14, 3, 1, 0, 1});
    Array e = testing::BuildArray(output_shape).WithData<T>({0, 2, 3, 1, 0, 1, 4, 6, 7, 5, 4, 5});

    Run([&]() {
        testing::CheckForward(
                [&indices, &axis](const std::vector<Array>& xs) { return std::vector<Array>{Take(xs[0], indices, axis)}; }, {a}, {e});
    });
}

TEST_P(IndexingTest, TakeUInt8IndicesBackward) {
    using T = double;
    Shape input_shape{2, 4};
    Shape indices_shape{2, 3};
    Shape output_shape{2, 2, 3};
    int8_t axis = 1;
    Array a = (*testing::BuildArray(input_shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Array indices = testing::BuildArray(indices_shape).WithData<uint8_t>({0, 14, 3, 1, 0, 1});
    Array go = testing::BuildArray(output_shape).WithLinearData<T>(0.1, 0.1).WithPadding(1);
    Array eps = Full(input_shape, 1e-3, Dtype::kFloat64);

    CheckBackward(
            [&indices, axis](const std::vector<Array>& xs) -> std::vector<Array> { return {Take(xs[0], indices, axis)}; },
            {a},
            {go},
            {eps});
}

TEST_THREAD_SAFE_P(IndexingTest, TakeInt8Indices) {
    using T = int8_t;
    Shape input_shape{2, 4};
    Shape indices_shape{2, 3};
    Shape output_shape{2, 2, 3};
    int8_t axis = 1;
    Array a = testing::BuildArray(input_shape).WithLinearData<T>().WithPadding(1);
    Array indices = testing::BuildArray(indices_shape).WithData<int8_t>({0, 14, 3, 1, -10, 1});
    Array e = testing::BuildArray(output_shape).WithData<T>({0, 2, 3, 1, 2, 1, 4, 6, 7, 5, 6, 5});

    Run([&]() {
        testing::CheckForward(
                [&indices, &axis](const std::vector<Array>& xs) { return std::vector<Array>{Take(xs[0], indices, axis)}; }, {a}, {e});
    });
}

TEST_P(IndexingTest, TakeInt8IndicesBackward) {
    using T = double;
    Shape input_shape{2, 4};
    Shape indices_shape{2, 3};
    Shape output_shape{2, 2, 3};
    int8_t axis = 1;
    Array a = (*testing::BuildArray(input_shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Array indices = testing::BuildArray(indices_shape).WithData<int8_t>({0, 14, 3, 1, -10, 1});
    Array go = testing::BuildArray(output_shape).WithLinearData<T>(0.1, 0.1).WithPadding(1);
    Array eps = Full(input_shape, 1e-3, Dtype::kFloat64);

    CheckBackward(
            [&indices, axis](const std::vector<Array>& xs) -> std::vector<Array> { return {Take(xs[0], indices, axis)}; },
            {a},
            {go},
            {eps});
}

TEST_THREAD_SAFE_P(IndexingTest, TakeInt16Indices) {
    using T = int8_t;
    Shape input_shape{2, 4};
    Shape indices_shape{2, 3};
    Shape output_shape{2, 2, 3};
    int8_t axis = 1;
    Array a = testing::BuildArray(input_shape).WithLinearData<T>().WithPadding(1);
    Array indices = testing::BuildArray(indices_shape).WithData<int16_t>({0, 14, 3, 1, -10, 1});
    Array e = testing::BuildArray(output_shape).WithData<T>({0, 2, 3, 1, 2, 1, 4, 6, 7, 5, 6, 5});

    Run([&]() {
        testing::CheckForward(
                [&indices, &axis](const std::vector<Array>& xs) { return std::vector<Array>{Take(xs[0], indices, axis)}; }, {a}, {e});
    });
}

TEST_P(IndexingTest, TakeInt16IndicesBackward) {
    using T = double;
    Shape input_shape{2, 4};
    Shape indices_shape{2, 3};
    Shape output_shape{2, 2, 3};
    int8_t axis = 1;
    Array a = (*testing::BuildArray(input_shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Array indices = testing::BuildArray(indices_shape).WithData<int16_t>({0, 14, 3, 1, -10, 1});
    Array go = testing::BuildArray(output_shape).WithLinearData<T>(0.1, 0.1).WithPadding(1);
    Array eps = Full(input_shape, 1e-3, Dtype::kFloat64);

    CheckBackward(
            [&indices, axis](const std::vector<Array>& xs) -> std::vector<Array> { return {Take(xs[0], indices, axis)}; },
            {a},
            {go},
            {eps});
}

TEST_THREAD_SAFE_P(IndexingTest, TakeInt32Indices) {
    using T = int8_t;
    Shape input_shape{2, 4};
    Shape indices_shape{2, 3};
    Shape output_shape{2, 2, 3};
    int8_t axis = 1;
    Array a = testing::BuildArray(input_shape).WithLinearData<T>().WithPadding(1);
    Array indices = testing::BuildArray(indices_shape).WithData<int32_t>({0, 14, 3, 1, -10, 1});
    Array e = testing::BuildArray(output_shape).WithData<T>({0, 2, 3, 1, 2, 1, 4, 6, 7, 5, 6, 5});

    Run([&]() {
        testing::CheckForward(
                [&indices, &axis](const std::vector<Array>& xs) { return std::vector<Array>{Take(xs[0], indices, axis)}; }, {a}, {e});
    });
}

TEST_P(IndexingTest, TakeInt32IndicesBackward) {
    using T = double;
    Shape input_shape{2, 4};
    Shape indices_shape{2, 3};
    Shape output_shape{2, 2, 3};
    int8_t axis = 1;
    Array a = (*testing::BuildArray(input_shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Array indices = testing::BuildArray(indices_shape).WithData<int32_t>({0, 14, 3, 1, -10, 1});
    Array go = testing::BuildArray(output_shape).WithLinearData<T>(0.1, 0.1).WithPadding(1);
    Array eps = Full(input_shape, 1e-3, Dtype::kFloat64);

    CheckBackward(
            [&indices, axis](const std::vector<Array>& xs) -> std::vector<Array> { return {Take(xs[0], indices, axis)}; },
            {a},
            {go},
            {eps});
}

TEST_THREAD_SAFE_P(IndexingTest, TakeInt64Indices) {
    using T = int8_t;
    Shape input_shape{2, 4};
    Shape indices_shape{2, 3};
    Shape output_shape{2, 2, 3};
    int8_t axis = 1;
    Array a = testing::BuildArray(input_shape).WithLinearData<T>().WithPadding(1);
    Array indices = testing::BuildArray(indices_shape).WithData<int64_t>({0, 14, 3, 1, -10, 1});
    Array e = testing::BuildArray(output_shape).WithData<T>({0, 2, 3, 1, 2, 1, 4, 6, 7, 5, 6, 5});

    Run([&]() {
        testing::CheckForward(
                [&indices, &axis](const std::vector<Array>& xs) { return std::vector<Array>{Take(xs[0], indices, axis)}; }, {a}, {e});
    });
}

TEST_P(IndexingTest, TakeInt64IndicesBackward) {
    using T = double;
    Shape input_shape{2, 4};
    Shape indices_shape{2, 3};
    Shape output_shape{2, 2, 3};
    int8_t axis = 1;
    Array a = (*testing::BuildArray(input_shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Array indices = testing::BuildArray(indices_shape).WithData<int64_t>({0, 14, 3, 1, -10, 1});
    Array go = testing::BuildArray(output_shape).WithLinearData<T>(0.1, 0.1).WithPadding(1);
    Array eps = Full(input_shape, 1e-3, Dtype::kFloat64);

    CheckBackward(
            [&indices, axis](const std::vector<Array>& xs) -> std::vector<Array> { return {Take(xs[0], indices, axis)}; },
            {a},
            {go},
            {eps});
}

TEST_P(IndexingTest, TakeDoubleBackward) {
    using T = double;
    Shape input_shape{2, 4};
    Shape indices_shape{2, 3};
    Shape output_shape{2, 2, 3};
    int8_t axis = 1;
    Array a = (*testing::BuildArray(input_shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Array indices = testing::BuildArray(indices_shape).WithData<int64_t>({0, 14, 3, 1, -10, 1});
    Array go = (*testing::BuildArray(output_shape).WithLinearData<T>(0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(input_shape).WithLinearData<T>(0.1, 0.1).WithPadding(1);
    Array epsi = Full(input_shape, 1e-3, Dtype::kFloat64);
    Array epso = Full(output_shape, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [&indices, axis](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Take(xs[0], indices, axis);
                return {y * y};  // to make it nonlinear
            },
            {a},
            {go},
            {ggi},
            {epsi, epso});
}

TEST_THREAD_SAFE_P(IndexingTest, TakeLongAxis) {
    Array a = testing::BuildArray({128}).WithLinearData<float>();
    Array indices = Full({1}, 10, Dtype::kInt64);
    Array e = Full({1}, 10.f);

    Run([&]() {
        testing::CheckForward([&indices](const std::vector<Array>& xs) { return std::vector<Array>{Take(xs[0], indices, 0)}; }, {a}, {e});
    });
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        IndexingTest,
        ::testing::Values(
#ifdef CHAINERX_ENABLE_CUDA
                std::string{"cuda"},
#endif  // CHAINERX_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace chainerx
