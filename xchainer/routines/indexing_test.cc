#include "xchainer/routines/indexing.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/check_backward.h"
#include "xchainer/error.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
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

TEST_P(IndexingTest, At) {
    using T = int32_t;
    Shape input_shape{2, 3, 1};
    Shape output_shape{1, 2, 1};
    std::vector<ArrayIndex> indices{-1, NewAxis{}, Slice{1, 3}};
    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    Array b = internal::At(a, indices);

    EXPECT_EQ(output_shape, b.shape());
    Array e = testing::BuildArray(output_shape).WithData<T>({4, 5});
    testing::ExpectEqual(e, b);

    // Check if strides are 0 for newaxis.
    EXPECT_EQ(0, b.strides()[0]);
    EXPECT_NE(0, b.strides()[1]);
    EXPECT_NE(0, b.strides()[2]);
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
    CheckBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                std::vector<ArrayIndex> indices{1, NewAxis{}, Slice{1, 3}};
                return {internal::At(xs[0], indices)};
            },
            {(*testing::BuildArray({2, 3}, {1.f, -1.f, 2.f, -2.f, 3.f, -3.f})).RequireGrad()},
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
            {(*testing::BuildArray({2, 3}, {1.f, -1.f, 2.f, -2.f, 3.f, -3.f})).RequireGrad()},
            {Ones({1, 2}, Dtype::kFloat32).RequireGrad()},
            {Ones({2, 3}, Dtype::kFloat32)},
            {Full({2, 3}, 1e-3f), Full({1, 2}, 1e-3f)});
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        IndexingTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace xchainer
