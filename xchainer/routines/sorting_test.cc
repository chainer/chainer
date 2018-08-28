#include "xchainer/routines/sorting.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/device_id.h"
#include "xchainer/error.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/device_session.h"
#include "xchainer/testing/routines.h"

namespace xchainer {
namespace {

class SortingTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(SortingTest, ArgMax) {
    Array a = testing::BuildArray({2, 3}).WithData<float>({1, 4, 3, 0, 1, 4}).WithPadding(1);
    Array e = testing::BuildArray({3}).WithData<int64_t>({0, 0, 1});

    testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{ArgMax(xs[0], 0)}; }, {a}, {e}, 1);
}

TEST_P(SortingTest, ArgMaxNegativeAxis) {
    Array a = testing::BuildArray({2, 3}).WithData<float>({1, 4, 3, 0, 1, 4});
    Array e = testing::BuildArray({2}).WithData<int64_t>({1, 2});

    testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{ArgMax(xs[0], -1)}; }, {a}, {e}, 1);
}

TEST_P(SortingTest, ArgMaxAllAxes) {
    Array a = testing::BuildArray({2, 3}).WithData<float>({1, 4, 3, 0, 1, 4});
    Array e = testing::BuildArray({}).WithData<int64_t>({1});

    testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{ArgMax(xs[0])}; }, {a}, {e}, 1);
}

TEST_P(SortingTest, ArgMaxInvalidAxis) {
    Array a = testing::BuildArray({2, 3}).WithData<float>({1, 4, 3, 0, 1, 4});
    EXPECT_THROW(ArgMax(a, 3), DimensionError);
}

TEST_P(SortingTest, ArgMaxEmpty) {
    Array a = Zeros({0}, Dtype::kFloat32);
    EXPECT_THROW(ArgMax(a), DimensionError);
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        SortingTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace xchainer
