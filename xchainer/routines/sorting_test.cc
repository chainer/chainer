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
    // TODO(hvy): Run CUDA tests when CudaDevice::ArgMax is implemented.
    if (GetDefaultDevice().backend().GetName() == "cuda") {
        return;
    }
    Array a = testing::BuildArray({2, 3}).WithData<float>({1, 4, 3, 0, 1, 4}).WithPadding(1);
    Array b = ArgMax(a, 0);
    Array e = testing::BuildArray<int64_t>({3}, {0, 0, 1});
    testing::ExpectEqual(e, b);
}

TEST_P(SortingTest, ArgMaxNegativeAxis) {
    // TODO(hvy): Run CUDA tests when CudaDevice::ArgMax is implemented.
    if (GetDefaultDevice().backend().GetName() == "cuda") {
        return;
    }
    Array a = testing::BuildArray({2, 3}).WithData<float>({1, 4, 3, 0, 1, 4});
    Array b = ArgMax(a, -1);
    Array e = testing::BuildArray<int64_t>({2}, {1, 2});
    testing::ExpectEqual(e, b);
}

TEST_P(SortingTest, ArgMaxAllAxes) {
    // TODO(hvy): Run CUDA tests when CudaDevice::ArgMax is implemented.
    if (GetDefaultDevice().backend().GetName() == "cuda") {
        return;
    }
    Array a = testing::BuildArray({2, 3}).WithData<float>({1, 4, 3, 0, 1, 4});
    Array b = ArgMax(a);
    Array e = testing::BuildArray<int64_t>({}, {1});
    testing::ExpectEqual(e, b);
}

TEST_P(SortingTest, ArgMaxInvalidAxis) {
    Array a = testing::BuildArray({2, 3}).WithData<float>({1, 4, 3, 0, 1, 4});
    EXPECT_THROW(ArgMax(a, 3), DimensionError);
}

TEST_P(SortingTest, ArgMaxEmpty) {
    Array a = Array::Zeros({0}, Dtype::kFloat32);
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
