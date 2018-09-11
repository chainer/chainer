#include "chainerx/routines/sorting.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/device_id.h"
#include "chainerx/error.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/array_check.h"
#include "chainerx/testing/device_session.h"
#include "chainerx/testing/routines.h"
#include "chainerx/testing/threading.h"

namespace chainerx {
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

TEST_THREAD_SAFE_P(SortingTest, ArgMax) {
    Array a = testing::BuildArray({2, 3}).WithData<float>({1, 4, 3, 0, 1, 4}).WithPadding(1);
    Array e = testing::BuildArray({3}).WithData<int64_t>({0, 0, 1});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{ArgMax(xs[0], 0)}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(SortingTest, ArgMaxNegativeAxis) {
    Array a = testing::BuildArray({2, 3}).WithData<float>({1, 4, 3, 0, 1, 4});
    Array e = testing::BuildArray({2}).WithData<int64_t>({1, 2});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{ArgMax(xs[0], -1)}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(SortingTest, ArgMaxAllAxes) {
    Array a = testing::BuildArray({2, 3}).WithData<float>({1, 4, 3, 0, 1, 4});
    Array e = testing::BuildArray({}).WithData<int64_t>({1});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{ArgMax(xs[0])}; }, {a}, {e}); });
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
#ifdef CHAINERX_ENABLE_CUDA
                std::string{"cuda"},
#endif  // CHAINERX_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace chainerx
