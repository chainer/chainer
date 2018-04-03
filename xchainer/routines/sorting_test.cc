#include "xchainer/routines/sorting.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/device_id.h"
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
    {
        Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::BuildArray<float>({1}, {2});
        ArgMax(a, std::vector<int8_t>{0});
        testing::ExpectEqual<float>(e, a);
    }
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
