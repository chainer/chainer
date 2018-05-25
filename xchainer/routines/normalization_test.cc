#include "xchainer/routines/normalization.h"

#include <string>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

class NormalizationTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(NormalizationTest, BatchNormalization) {
    if (GetParam() == "cuda") {
        // TODO(hvy): Add CUDA implementation
        return;
    }
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        NormalizationTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace xchainer
