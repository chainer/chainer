#include "xchainer/routines/linalg.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/check_backward.h"
#include "xchainer/device_id.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

class LinalgTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(LinalgTest, Dot) {
    Array a = testing::BuildArray({2, 3}).WithLinearData(1.f).WithPadding(1);
    Array b = testing::BuildArray({3, 2}).WithData<float>({1.f, 2.f, -1.f, -3.f, 2.f, 4.f}).WithPadding(2);
    Array c = Dot(a, b);
    Array e = testing::BuildArray({2, 2}).WithData<float>({5.f, 8.f, 11.f, 17.f});
    XCHAINER_EXPECT_EQ(e, c);
}

TEST_P(LinalgTest, DotZeroDim) {
    Array a = testing::BuildArray({2, 3}).WithLinearData<float>(1.f);
    Array b = testing::BuildArray({}).WithData<float>({2.f});
    Array c = Dot(a, b);
    Array e = testing::BuildArray({2, 3}).WithLinearData(2.f, 2.f);
    XCHAINER_EXPECT_EQ(e, c);
}

TEST_P(LinalgTest, DotVecVec) {
    Array a = testing::BuildArray({3}).WithLinearData(1.f);
    Array b = testing::BuildArray({3}).WithLinearData(1.f, 2.f);
    Array c = Dot(a, b);
    Array e = testing::BuildArray({}).WithData<float>({22.f});
    XCHAINER_EXPECT_EQ(e, c);
}

TEST_P(LinalgTest, DotMatVec) {
    Array a = testing::BuildArray({2, 3}).WithLinearData(1.f);
    Array b = testing::BuildArray({3}).WithLinearData(1.f, 2.f);
    Array c = Dot(a, b);
    Array e = testing::BuildArray({2}).WithData<float>({22.f, 49.f});
    XCHAINER_EXPECT_EQ(e, c);
}

TEST_P(LinalgTest, DotInvalidShape) {
    Array a = Zeros({2, 3}, Dtype::kFloat32);
    Array b = Zeros({2, 2}, a.dtype());
    EXPECT_THROW(Dot(a, b), DimensionError);
}

TEST_P(LinalgTest, DotAlongZeroLengthAxis) {
    Array a = Empty({2, 0}, Dtype::kFloat32);
    Array b = Empty({0, 2}, a.dtype());
    Array c = Dot(a, b);
    Array e = Zeros({2, 2}, a.dtype());
    XCHAINER_EXPECT_EQ(e, c);
}

TEST_P(LinalgTest, DotBackward) {
    Array a = (*testing::BuildArray({2, 3}).WithLinearData(1.f)).RequireGrad();
    Array b = (*testing::BuildArray({3, 2}).WithData<float>({1.f, 2.f, -1.f, -3.f, 2.f, 4.f})).RequireGrad();

    Array go = testing::BuildArray({2, 2}).WithLinearData(-0.1f, 0.1f).WithPadding(1);
    Array a_eps = Full(a.shape(), 1e-1f);
    Array b_eps = Full(b.shape(), 1e-1f);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Dot(xs[0], xs[1])}; }, {a, b}, {go}, {a_eps, b_eps});
}

TEST_P(LinalgTest, DotMatVecBackward) {
    Array a = (*testing::BuildArray({2, 3}).WithLinearData(1.f)).RequireGrad();
    Array b = (*testing::BuildArray({3}).WithData<float>({1.f, 2.f, -1.f})).RequireGrad();

    Array go = testing::BuildArray({2}).WithData<float>({-0.1f, 0.1f}).WithPadding(1);
    Array a_eps = Full(a.shape(), 1e-1f);
    Array b_eps = Full(b.shape(), 1e-1f);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Dot(xs[0], xs[1])}; }, {a, b}, {go}, {a_eps, b_eps});
}

TEST_P(LinalgTest, DotDoubleBackward) {
    Array a = (*testing::BuildArray({2, 3}).WithLinearData(1.f)).RequireGrad();
    Array b = (*testing::BuildArray({3, 2}).WithData<float>({1.f, 2.f, -1.f, -3.f, 2.f, 4.f})).RequireGrad();
    Array go = (*testing::BuildArray({2, 2}).WithLinearData(-0.1f, 0.1f).WithPadding(1)).RequireGrad();

    Array gga = testing::BuildArray(a.shape()).WithLinearData(-0.3f, 0.1f).WithPadding(1);
    Array ggb = testing::BuildArray(b.shape()).WithLinearData(-0.2f, 0.1f).WithPadding(1);
    Array a_eps = Full(a.shape(), 1e-1f);
    Array b_eps = Full(b.shape(), 1e-1f);
    Array go_eps = Full(go.shape(), 1e-1f);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Dot(xs[0], xs[1])}; },
            {a, b},
            {go},
            {gga, ggb},
            {a_eps, b_eps, go_eps});
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        LinalgTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace xchainer
