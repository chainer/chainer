#include "xchainer/routines/normalization.h"

#include <string>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/shape.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
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
    using T = float;

    Shape x_shape{3, 1, 4, 4};
    Shape reduced{1, 4, 4};
    float eps = 2e-5f;
    float decay = 0.9f;

    Array a = testing::BuildArray(x_shape).WithLinearData<T>();
    Array gamma = testing::BuildArray(reduced).WithLinearData<T>();
    Array beta = testing::BuildArray(reduced).WithLinearData<T>();
    Array running_mean = testing::BuildArray(reduced).WithLinearData<T>();
    Array running_var = testing::BuildArray(reduced).WithLinearData<T>();
    Array out = BatchNormalization(a, gamma, beta, running_mean, running_var, eps, decay);

    Array e_out = testing::BuildArray(x_shape).WithData<float>(
            {0.,        -0.2247448, -0.4494896, -0.6742344, -0.8989792, -1.123724, -1.3484688, -1.5732136, -1.7979584, -2.0227032,
             -2.247448, -2.4721928, -2.6969376, -2.9216824, -3.1464272, -3.371172, 0.,         1.,         2.,         3.,
             4.,        5.,         6.,         7.,         8.,         9.,        10.,        11.,        12.,        13.,
             14.,       15.,        0.,         2.2247448,  4.4494896,  6.6742344, 8.898979,   11.123724,  13.348469,  15.573214,
             17.797958, 20.022703,  22.247448,  24.472193,  26.696938,  28.921682, 31.146427,  33.37117});  // Computed with Chainer.

    Array e_running_mean = testing::BuildArray(reduced).WithData<float>({1.6,
                                                                         2.6,
                                                                         3.6,
                                                                         4.6,
                                                                         5.6,
                                                                         6.6000004,
                                                                         7.5999994,
                                                                         8.599999,
                                                                         9.6,
                                                                         10.599999,
                                                                         11.6,
                                                                         12.599999,
                                                                         13.599999,
                                                                         14.6,
                                                                         15.599999,
                                                                         16.6});  // Computed with Chainer.

    Array e_running_var = testing::BuildArray(reduced).WithData<float>({25.600002,
                                                                        26.500002,
                                                                        27.400002,
                                                                        28.300003,
                                                                        29.200003,
                                                                        30.100002,
                                                                        31.000002,
                                                                        31.900002,
                                                                        32.800003,
                                                                        33.7,
                                                                        34.600002,
                                                                        35.5,
                                                                        36.4,
                                                                        37.300003,
                                                                        38.2,
                                                                        39.100002});  // Computed with Chainer.
    testing::ExpectEqual(e_out, out);
    testing::ExpectAllClose(e_running_mean, running_mean, 1e-5f, 1e-5f);
    testing::ExpectAllClose(e_running_var, running_var, 1e-5f, 1e-5f);
}

TEST_P(NormalizationTest, BatchNormalizationWithAxis) {
    if (GetParam() == "cuda") {
        // TODO(hvy): Add CUDA implementation
        return;
    }
    using T = float;

    Shape x_shape{3, 1, 4, 4};
    Shape reduced{4};
    Axes axis{0, 1, 2};
    float eps = 2e-5f;
    float decay = 0.9f;

    Array a = testing::BuildArray(x_shape).WithLinearData<T>();
    Array gamma = testing::BuildArray(reduced).WithLinearData<T>();
    Array beta = testing::BuildArray(reduced).WithLinearData<T>();
    Array running_mean = testing::BuildArray(reduced).WithLinearData<T>();
    Array running_var = testing::BuildArray(reduced).WithLinearData<T>();
    Array out = BatchNormalization(a, gamma, beta, running_mean, running_var, eps, decay, axis);

    Array e_out = testing::BuildArray(x_shape).WithData<float>(
            {0.,         -0.5932549,  -1.1865098, -1.7797647, 0.,        -0.30357218, -0.60714436, -0.91071653, 0.,        -0.01388955,
             -0.0277791, -0.04166865, 0.,         0.2757932,  0.5515864, 0.8273797,   0.,          0.56547594,  1.1309519, 1.6964278,
             0.,         0.8551586,   1.7103173,  2.565476,   0.,        1.1448413,   2.2896826,   3.434524,    0.,        1.434524,
             2.869048,   4.303572,    0.,         1.7242068,  3.4484136, 5.1726203,   0.,          2.0138896,   4.027779,  6.041669,
             0.,         2.3035722,   4.6071444,  6.9107165,  0.,        2.593255,    5.18651,     7.7797647});  // Computed with Chainer.

    Array e_running_mean = testing::BuildArray(reduced).WithData<float>({2.2, 3.1999998, 4.2, 5.2});  // Computed with Chainer.

    Array e_running_var = testing::BuildArray(reduced).WithData<float>({20.800001, 21.7, 22.6, 23.5});  // Computed with Chainer.
    testing::ExpectAllClose(e_out, out, 1e-5f, 1e-5f);
    testing::ExpectAllClose(e_running_mean, running_mean, 1e-5f, 1e-5f);
    testing::ExpectAllClose(e_running_var, running_var, 1e-5f, 1e-5f);
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
