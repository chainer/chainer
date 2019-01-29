#include "chainerx/routines/connection.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/check_backward.h"
#include "chainerx/constant.h"
#include "chainerx/device_id.h"
#include "chainerx/error.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/array_check.h"
#include "chainerx/testing/device_session.h"
#include "chainerx/testing/routines.h"
#include "chainerx/testing/threading.h"

namespace chainerx {
namespace {

class ConnectionTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_THREAD_SAFE_P(ConnectionTest, Conv2d) {
    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{10, 7};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};
    StackVector<int64_t, kMaxNdim> stride{3, 2};
    StackVector<int64_t, kMaxNdim> pad{2, 0};
    bool cover_all = false;
    Shape out_dims{5, 3};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{out_channels, in_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};
    Shape out_shape{batch_size, out_channels};
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));

    Array x = testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2.0f, 1.0f).WithPadding(1);
    Array w = testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2.0f, 1.0f);
    Array b = testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f});

    Array e = testing::BuildArray(out_shape).WithData<float>(
            {-2.00000e-01, -2.00000e-01, -2.00000e-01, 2.71198e+04,  2.67778e+04,  2.64358e+04,  2.35288e+04,  2.31868e+04,  2.28448e+04,
             1.99378e+04,  1.95958e+04,  1.92538e+04,  -2.00000e-01, -2.00000e-01, -2.00000e-01, 1.30000e+00,  1.30000e+00,  1.30000e+00,
             -1.45127e+04, -1.42067e+04, -1.39007e+04, -1.12997e+04, -1.09937e+04, -1.06877e+04, -8.08670e+03, -7.78070e+03, -7.47470e+03,
             1.30000e+00,  1.30000e+00,  1.30000e+00,  -2.00000e-01, -2.00000e-01, -2.00000e-01, -8.79020e+03, -9.13220e+03, -9.47420e+03,
             -1.23812e+04, -1.27232e+04, -1.30652e+04, -1.59722e+04, -1.63142e+04, -1.66562e+04, -2.00000e-01, -2.00000e-01, -2.00000e-01,
             1.30000e+00,  1.30000e+00,  1.30000e+00,  1.76173e+04,  1.79233e+04,  1.82293e+04,  2.08303e+04,  2.11363e+04,  2.14423e+04,
             2.40433e+04,  2.43493e+04,  2.46553e+04,  1.30000e+00,  1.30000e+00,  1.30000e+00});  // Computed with Chainer.

    Run([&]() {
        testing::CheckForward(
                [&stride, &pad, &cover_all](const std::vector<Array>& xs) {
                    return std::vector<Array>{Conv(xs[0], xs[1], xs[2], stride, pad, cover_all)};
                },
                {x, w, b},
                {e});
    });
}

TEST_THREAD_SAFE_P(ConnectionTest, ConvNd) {
    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{2, 3, 4};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3, 4};
    StackVector<int64_t, kMaxNdim> stride{3, 2, 1};
    StackVector<int64_t, kMaxNdim> pad{2, 0, 1};
    Shape out_dims{2, 1, 3};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{out_channels, in_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};
    Shape out_shape{batch_size, out_channels};
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));

    Array x = testing::BuildArray(x_shape).WithLinearData<double>(-x_shape.GetTotalSize() / 2.0f, 1.0).WithPadding(1);
    Array w = testing::BuildArray(w_shape).WithLinearData<double>(-w_shape.GetTotalSize() / 2.0f, 1.0);
    Array b = testing::BuildArray(b_shape).WithData<double>({-0.2, 1.3});

    Array e = testing::BuildArray(out_shape).WithData<double>(
            {-2.00000e-01, -2.00000e-01, -2.00000e-01, 4.58278e+04,  6.09178e+04,  4.55038e+04,  1.30000e+00,  1.30000e+00,  1.30000e+00,
             -1.44347e+04, -1.81367e+04, -1.28147e+04, -2.00000e-01, -2.00000e-01, -2.00000e-01, -3.58202e+04, -4.92422e+04, -3.80882e+04,
             1.30000e+00,  1.30000e+00,  1.30000e+00,  4.38853e+04,  5.83273e+04,  4.35613e+04});  // Computed with Chainer.

    Run([&]() {
        testing::CheckForward(
                [&stride, &pad](const std::vector<Array>& xs) { return std::vector<Array>{Conv(xs[0], xs[1], xs[2], stride, pad, false)}; },
                {x, w, b},
                {e});
    });
}

TEST_THREAD_SAFE_P(ConnectionTest, ConvCoverAll) {
    if (GetParam() == "cuda") {
        // CuDNN convolution does not support cover_all
        Skip();
        return;
    }
    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{10, 8};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};
    StackVector<int64_t, kMaxNdim> stride{3, 2};
    StackVector<int64_t, kMaxNdim> pad{2, 0};
    bool cover_all = true;
    Shape out_dims{5, 4};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{out_channels, in_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};
    Shape out_shape{batch_size, out_channels};
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));

    Array x = testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2.0f, 1.0f).WithPadding(1);
    Array w = testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2.0f, 1.0f);
    Array b = testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f});

    Array e =
            testing::BuildArray(out_shape).WithData<float>(
                    {-2.00000e-01, -2.00000e-01, -2.00000e-01, -2.00000e-01, 3.10168e+04,  3.06748e+04,  3.03328e+04,
                     2.08948e+04,  2.69128e+04,  2.65708e+04,  2.62288e+04,  1.80148e+04,  2.28088e+04,  2.24668e+04,
                     2.21248e+04,  1.51348e+04,  -2.00000e-01, -2.00000e-01, -2.00000e-01, -2.00000e-01, 1.30000e+00,
                     1.30000e+00,  1.30000e+00,  1.30000e+00,  -1.66097e+04, -1.63037e+04, -1.59977e+04, -9.66770e+03,
                     -1.29377e+04, -1.26317e+04, -1.23257e+04, -7.36370e+03, -9.26570e+03, -8.95970e+03, -8.65370e+03,
                     -5.05970e+03, 1.30000e+00,  1.30000e+00,  1.30000e+00,  1.30000e+00,  -2.00000e-01, -2.00000e-01,
                     -2.00000e-01, -2.00000e-01, -1.00232e+04, -1.03652e+04, -1.07072e+04, -7.90520e+03, -1.41272e+04,
                     -1.44692e+04, -1.48112e+04, -1.07852e+04, -1.82312e+04, -1.85732e+04, -1.89152e+04, -1.36652e+04,
                     -2.00000e-01, -2.00000e-01, -2.00000e-01, -2.00000e-01, 1.30000e+00,  1.30000e+00,  1.30000e+00,
                     1.30000e+00,  2.01103e+04,  2.04163e+04,  2.07223e+04,  1.33723e+04,  2.37823e+04,  2.40883e+04,
                     2.43943e+04,  1.56763e+04,  2.74543e+04,  2.77603e+04,  2.80663e+04,  1.79803e+04,  1.30000e+00,
                     1.30000e+00,  1.30000e+00,  1.30000e+00});  // Computed with Chainer.

    Run([&]() {
        testing::CheckForward(
                [&stride, &pad, &cover_all](const std::vector<Array>& xs) {
                    return std::vector<Array>{Conv(xs[0], xs[1], xs[2], stride, pad, cover_all)};
                },
                {x, w, b},
                {e});
    });
}

TEST_P(ConnectionTest, ConvInvalidStride) {
    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{10, 7};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};
    StackVector<int64_t, kMaxNdim> stride{3, 0};  // Invalid stride element 0.
    StackVector<int64_t, kMaxNdim> pad{2, 0};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{out_channels, in_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};

    Array x = testing::BuildArray(x_shape).WithLinearData<float>();
    Array w = testing::BuildArray(w_shape).WithLinearData<float>();
    Array b = testing::BuildArray(b_shape).WithLinearData<float>();

    EXPECT_THROW(Conv(x, w, b, stride, pad), DimensionError);
}

TEST_P(ConnectionTest, ConvBackward) {
    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{10, 7};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};
    StackVector<int64_t, kMaxNdim> stride{3, 2};
    StackVector<int64_t, kMaxNdim> pad{2, 0};
    bool cover_all = false;
    Shape out_dims{5, 3};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{out_channels, in_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};
    Shape out_shape{batch_size, out_channels};
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));

    Array x = (*testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2.0f, 1.0f).WithPadding(1)).RequireGrad();
    Array w = (*testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2.0f, 1.0f)).RequireGrad();
    Array b = (*testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f})).RequireGrad();

    Array go = testing::BuildArray(out_shape).WithLinearData(-0.1f, 0.1f).WithPadding(1);

    Array x_eps = Full(x.shape(), 1e0f);
    Array w_eps = Full(w.shape(), 1e0f);
    Array b_eps = Full(b.shape(), 1e0f);

    CheckBackward(
            [&](const std::vector<Array>& xs) -> std::vector<Array> {
                const Array& xx = xs[0];
                const Array& ww = xs[1];
                const Array& bb = xs[2];
                return {Conv(xx, ww, bb, stride, pad, cover_all)};
            },
            {x, w, b},
            {go},
            {x_eps, w_eps, b_eps},
            2U,
            1e-6,
            1e-3);
}

TEST_P(ConnectionTest, ConvCoverAllBackward) {
    if (GetParam() == "cuda") {
        // CuDNN convolution does not support cover_all
        return;
    }
    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{10, 8};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};
    StackVector<int64_t, kMaxNdim> stride{3, 2};
    StackVector<int64_t, kMaxNdim> pad{2, 0};
    bool cover_all = true;
    Shape out_dims{5, 4};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{out_channels, in_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};
    Shape out_shape{batch_size, out_channels};
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));

    Array x = (*testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2.0f, 1.0f).WithPadding(1)).RequireGrad();
    Array w = (*testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2.0f, 1.0f)).RequireGrad();
    Array b = (*testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f})).RequireGrad();

    Array go = testing::BuildArray(out_shape).WithLinearData(-0.1f, 0.1f).WithPadding(1);

    Array x_eps = Full(x.shape(), 1e0f);
    Array w_eps = Full(w.shape(), 1e0f);
    Array b_eps = Full(b.shape(), 1e0f);

    CheckBackward(
            [&](const std::vector<Array>& xs) -> std::vector<Array> {
                const Array& xx = xs[0];
                const Array& ww = xs[1];
                const Array& bb = xs[2];
                return {Conv(xx, ww, bb, stride, pad, cover_all)};
            },
            {x, w, b},
            {go},
            {x_eps, w_eps, b_eps},
            2U,
            1e-6,
            1e-3);
}

TEST_P(ConnectionTest, ConvDoubleBackward) {
    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{10, 7};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};
    StackVector<int64_t, kMaxNdim> stride{3, 2};
    StackVector<int64_t, kMaxNdim> pad{2, 0};
    bool cover_all = false;
    Shape out_dims{5, 3};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{out_channels, in_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};
    Shape out_shape{batch_size, out_channels};
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));

    Array x = (*testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2.0f, 1.0f).WithPadding(1)).RequireGrad();
    Array w = (*testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2.0f, 1.0f)).RequireGrad();
    Array b = (*testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f})).RequireGrad();

    Array go = (*testing::BuildArray(out_shape).WithLinearData(-0.3f, 0.1f).WithPadding(1)).RequireGrad();
    Array ggx = testing::BuildArray(x_shape).WithLinearData(0.4f, -0.2f).WithPadding(1);
    Array ggw = testing::BuildArray(w_shape).WithLinearData(-0.5f, 0.3f).WithPadding(1);
    Array ggb = testing::BuildArray(b_shape).WithLinearData(-0.6f, -0.4f).WithPadding(1);

    Array x_eps = Full(x.shape(), 1e3f);
    Array w_eps = Full(w.shape(), 1e3f);
    Array b_eps = Full(b.shape(), 1e3f);
    Array go_eps = Full(out_shape, 1e3f);

    CheckDoubleBackwardComputation(
            [&](const std::vector<Array>& xs) -> std::vector<Array> {
                const Array& xx = xs[0];
                const Array& ww = xs[1];
                const Array& bb = xs[2];
                Array y = Conv(xx, ww, bb, stride, pad, cover_all);
                return {y * y};
            },
            {x, w, b},
            {go},
            {ggx, ggw, ggb},
            {x_eps, w_eps, b_eps, go_eps},
            2,
            1e-2,
            1e-3);
}

TEST_P(ConnectionTest, ConvCoverAllDoubleBackward) {
    if (GetParam() == "cuda") {
        // CuDNN convolution does not support cover_all
        return;
    }
    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{10, 8};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};
    StackVector<int64_t, kMaxNdim> stride{3, 2};
    StackVector<int64_t, kMaxNdim> pad{2, 0};
    bool cover_all = true;
    Shape out_dims{5, 4};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{out_channels, in_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};
    Shape out_shape{batch_size, out_channels};
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));

    Array x = (*testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2.0f, 1.0f).WithPadding(1)).RequireGrad();
    Array w = (*testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2.0f, 1.0f)).RequireGrad();
    Array b = (*testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f})).RequireGrad();

    Array go = (*testing::BuildArray(out_shape).WithLinearData(-0.3f, 0.1f).WithPadding(1)).RequireGrad();
    Array ggx = testing::BuildArray(x_shape).WithLinearData(0.4f, -0.2f).WithPadding(1);
    Array ggw = testing::BuildArray(w_shape).WithLinearData(-0.5f, 0.3f).WithPadding(1);
    Array ggb = testing::BuildArray(b_shape).WithLinearData(-0.6f, -0.4f).WithPadding(1);

    Array x_eps = Full(x.shape(), 1e3f);
    Array w_eps = Full(w.shape(), 1e3f);
    Array b_eps = Full(b.shape(), 1e3f);
    Array go_eps = Full(out_shape, 1e3f);

    CheckDoubleBackwardComputation(
            [&](const std::vector<Array>& xs) -> std::vector<Array> {
                const Array& xx = xs[0];
                const Array& ww = xs[1];
                const Array& bb = xs[2];
                Array y = Conv(xx, ww, bb, stride, pad, cover_all);
                return {y * y};
            },
            {x, w, b},
            {go},
            {ggx, ggw, ggb},
            {x_eps, w_eps, b_eps, go_eps},
            2,
            1e-2,
            1e-3);
}

TEST_THREAD_SAFE_P(ConnectionTest, ConvTranspose) {
    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{5, 3};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};
    StackVector<int64_t, kMaxNdim> stride{3, 2};
    StackVector<int64_t, kMaxNdim> pad{2, 0};
    Shape out_dims{10, 7};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{in_channels, out_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};
    Shape out_shape{batch_size, out_channels};
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));

    Array x = testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2.0f, 1.0f).WithPadding(1);
    Array w = testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2.0f, 1.0f);
    Array b = testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f});

    Array e = testing::BuildArray(out_shape).WithData<float>(
            {-2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, 8.4580e+02,  7.6480e+02,
             1.5118e+03,  7.4980e+02,  1.4818e+03,  7.3480e+02,  6.5980e+02,  6.0280e+02,  5.2180e+02,  1.0348e+03,  5.1580e+02,
             1.0228e+03,  5.0980e+02,  4.3480e+02,  -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01,
             -2.0000e-01, 7.9180e+02,  7.1980e+02,  1.4218e+03,  7.0480e+02,  1.3918e+03,  6.8980e+02,  6.2380e+02,  5.7580e+02,
             5.0380e+02,  9.9880e+02,  4.9780e+02,  9.8680e+02,  4.9180e+02,  4.2580e+02,  -2.0000e-01, -2.0000e-01, -2.0000e-01,
             -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, 7.3780e+02,  6.7480e+02,  1.3318e+03,  6.5980e+02,  1.3018e+03,
             6.4480e+02,  5.8780e+02,  5.4880e+02,  4.8580e+02,  9.6280e+02,  4.7980e+02,  9.5080e+02,  4.7380e+02,  4.1680e+02,
             -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, 1.3000e+00,  1.3000e+00,
             1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  3.6130e+02,  2.8030e+02,  5.5930e+02,  2.8330e+02,
             5.6530e+02,  2.8630e+02,  2.1130e+02,  1.1830e+02,  3.7300e+01,  8.2300e+01,  4.9300e+01,  1.0630e+02,  6.1300e+01,
             -1.3700e+01, 1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  3.6130e+02,
             2.8930e+02,  5.7730e+02,  2.9230e+02,  5.8330e+02,  2.9530e+02,  2.2930e+02,  1.4530e+02,  7.3300e+01,  1.5430e+02,
             8.5300e+01,  1.7830e+02,  9.7300e+01,  3.1300e+01,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,
             1.3000e+00,  1.3000e+00,  3.6130e+02,  2.9830e+02,  5.9530e+02,  3.0130e+02,  6.0130e+02,  3.0430e+02,  2.4730e+02,
             1.7230e+02,  1.0930e+02,  2.2630e+02,  1.2130e+02,  2.5030e+02,  1.3330e+02,  7.6300e+01,  1.3000e+00,  1.3000e+00,
             1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01,
             -2.0000e-01, -2.0000e-01, -2.0000e-01, 3.5800e+01,  8.9800e+01,  1.6180e+02,  7.4800e+01,  1.3180e+02,  5.9800e+01,
             1.1980e+02,  1.9780e+02,  2.5180e+02,  4.9480e+02,  2.4580e+02,  4.8280e+02,  2.3980e+02,  2.9980e+02,  -2.0000e-01,
             -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -1.8200e+01, 4.4800e+01,  7.1800e+01,
             2.9800e+01,  4.1800e+01,  1.4800e+01,  8.3800e+01,  1.7080e+02,  2.3380e+02,  4.5880e+02,  2.2780e+02,  4.4680e+02,
             2.2180e+02,  2.9080e+02,  -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01,
             -7.2200e+01, -2.0000e-01, -1.8200e+01, -1.5200e+01, -4.8200e+01, -3.0200e+01, 4.7800e+01,  1.4380e+02,  2.1580e+02,
             4.2280e+02,  2.0980e+02,  4.1080e+02,  2.0380e+02,  2.8180e+02,  -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01,
             -2.0000e-01, -2.0000e-01, -2.0000e-01, 1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,
             1.3000e+00,  3.6130e+02,  4.1530e+02,  8.2930e+02,  4.1830e+02,  8.3530e+02,  4.2130e+02,  4.8130e+02,  5.2330e+02,
             5.7730e+02,  1.1623e+03,  5.8930e+02,  1.1863e+03,  6.0130e+02,  6.6130e+02,  1.3000e+00,  1.3000e+00,  1.3000e+00,
             1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  3.6130e+02,  4.2430e+02,  8.4730e+02,  4.2730e+02,  8.5330e+02,
             4.3030e+02,  4.9930e+02,  5.5030e+02,  6.1330e+02,  1.2343e+03,  6.2530e+02,  1.2583e+03,  6.3730e+02,  7.0630e+02,
             1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  3.6130e+02,  4.3330e+02,
             8.6530e+02,  4.3630e+02,  8.7130e+02,  4.3930e+02,  5.1730e+02,  5.7730e+02,  6.4930e+02,  1.3063e+03,  6.6130e+02,
             1.3303e+03,  6.7330e+02,  7.5130e+02,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,
             1.3000e+00});  // Computed with Chainer.

    Run([&]() {
        testing::CheckForward(
                [&stride, &pad](const std::vector<Array>& xs) {
                    return std::vector<Array>{ConvTranspose(xs[0], xs[1], xs[2], stride, pad)};
                },
                {x, w, b},
                {e});
    });
}

TEST_THREAD_SAFE_P(ConnectionTest, ConvTransposeOutSize) {
    if (GetParam() == "cuda") {
        // CUDA Convolution does not support out_size
        Skip();
        return;
    }
    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{5, 4};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};
    StackVector<int64_t, kMaxNdim> stride{3, 2};
    StackVector<int64_t, kMaxNdim> pad{2, 0};
    Shape out_dims{10, 8};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{in_channels, out_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};
    Shape out_shape{batch_size, out_channels};
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));

    Array x = testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2.0f, 1.0f).WithPadding(1);
    Array w = testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2.0f, 1.0f);
    Array b = testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f});

    Array e = testing::BuildArray(out_shape).WithData<float>(
            {-2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, 1.1278e+03,
             1.0198e+03,  2.0218e+03,  1.0048e+03,  1.9918e+03,  9.8980e+02,  1.9618e+03,  9.7480e+02,  8.0380e+02,  6.9580e+02,
             1.3828e+03,  6.8980e+02,  1.3708e+03,  6.8380e+02,  1.3588e+03,  6.7780e+02,  -2.0000e-01, -2.0000e-01, -2.0000e-01,
             -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, 1.0558e+03,  9.5980e+02,  1.9018e+03,  9.4480e+02,
             1.8718e+03,  9.2980e+02,  1.8418e+03,  9.1480e+02,  7.6780e+02,  6.7180e+02,  1.3348e+03,  6.6580e+02,  1.3228e+03,
             6.5980e+02,  1.3108e+03,  6.5380e+02,  -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01,
             -2.0000e-01, -2.0000e-01, 9.8380e+02,  8.9980e+02,  1.7818e+03,  8.8480e+02,  1.7518e+03,  8.6980e+02,  1.7218e+03,
             8.5480e+02,  7.3180e+02,  6.4780e+02,  1.2868e+03,  6.4180e+02,  1.2748e+03,  6.3580e+02,  1.2628e+03,  6.2980e+02,
             -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, 1.3000e+00,
             1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  4.8130e+02,  3.7330e+02,
             7.4530e+02,  3.7630e+02,  7.5130e+02,  3.7930e+02,  7.5730e+02,  3.8230e+02,  1.5730e+02,  4.9300e+01,  1.0630e+02,
             6.1300e+01,  1.3030e+02,  7.3300e+01,  1.5430e+02,  8.5300e+01,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,
             1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  4.8130e+02,  3.8530e+02,  7.6930e+02,  3.8830e+02,  7.7530e+02,
             3.9130e+02,  7.8130e+02,  3.9430e+02,  1.9330e+02,  9.7300e+01,  2.0230e+02,  1.0930e+02,  2.2630e+02,  1.2130e+02,
             2.5030e+02,  1.3330e+02,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,
             1.3000e+00,  4.8130e+02,  3.9730e+02,  7.9330e+02,  4.0030e+02,  7.9930e+02,  4.0330e+02,  8.0530e+02,  4.0630e+02,
             2.2930e+02,  1.4530e+02,  2.9830e+02,  1.5730e+02,  3.2230e+02,  1.6930e+02,  3.4630e+02,  1.8130e+02,  1.3000e+00,
             1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  -2.0000e-01, -2.0000e-01,
             -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, 4.7800e+01,  1.1980e+02,  2.2180e+02,
             1.0480e+02,  1.9180e+02,  8.9800e+01,  1.6180e+02,  7.4800e+01,  2.6380e+02,  3.3580e+02,  6.6280e+02,  3.2980e+02,
             6.5080e+02,  3.2380e+02,  6.3880e+02,  3.1780e+02,  -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01,
             -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.4200e+01, 5.9800e+01,  1.0180e+02,  4.4800e+01,  7.1800e+01,  2.9800e+01,
             4.1800e+01,  1.4800e+01,  2.2780e+02,  3.1180e+02,  6.1480e+02,  3.0580e+02,  6.0280e+02,  2.9980e+02,  5.9080e+02,
             2.9380e+02,  -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01,
             -9.6200e+01, -2.0000e-01, -1.8200e+01, -1.5200e+01, -4.8200e+01, -3.0200e+01, -7.8200e+01, -4.5200e+01, 1.9180e+02,
             2.8780e+02,  5.6680e+02,  2.8180e+02,  5.5480e+02,  2.7580e+02,  5.4280e+02,  2.6980e+02,  -2.0000e-01, -2.0000e-01,
             -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, -2.0000e-01, 1.3000e+00,  1.3000e+00,  1.3000e+00,
             1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  4.8130e+02,  5.5330e+02,  1.1053e+03,  5.5630e+02,
             1.1113e+03,  5.5930e+02,  1.1173e+03,  5.6230e+02,  6.9730e+02,  7.6930e+02,  1.5463e+03,  7.8130e+02,  1.5703e+03,
             7.9330e+02,  1.5943e+03,  8.0530e+02,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,
             1.3000e+00,  1.3000e+00,  4.8130e+02,  5.6530e+02,  1.1293e+03,  5.6830e+02,  1.1353e+03,  5.7130e+02,  1.1413e+03,
             5.7430e+02,  7.3330e+02,  8.1730e+02,  1.6423e+03,  8.2930e+02,  1.6663e+03,  8.4130e+02,  1.6903e+03,  8.5330e+02,
             1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  4.8130e+02,
             5.7730e+02,  1.1533e+03,  5.8030e+02,  1.1593e+03,  5.8330e+02,  1.1653e+03,  5.8630e+02,  7.6930e+02,  8.6530e+02,
             1.7383e+03,  8.7730e+02,  1.7623e+03,  8.8930e+02,  1.7863e+03,  9.0130e+02,  1.3000e+00,  1.3000e+00,  1.3000e+00,
             1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00,  1.3000e+00});  // Computed with Chainer.

    Run([&]() {
        testing::CheckForward(
                [&stride, &pad, &out_dims](const std::vector<Array>& xs) {
                    return std::vector<Array>{ConvTranspose(xs[0], xs[1], xs[2], stride, pad, out_dims)};
                },
                {x, w, b},
                {e});
    });
}

TEST_P(ConnectionTest, ConvTransposeInvalidStride) {
    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{5, 3};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};
    StackVector<int64_t, kMaxNdim> stride{0, 2};  // Invalid stride element 0.
    StackVector<int64_t, kMaxNdim> pad{2, 0};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{in_channels, out_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};

    Array x = testing::BuildArray(x_shape).WithLinearData<float>();
    Array w = testing::BuildArray(w_shape).WithLinearData<float>();
    Array b = testing::BuildArray(b_shape).WithLinearData<float>();

    EXPECT_THROW(ConvTranspose(x, w, b, stride, pad), DimensionError);
}

TEST_P(ConnectionTest, ConvTransposeBackward) {
    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{5, 3};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};
    StackVector<int64_t, kMaxNdim> stride{3, 2};
    StackVector<int64_t, kMaxNdim> pad{2, 0};
    Shape out_dims{10, 7};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{in_channels, out_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};
    Shape out_shape{batch_size, out_channels};
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));

    Array x = (*testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2.0f, 1.0f).WithPadding(1)).RequireGrad();
    Array w = (*testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2.0f, 1.0f)).RequireGrad();
    Array b = (*testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f})).RequireGrad();

    Array go = testing::BuildArray(out_shape).WithLinearData(-0.1f, 0.1f).WithPadding(1);

    Array x_eps = Full(x.shape(), 1e0f);
    Array w_eps = Full(w.shape(), 1e0f);
    Array b_eps = Full(b.shape(), 1e0f);

    CheckBackward(
            [&](const std::vector<Array>& xs) -> std::vector<Array> {
                const Array& xx = xs[0];
                const Array& ww = xs[1];
                const Array& bb = xs[2];
                return {ConvTranspose(xx, ww, bb, stride, pad, out_dims)};
            },
            {x, w, b},
            {go},
            {x_eps, w_eps, b_eps},
            2U,
            1e-6,
            1e-3);
}

TEST_P(ConnectionTest, ConvTransposeDoubleBackward) {
    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{5, 3};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};
    StackVector<int64_t, kMaxNdim> stride{3, 2};
    StackVector<int64_t, kMaxNdim> pad{2, 0};
    Shape out_dims{10, 7};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{in_channels, out_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};
    Shape out_shape{batch_size, out_channels};
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));

    Array x = (*testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2.0f, 1.0f).WithPadding(1)).RequireGrad();
    Array w = (*testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2.0f, 1.0f)).RequireGrad();
    Array b = (*testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f})).RequireGrad();

    Array go = (*testing::BuildArray(out_shape).WithLinearData(-0.3f, 0.1f).WithPadding(1)).RequireGrad();
    Array ggx = testing::BuildArray(x_shape).WithLinearData(0.4f, -0.2f).WithPadding(1);
    Array ggw = testing::BuildArray(w_shape).WithLinearData(-0.5f, 0.3f).WithPadding(1);
    Array ggb = testing::BuildArray(b_shape).WithLinearData(-0.6f, -0.4f).WithPadding(1);

    Array x_eps = Full(x.shape(), 1e3f);
    Array w_eps = Full(w.shape(), 1e3f);
    Array b_eps = Full(b.shape(), 1e3f);
    Array go_eps = Full(out_shape, 1e3f);

    CheckDoubleBackwardComputation(
            [&](const std::vector<Array>& xs) -> std::vector<Array> {
                const Array& xx = xs[0];
                const Array& ww = xs[1];
                const Array& bb = xs[2];
                Array y = ConvTranspose(xx, ww, bb, stride, pad, out_dims);
                return {y * y};
            },
            {x, w, b},
            {go},
            {ggx, ggw, ggb},
            {x_eps, w_eps, b_eps, go_eps},
            2,
            1e-2,
            1e-3);
}

TEST_P(ConnectionTest, Linear) {
    Array x = testing::BuildArray({2, 3}).WithLinearData(1.f).WithPadding(1);
    Array w = testing::BuildArray({4, 3}).WithLinearData(1.f).WithPadding(1);
    Array b = testing::BuildArray({4}).WithData<float>({3.f, 2.f, 1.f, -1.f}).WithPadding(2);
    Array a = Linear(x, w, b);
    Array e = Dot(x, w.Transpose()) + b;
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(ConnectionTest, LinearNoBias) {
    Array x = testing::BuildArray({2, 3}).WithLinearData(1.f).WithPadding(1);
    Array w = testing::BuildArray({4, 3}).WithLinearData(1.f).WithPadding(1);
    Array a = Linear(x, w);
    Array e = Dot(x, w.Transpose());
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(ConnectionTest, LinearSpecifyNBatchAxes) {
    Array x = testing::BuildArray({5, 4, 3, 2}).WithLinearData(1.f).WithPadding(1);
    Array w = testing::BuildArray({7, 6}).WithLinearData(1.f).WithPadding(1);
    Array b = testing::BuildArray({7}).WithData<float>({3.f, 2.f, -1.f, 3.f, 3.f, 7.f, 9.f}).WithPadding(2);
    Array a = Linear(x, w, b, 2);
    Array e = Dot(x.Reshape({20, 6}), w.Transpose()).Reshape({5, 4, 7}) + b.BroadcastTo({5, 4, 7});
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(ConnectionTest, LinearSpecifyNBatchAxesNoBias) {
    Array x = testing::BuildArray({5, 4, 3, 2}).WithLinearData(1.f).WithPadding(1);
    Array w = testing::BuildArray({7, 6}).WithLinearData(1.f).WithPadding(1);
    Array a = Linear(x, w, nonstd::nullopt, 2);
    Array e = Dot(x.Reshape({20, 6}), w.Transpose()).Reshape({5, 4, 7});
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(ConnectionTest, LinearSpecifyNBatchAxesEqualsZero) {
    Array x = testing::BuildArray({3, 2}).WithLinearData(1.f).WithPadding(1);
    Array w = testing::BuildArray({7, 6}).WithLinearData(1.f).WithPadding(1);
    Array a = Linear(x, w, nonstd::nullopt, 0);
    Array e = Dot(x.Reshape({6}), w.Transpose()).Reshape({7});
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(ConnectionTest, LinearReturnsZeros) {
    Array x = testing::BuildArray({2, 0}).WithLinearData(1.f).WithPadding(1);
    Array w = testing::BuildArray({4, 0}).WithLinearData(1.f).WithPadding(1);
    Array a = Linear(x, w);
    Array e = Zeros({2, 4}, x.dtype(), x.device());
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(ConnectionTest, LinearReturnsBias) {
    Array x = testing::BuildArray({2, 0}).WithLinearData(1.f).WithPadding(1);
    Array w = testing::BuildArray({4, 0}).WithLinearData(1.f).WithPadding(1);
    Array b = testing::BuildArray({4}).WithData<float>({3.f, 2.f, -1.f, 3.f}).WithPadding(2);
    Array a = Linear(x, w, b);
    Array e = b.BroadcastTo({2, 4});
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(ConnectionTest, LinearBackward) {
    Array x = (*testing::BuildArray({2, 3}).WithLinearData(1.f)).RequireGrad();
    Array w = (*testing::BuildArray({4, 3}).WithLinearData(2.f)).RequireGrad();
    Array b = (*testing::BuildArray({4}).WithLinearData(3.f)).RequireGrad();

    Array go = testing::BuildArray({2, 4}).WithLinearData(-0.1f, 0.1f).WithPadding(1);
    Array x_eps = Full(x.shape(), 1e-1f);
    Array w_eps = Full(w.shape(), 1e-1f);
    Array b_eps = Full(b.shape(), 1e-1f);

    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Linear(xs[0], xs[1], xs[2])}; },
            {x, w, b},
            {go},
            {x_eps, w_eps, b_eps});
}

TEST_P(ConnectionTest, LinearBackwardNoBias) {
    Array x = (*testing::BuildArray({2, 3}).WithLinearData(1.f)).RequireGrad();
    Array w = (*testing::BuildArray({4, 3}).WithLinearData(2.f)).RequireGrad();

    Array go = testing::BuildArray({2, 4}).WithLinearData(-0.1f, 0.1f).WithPadding(1);
    Array x_eps = Full(x.shape(), 1e-1f);
    Array w_eps = Full(w.shape(), 1e-1f);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Linear(xs[0], xs[1])}; }, {x, w}, {go}, {x_eps, w_eps});
}

TEST_P(ConnectionTest, LinearDoubleBackward) {
    Array x = (*testing::BuildArray({2, 3}).WithLinearData(1.f)).RequireGrad();
    Array w = (*testing::BuildArray({4, 3}).WithLinearData(2.f)).RequireGrad();
    Array b = (*testing::BuildArray({4}).WithLinearData(-3.f)).RequireGrad();
    Array go = (*testing::BuildArray({2, 4}).WithLinearData(-0.1f, 0.1f).WithPadding(1)).RequireGrad();

    Array ggx = testing::BuildArray(x.shape()).WithLinearData(-0.3f, 0.1f).WithPadding(1);
    Array ggw = testing::BuildArray(w.shape()).WithLinearData(-0.2f, 0.1f).WithPadding(1);
    Array ggb = testing::BuildArray(b.shape()).WithLinearData(-0.4f, 0.1f).WithPadding(1);
    Array x_eps = Full(x.shape(), 1e-1f);
    Array w_eps = Full(w.shape(), 1e-1f);
    Array b_eps = Full(b.shape(), 1e-1f);
    Array go_eps = Full(go.shape(), 1e-1f);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Linear(xs[0], xs[1], xs[2]);
                return {y * y};
            },
            {x, w, b},
            {go},
            {ggx, ggw, ggb},
            {x_eps, w_eps, b_eps, go_eps},
            2U,
            1e-2,
            1e-3);
}

TEST_P(ConnectionTest, LinearDoubleBackwardNoBias) {
    Array x = (*testing::BuildArray({2, 3}).WithLinearData(1.f)).RequireGrad();
    Array w = (*testing::BuildArray({4, 3}).WithLinearData(2.f)).RequireGrad();
    Array go = (*testing::BuildArray({2, 4}).WithLinearData(-0.1f, 0.1f).WithPadding(1)).RequireGrad();

    Array ggx = testing::BuildArray(x.shape()).WithLinearData(-0.3f, 0.1f).WithPadding(1);
    Array ggw = testing::BuildArray(w.shape()).WithLinearData(-0.2f, 0.1f).WithPadding(1);
    Array x_eps = Full(x.shape(), 1e-1f);
    Array w_eps = Full(w.shape(), 1e-1f);
    Array go_eps = Full(go.shape(), 1e-1f);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Linear(xs[0], xs[1]);
                return {y * y};
            },
            {x, w},
            {go},
            {ggx, ggw},
            {x_eps, w_eps, go_eps},
            2U,
            1e-2,
            1e-3);
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        ConnectionTest,
        ::testing::Values(
#ifdef CHAINERX_ENABLE_CUDA
                std::string{"cuda"},
#endif  // CHAINERX_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace chainerx
