#include "xchainer/routines/connection.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/device_id.h"
#include "xchainer/shape.h"
#include "xchainer/stack_vector.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
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

TEST_P(ConnectionTest, Convolution) {
    if (GetParam() == "cuda") {
        // TODO(niboshi): Add CUDA implementation
        return;
    }
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

    Array x = testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2, 1.0f).WithPadding(1);
    Array w = testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2, 1.0f);
    Array b = testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f});
    Array y = Convolution(x, w, b, stride, pad, cover_all);

    Array e = testing::BuildArray(out_shape).WithData<float>(
            {-2.00000e-01, -2.00000e-01, -2.00000e-01, 2.71198e+04,  2.67778e+04,  2.64358e+04,  2.35288e+04,  2.31868e+04,  2.28448e+04,
             1.99378e+04,  1.95958e+04,  1.92538e+04,  -2.00000e-01, -2.00000e-01, -2.00000e-01, 1.30000e+00,  1.30000e+00,  1.30000e+00,
             -1.45127e+04, -1.42067e+04, -1.39007e+04, -1.12997e+04, -1.09937e+04, -1.06877e+04, -8.08670e+03, -7.78070e+03, -7.47470e+03,
             1.30000e+00,  1.30000e+00,  1.30000e+00,  -2.00000e-01, -2.00000e-01, -2.00000e-01, -8.79020e+03, -9.13220e+03, -9.47420e+03,
             -1.23812e+04, -1.27232e+04, -1.30652e+04, -1.59722e+04, -1.63142e+04, -1.66562e+04, -2.00000e-01, -2.00000e-01, -2.00000e-01,
             1.30000e+00,  1.30000e+00,  1.30000e+00,  1.76173e+04,  1.79233e+04,  1.82293e+04,  2.08303e+04,  2.11363e+04,  2.14423e+04,
             2.40433e+04,  2.43493e+04,  2.46553e+04,  1.30000e+00,  1.30000e+00,  1.30000e+00});
    testing::ExpectEqual(e, y);
}

TEST_P(ConnectionTest, ConvolutionCoverAll) {
    if (GetParam() == "cuda") {
        // TODO(niboshi): Add CUDA implementation
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

    Array x = testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2, 1.0f).WithPadding(1);
    Array w = testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2, 1.0f);
    Array b = testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f});
    Array y = Convolution(x, w, b, stride, pad, cover_all);

    Array e = testing::BuildArray(out_shape).WithData<float>(
            {-2.00000e-01, -2.00000e-01, -2.00000e-01, -2.00000e-01, 3.10168e+04,  3.06748e+04,  3.03328e+04,  2.08948e+04,  2.69128e+04,
             2.65708e+04,  2.62288e+04,  1.80148e+04,  2.28088e+04,  2.24668e+04,  2.21248e+04,  1.51348e+04,  -2.00000e-01, -2.00000e-01,
             -2.00000e-01, -2.00000e-01, 1.30000e+00,  1.30000e+00,  1.30000e+00,  1.30000e+00,  -1.66097e+04, -1.63037e+04, -1.59977e+04,
             -9.66770e+03, -1.29377e+04, -1.26317e+04, -1.23257e+04, -7.36370e+03, -9.26570e+03, -8.95970e+03, -8.65370e+03, -5.05970e+03,
             1.30000e+00,  1.30000e+00,  1.30000e+00,  1.30000e+00,  -2.00000e-01, -2.00000e-01, -2.00000e-01, -2.00000e-01, -1.00232e+04,
             -1.03652e+04, -1.07072e+04, -7.90520e+03, -1.41272e+04, -1.44692e+04, -1.48112e+04, -1.07852e+04, -1.82312e+04, -1.85732e+04,
             -1.89152e+04, -1.36652e+04, -2.00000e-01, -2.00000e-01, -2.00000e-01, -2.00000e-01, 1.30000e+00,  1.30000e+00,  1.30000e+00,
             1.30000e+00,  2.01103e+04,  2.04163e+04,  2.07223e+04,  1.33723e+04,  2.37823e+04,  2.40883e+04,  2.43943e+04,  1.56763e+04,
             2.74543e+04,  2.77603e+04,  2.80663e+04,  1.79803e+04,  1.30000e+00,  1.30000e+00,  1.30000e+00,  1.30000e+00});
    std::cout << y << std::endl;
    testing::ExpectEqual(e, y);
}

TEST_P(ConnectionTest, ConvolutionBackward) {
    // TODO(niboshi): Implement
}

TEST_P(ConnectionTest, ConvolutionDoubleBackward) {
    // TODO(niboshi): Implement
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        ConnectionTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace xchainer
