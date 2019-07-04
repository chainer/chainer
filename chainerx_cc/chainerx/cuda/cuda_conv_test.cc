#include "chainerx/cuda/cuda_conv.h"

#include <algorithm>
#include <cstdint>

#include <gtest/gtest.h>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/cuda/cuda_device.h"
#include "chainerx/device_id.h"
#include "chainerx/dims.h"
#include "chainerx/kernels/connection.h"
#include "chainerx/routines/connection.h"
#include "chainerx/shape.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/array_check.h"
#include "chainerx/testing/device_session.h"

namespace chainerx {
namespace cuda {
namespace cuda_internal {

class CudaConvTest {
public:
    static size_t GetFwdAlgoCacheMapSize(const CudaConv& cuda_conv) { return cuda_conv.fwd_algo_cache_map_.size(); }
    static size_t GetBwdDataAlgoCacheMapSize(const CudaConv& cuda_conv) { return cuda_conv.bwd_data_algo_cache_map_.size(); }
    static size_t GetBwdFilterAlgoCacheMapSize(const CudaConv& cuda_conv) { return cuda_conv.bwd_filter_algo_cache_map_.size(); }
    static CudaConv& GetCudaConv(CudaDevice& cuda_device) {
        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(cuda_device);
        return device_internals.cuda_conv();
    }
};

}  // namespace cuda_internal

TEST(CudaConvTest, FwdAlgoCache) {
    testing::DeviceSession device_session{DeviceId{"cuda", 0}};
    auto& device = static_cast<CudaDevice&>(device_session.device());  // NOLINT
    cuda_internal::CudaConv& cuda_conv = cuda_internal::CudaConvTest::GetCudaConv(device);

    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{10, 7};
    Dims kernel_size{2, 3};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{out_channels, in_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};

    Array x = testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2.0f, 1.0f).WithPadding(1);
    Array w = testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2.0f, 1.0f);
    Array b = testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f});

    // New parameters should create new auto tuning caches, and same parameters should not.
    {
        Dims stride{3, 2};
        Dims pad{2, 0};
        bool cover_all = false;

        EXPECT_EQ(size_t{0}, cuda_internal::CudaConvTest::GetFwdAlgoCacheMapSize(cuda_conv));
        cuda_conv.Conv(device, x, w, b, stride, pad, cover_all, x.dtype());
        EXPECT_EQ(size_t{1}, cuda_internal::CudaConvTest::GetFwdAlgoCacheMapSize(cuda_conv));
        cuda_conv.Conv(device, x, w, b, stride, pad, cover_all, x.dtype());
        EXPECT_EQ(size_t{1}, cuda_internal::CudaConvTest::GetFwdAlgoCacheMapSize(cuda_conv));
    }
    {
        Dims stride{1, 1};
        Dims pad{0, 0};
        bool cover_all = false;

        EXPECT_EQ(size_t{1}, cuda_internal::CudaConvTest::GetFwdAlgoCacheMapSize(cuda_conv));
        Conv(x, w, b, stride, pad, cover_all);
        EXPECT_EQ(size_t{2}, cuda_internal::CudaConvTest::GetFwdAlgoCacheMapSize(cuda_conv));
        Conv(x, w, b, stride, pad, cover_all);
        EXPECT_EQ(size_t{2}, cuda_internal::CudaConvTest::GetFwdAlgoCacheMapSize(cuda_conv));
    }
}

TEST(CudaConvTest, BwdDatadAlgoCache) {
    testing::DeviceSession device_session{DeviceId{"cuda", 0}};
    auto& device = static_cast<CudaDevice&>(device_session.device());  // NOLINT
    cuda_internal::CudaConv& cuda_conv = cuda_internal::CudaConvTest::GetCudaConv(device);

    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{5, 3};
    Dims kernel_size{2, 3};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{in_channels, out_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};

    Array x = testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2.0f, 1.0f).WithPadding(1);
    Array w = testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2.0f, 1.0f);
    Array b = testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f});

    // New parameters should create new auto tuning caches, and same parameters should not.
    {
        Dims stride{3, 2};
        Dims pad{2, 0};

        EXPECT_EQ(size_t{0}, cuda_internal::CudaConvTest::GetBwdDataAlgoCacheMapSize(cuda_conv));
        ConvTranspose(x, w, b, stride, pad);
        EXPECT_EQ(size_t{1}, cuda_internal::CudaConvTest::GetBwdDataAlgoCacheMapSize(cuda_conv));
        ConvTranspose(x, w, b, stride, pad);
        EXPECT_EQ(size_t{1}, cuda_internal::CudaConvTest::GetBwdDataAlgoCacheMapSize(cuda_conv));
    }
    {
        Dims stride{1, 1};
        Dims pad{0, 0};

        EXPECT_EQ(size_t{1}, cuda_internal::CudaConvTest::GetBwdDataAlgoCacheMapSize(cuda_conv));
        ConvTranspose(x, w, b, stride, pad);
        EXPECT_EQ(size_t{2}, cuda_internal::CudaConvTest::GetBwdDataAlgoCacheMapSize(cuda_conv));
        ConvTranspose(x, w, b, stride, pad);
        EXPECT_EQ(size_t{2}, cuda_internal::CudaConvTest::GetBwdDataAlgoCacheMapSize(cuda_conv));
    }
}

TEST(CudaConvTest, BwdFilterAlgoCache) {
    testing::DeviceSession device_session{DeviceId{"cuda", 0}};
    auto& device = static_cast<CudaDevice&>(device_session.device());  // NOLINT
    cuda_internal::CudaConv& cuda_conv = cuda_internal::CudaConvTest::GetCudaConv(device);

    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{10, 7};
    Dims kernel_size{2, 3};

    Dtype w_dtype = Dtype::kFloat32;
    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{out_channels, in_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));

    Array x = testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2.0f, 1.0f).WithPadding(1);

    // New parameters should create new auto tuning caches, and same parameters should not.
    // ConvGradW is not exposed as routines function, so call CudaDevice::ConvGradWeight directly.
    {
        Dims stride{3, 2};
        Dims pad{2, 0};
        bool cover_all = false;

        Shape out_dims{5, 3};
        Shape out_shape{batch_size, out_channels};
        std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));
        Array gy = testing::BuildArray(out_shape).WithLinearData(-0.3f, 0.1f).WithPadding(1);

        EXPECT_EQ(size_t{0}, cuda_internal::CudaConvTest::GetBwdFilterAlgoCacheMapSize(cuda_conv));
        device.backend().CallKernel<ConvGradWeightKernel>(w_dtype, w_shape, x, gy, stride, pad, cover_all, absl::nullopt);
        EXPECT_EQ(size_t{1}, cuda_internal::CudaConvTest::GetBwdFilterAlgoCacheMapSize(cuda_conv));
        device.backend().CallKernel<ConvGradWeightKernel>(w_dtype, w_shape, x, gy, stride, pad, cover_all, absl::nullopt);
        EXPECT_EQ(size_t{1}, cuda_internal::CudaConvTest::GetBwdFilterAlgoCacheMapSize(cuda_conv));
    }
    {
        Dims stride{1, 1};
        Dims pad{0, 0};
        bool cover_all = false;

        Shape out_dims{9, 5};
        Shape out_shape{batch_size, out_channels};
        std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));
        Array gy = testing::BuildArray(out_shape).WithLinearData(-0.3f, 0.1f).WithPadding(1);

        EXPECT_EQ(size_t{1}, cuda_internal::CudaConvTest::GetBwdFilterAlgoCacheMapSize(cuda_conv));
        device.backend().CallKernel<ConvGradWeightKernel>(w_dtype, w_shape, x, gy, stride, pad, cover_all, absl::nullopt);
        EXPECT_EQ(size_t{2}, cuda_internal::CudaConvTest::GetBwdFilterAlgoCacheMapSize(cuda_conv));
        device.backend().CallKernel<ConvGradWeightKernel>(w_dtype, w_shape, x, gy, stride, pad, cover_all, absl::nullopt);
        EXPECT_EQ(size_t{2}, cuda_internal::CudaConvTest::GetBwdFilterAlgoCacheMapSize(cuda_conv));
    }
}

}  // namespace cuda
}  // namespace chainerx
