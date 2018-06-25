#ifdef XCHAINER_ENABLE_CUDA

#include "xchainer/cuda/cuda_conv.h"

#include <cstdint>

#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/cuda/cuda_conv.h"
#include "xchainer/cuda/cuda_device.h"
#include "xchainer/device_id.h"
#include "xchainer/routines/connection.h"
#include "xchainer/shape.h"
#include "xchainer/stack_vector.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace cuda {
namespace internal {

CudaConv& GetCudaConv(CudaDevice& device) { return device.cuda_conv_; }

size_t GetFwdAlgoCacheMapSize(const CudaConv& cuda_conv) { return cuda_conv.fwd_algo_cache_map_.size(); }
size_t GetBwdDataAlgoCacheMapSize(const CudaConv& cuda_conv) { return cuda_conv.bwd_data_algo_cache_map_.size(); }
size_t GetBwdFilterAlgoCacheMapSize(const CudaConv& cuda_conv) { return cuda_conv.bwd_filter_algo_cache_map_.size(); }

}  // namespace internal

TEST(CudaConvTest, FwdAlgoCache) {
    testing::DeviceSession device_session{DeviceId{"cuda", 0}};
    CudaDevice& device = static_cast<CudaDevice&>(device_session.device());
    internal::CudaConv& cuda_conv = internal::GetCudaConv(device);

    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{10, 7};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{out_channels, in_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};

    Array x = testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2, 1.0f).WithPadding(1);
    Array w = testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2, 1.0f);
    Array b = testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f});

    // New parameters should create new auto tuning caches, and same parameters should not.
    {
        StackVector<int64_t, kMaxNdim> stride{3, 2};
        StackVector<int64_t, kMaxNdim> pad{2, 0};
        bool cover_all = false;

        EXPECT_EQ(size_t{0}, GetFwdAlgoCacheMapSize(cuda_conv));
        cuda_conv.Conv(device, x, w, b, stride, pad, cover_all);
        EXPECT_EQ(size_t{1}, GetFwdAlgoCacheMapSize(cuda_conv));
        cuda_conv.Conv(device, x, w, b, stride, pad, cover_all);
        EXPECT_EQ(size_t{1}, GetFwdAlgoCacheMapSize(cuda_conv));
    }
    {
        StackVector<int64_t, kMaxNdim> stride{1, 1};
        StackVector<int64_t, kMaxNdim> pad{0, 0};
        bool cover_all = false;

        EXPECT_EQ(size_t{1}, GetFwdAlgoCacheMapSize(cuda_conv));
        Conv(x, w, b, stride, pad, cover_all);
        EXPECT_EQ(size_t{2}, GetFwdAlgoCacheMapSize(cuda_conv));
        Conv(x, w, b, stride, pad, cover_all);
        EXPECT_EQ(size_t{2}, GetFwdAlgoCacheMapSize(cuda_conv));
    }
}

TEST(CudaConvTest, BwdDatadAlgoCache) {
    testing::DeviceSession device_session{DeviceId{"cuda", 0}};
    CudaDevice& device = static_cast<CudaDevice&>(device_session.device());
    internal::CudaConv& cuda_conv = internal::GetCudaConv(device);

    int64_t batch_size = 2;
    int64_t in_channels = 3;
    int64_t out_channels = 2;
    Shape in_dims{5, 3};
    StackVector<int64_t, kMaxNdim> kernel_size{2, 3};

    Shape x_shape{batch_size, in_channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape w_shape{in_channels, out_channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(w_shape));
    Shape b_shape{out_channels};

    Array x = testing::BuildArray(x_shape).WithLinearData<float>(-x_shape.GetTotalSize() / 2, 1.0f).WithPadding(1);
    Array w = testing::BuildArray(w_shape).WithLinearData<float>(-w_shape.GetTotalSize() / 2, 1.0f);
    Array b = testing::BuildArray(b_shape).WithData<float>({-0.2f, 1.3f});

    // New parameters should create new auto tuning caches, and same parameters should not.
    {
        StackVector<int64_t, kMaxNdim> stride{3, 2};
        StackVector<int64_t, kMaxNdim> pad{2, 0};

        EXPECT_EQ(size_t{0}, GetBwdDataAlgoCacheMapSize(cuda_conv));
        ConvTranspose(x, w, b, stride, pad);
        EXPECT_EQ(size_t{1}, GetBwdDataAlgoCacheMapSize(cuda_conv));
        ConvTranspose(x, w, b, stride, pad);
        EXPECT_EQ(size_t{1}, GetBwdDataAlgoCacheMapSize(cuda_conv));
    }
    {
        StackVector<int64_t, kMaxNdim> stride{1, 1};
        StackVector<int64_t, kMaxNdim> pad{0, 0};

        EXPECT_EQ(size_t{1}, GetBwdDataAlgoCacheMapSize(cuda_conv));
        ConvTranspose(x, w, b, stride, pad);
        EXPECT_EQ(size_t{2}, GetBwdDataAlgoCacheMapSize(cuda_conv));
        ConvTranspose(x, w, b, stride, pad);
        EXPECT_EQ(size_t{2}, GetBwdDataAlgoCacheMapSize(cuda_conv));
    }
}

}  // namespace cuda
}  // namespace xchainer

#endif  // XCHAINER_ENABLE_CUDA
