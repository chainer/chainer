#include "xchainer/cuda/cuda_backend.h"

#include <gtest/gtest.h>

namespace xchainer {
namespace cuda {
namespace {

TEST(CudaBackendTest, GetName) { EXPECT_EQ("cuda", cuda::CudaBackend().GetName()); }

}  // namespace
}  // namespace cuda
}  // namespace xchainer
