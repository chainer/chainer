#include "xchainer/cuda/cuda_backend.h"

#include <gtest/gtest.h>

namespace xchainer {
namespace cuda {
namespace {

TEST(CudaBackendTest, name) { EXPECT_EQ("cuda", cuda::CudaBackend().name()); }

}  // namespace
}  // namespace cuda
}  // namespace xchainer
