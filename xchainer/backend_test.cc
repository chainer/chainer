#include "xchainer/backend.h"

#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/native_backend.h"

namespace xchainer {
namespace {

TEST(BackendTest, name) {
    EXPECT_EQ("native", NativeBackend().name());
#ifdef XCHAINER_ENABLE_CUDA
    EXPECT_EQ("cuda", cuda::CudaBackend().name());
#endif  // XCHAINER_ENABLE_CUDA
}

}  // namespace
}  // namespace xchainer
