#include "xchainer/array_repr.h"

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "xchainer/backend.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/native_backend.h"

namespace xchainer {
namespace {

template <typename T>
void CheckArrayRepr(const std::string& expected, const std::vector<T>& data_vec, Shape shape, const Device& device,
                    const std::vector<GraphId> graph_ids = {}) {
    // Copy to a contiguous memory block because std::vector<bool> is not packed as a sequence of bool's.
    std::shared_ptr<T> data_ptr = std::make_unique<T[]>(data_vec.size());
    std::copy(data_vec.begin(), data_vec.end(), data_ptr.get());
    Array array = Array::FromBuffer(shape, TypeToDtype<T>, static_cast<std::shared_ptr<void>>(data_ptr), device);
    for (const GraphId& graph_id : graph_ids) {
        array.RequireGrad(graph_id);
    }

    // std::string version
    EXPECT_EQ(expected, ArrayRepr(array));

    // std::ostream version
    std::ostringstream os;
    os << array;
    EXPECT_EQ(expected, os.str());
}

TEST(ArrayReprTest, NativeBackend) {
    NativeBackend backend;

    // bool
    CheckArrayRepr<bool>("array([False], dtype=bool, device='cpu')", {false}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<bool>("array([ True], dtype=bool, device='cpu')", {true}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<bool>(
        "array([[False,  True,  True],\n"
        "       [ True, False,  True]], dtype=bool, device='cpu')",
        {false, true, true, true, false, true}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<bool>("array([[[[ True]]]], dtype=bool, device='cpu')", {true}, Shape({1, 1, 1, 1}), Device{"cpu", &backend});

    // int8
    CheckArrayRepr<int8_t>("array([0], dtype=int8, device='cpu')", {0}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<int8_t>("array([-2], dtype=int8, device='cpu')", {-2}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<int8_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=int8, device='cpu')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<int8_t>(
        "array([[ 0,  1,  2],\n"
        "       [-3,  4,  5]], dtype=int8, device='cpu')",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<int8_t>("array([[[[3]]]], dtype=int8, device='cpu')", {3}, Shape({1, 1, 1, 1}), Device{"cpu", &backend});

    // int16
    CheckArrayRepr<int16_t>("array([0], dtype=int16, device='cpu')", {0}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<int16_t>("array([-2], dtype=int16, device='cpu')", {-2}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<int16_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=int16, device='cpu')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<int16_t>(
        "array([[ 0,  1,  2],\n"
        "       [-3,  4,  5]], dtype=int16, device='cpu')",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<int16_t>("array([[[[3]]]], dtype=int16, device='cpu')", {3}, Shape({1, 1, 1, 1}), Device{"cpu", &backend});

    // int32
    CheckArrayRepr<int32_t>("array([0], dtype=int32, device='cpu')", {0}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<int32_t>("array([-2], dtype=int32, device='cpu')", {-2}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<int32_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=int32, device='cpu')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<int32_t>(
        "array([[ 0,  1,  2],\n"
        "       [-3,  4,  5]], dtype=int32, device='cpu')",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<int32_t>("array([[[[3]]]], dtype=int32, device='cpu')", {3}, Shape({1, 1, 1, 1}), Device{"cpu", &backend});

    // int64
    CheckArrayRepr<int64_t>("array([0], dtype=int64, device='cpu')", {0}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<int64_t>("array([-2], dtype=int64, device='cpu')", {-2}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<int64_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=int64, device='cpu')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<int64_t>(
        "array([[ 0,  1,  2],\n"
        "       [-3,  4,  5]], dtype=int64, device='cpu')",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<int64_t>("array([[[[3]]]], dtype=int64, device='cpu')", {3}, Shape({1, 1, 1, 1}), Device{"cpu", &backend});

    // uint8
    CheckArrayRepr<uint8_t>("array([0], dtype=uint8, device='cpu')", {0}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<uint8_t>("array([2], dtype=uint8, device='cpu')", {2}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<uint8_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=uint8, device='cpu')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<uint8_t>("array([[[[3]]]], dtype=uint8, device='cpu')", {3}, Shape({1, 1, 1, 1}), Device{"cpu", &backend});

    // float32
    CheckArrayRepr<float>("array([0.], dtype=float32, device='cpu')", {0}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<float>("array([3.25], dtype=float32, device='cpu')", {3.25}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<float>("array([-3.25], dtype=float32, device='cpu')", {-3.25}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<float>("array([ inf], dtype=float32, device='cpu')", {std::numeric_limits<float>::infinity()}, Shape({1}),
                          Device{"cpu", &backend});
    CheckArrayRepr<float>("array([ -inf], dtype=float32, device='cpu')", {-std::numeric_limits<float>::infinity()}, Shape({1}),
                          Device{"cpu", &backend});
    CheckArrayRepr<float>("array([ nan], dtype=float32, device='cpu')", {std::nanf("")}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<float>(
        "array([[0., 1., 2.],\n"
        "       [3., 4., 5.]], dtype=float32, device='cpu')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<float>(
        "array([[0.  , 1.  , 2.  ],\n"
        "       [3.25, 4.  , 5.  ]], dtype=float32, device='cpu')",
        {0, 1, 2, 3.25, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<float>(
        "array([[ 0.  ,  1.  ,  2.  ],\n"
        "       [-3.25,  4.  ,  5.  ]], dtype=float32, device='cpu')",
        {0, 1, 2, -3.25, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<float>("array([[[[3.25]]]], dtype=float32, device='cpu')", {3.25}, Shape({1, 1, 1, 1}), Device{"cpu", &backend});

    // float64
    CheckArrayRepr<double>("array([0.], dtype=float64, device='cpu')", {0}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<double>("array([3.25], dtype=float64, device='cpu')", {3.25}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<double>("array([-3.25], dtype=float64, device='cpu')", {-3.25}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<double>("array([ inf], dtype=float64, device='cpu')", {std::numeric_limits<double>::infinity()}, Shape({1}),
                           Device{"cpu", &backend});
    CheckArrayRepr<double>("array([ -inf], dtype=float64, device='cpu')", {-std::numeric_limits<double>::infinity()}, Shape({1}),
                           Device{"cpu", &backend});
    CheckArrayRepr<double>("array([ nan], dtype=float64, device='cpu')", {std::nan("")}, Shape({1}), Device{"cpu", &backend});
    CheckArrayRepr<double>(
        "array([[0., 1., 2.],\n"
        "       [3., 4., 5.]], dtype=float64, device='cpu')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<double>(
        "array([[0.  , 1.  , 2.  ],\n"
        "       [3.25, 4.  , 5.  ]], dtype=float64, device='cpu')",
        {0, 1, 2, 3.25, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<double>(
        "array([[ 0.  ,  1.  ,  2.  ],\n"
        "       [-3.25,  4.  ,  5.  ]], dtype=float64, device='cpu')",
        {0, 1, 2, -3.25, 4, 5}, Shape({2, 3}), Device{"cpu", &backend});
    CheckArrayRepr<double>("array([[[[3.25]]]], dtype=float64, device='cpu')", {3.25}, Shape({1, 1, 1, 1}), Device{"cpu", &backend});

    // Single graph
    CheckArrayRepr<int32_t>("array([-2], dtype=int32, device='cpu', graph_ids=['graph_1'])", {-2}, Shape({1}), Device{"cpu", &backend},
                            {"graph_1"});

    // Two graphs
    CheckArrayRepr<int32_t>("array([1], dtype=int32, device='cpu', graph_ids=['graph_1', 'graph_2'])", {1}, Shape({1}),
                            Device{"cpu", &backend}, {"graph_1", "graph_2"});

    // Multiple graphs
    CheckArrayRepr<int32_t>("array([-9], dtype=int32, device='cpu', graph_ids=['graph_1', 'graph_2', 'graph_3', 'graph_4', 'graph_5'])",
                            {-9}, Shape({1}), Device{"cpu", &backend}, {"graph_1", "graph_2", "graph_3", "graph_4", "graph_5"});
}

#ifdef XCHAINER_ENABLE_CUDA
TEST(ArrayReprTest, CudaBackend) {
    cuda::CudaBackend backend;

    // Randomly picked checks for CUDA
    CheckArrayRepr<bool>("array([False], dtype=bool, device='cuda')", {false}, Shape({1}), Device{"cuda", &backend});
    CheckArrayRepr<double>(
        "array([[ 0.  ,  1.  ,  2.  ],\n"
        "       [-3.25,  4.  ,  5.  ]], dtype=float64, device='cuda')",
        {0, 1, 2, -3.25, 4, 5}, Shape({2, 3}), Device{"cuda", &backend});
    CheckArrayRepr<int32_t>("array([1], dtype=int32, device='cuda', graph_ids=['graph_1', 'graph_2'])", {1}, Shape({1}),
                            Device{"cuda", &backend}, {"graph_1", "graph_2"});
}
#endif  // XCHAINER_ENABLE_CUDA

}  // namespace
}  // namespace xchainer
