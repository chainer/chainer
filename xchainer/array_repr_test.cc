#include "xchainer/array_repr.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "xchainer/context.h"
#include "xchainer/device.h"
#include "xchainer/native_backend.h"
#include "xchainer/native_device.h"

namespace xchainer {
namespace {

template <typename T>
void CheckArrayRepr(const std::string& expected, const std::vector<T>& data_vec, Shape shape, Device& device,
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

TEST(ArrayReprTest, AllDtypesOnNativeBackend) {
    Context ctx;
    NativeBackend backend{ctx};
    NativeDevice device{backend, 0};

    // bool
    CheckArrayRepr<bool>("array([False], dtype=bool, device='native:0')", {false}, Shape({1}), device);
    CheckArrayRepr<bool>("array([ True], dtype=bool, device='native:0')", {true}, Shape({1}), device);
    CheckArrayRepr<bool>(
        "array([[False,  True,  True],\n"
        "       [ True, False,  True]], dtype=bool, device='native:0')",
        {false, true, true, true, false, true}, Shape({2, 3}), device);
    CheckArrayRepr<bool>("array([[[[ True]]]], dtype=bool, device='native:0')", {true}, Shape({1, 1, 1, 1}), device);

    // int8
    CheckArrayRepr<int8_t>("array([0], dtype=int8, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<int8_t>("array([-2], dtype=int8, device='native:0')", {-2}, Shape({1}), device);
    CheckArrayRepr<int8_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=int8, device='native:0')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int8_t>(
        "array([[ 0,  1,  2],\n"
        "       [-3,  4,  5]], dtype=int8, device='native:0')",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int8_t>("array([[[[3]]]], dtype=int8, device='native:0')", {3}, Shape({1, 1, 1, 1}), device);

    // int16
    CheckArrayRepr<int16_t>("array([0], dtype=int16, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<int16_t>("array([-2], dtype=int16, device='native:0')", {-2}, Shape({1}), device);
    CheckArrayRepr<int16_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=int16, device='native:0')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int16_t>(
        "array([[ 0,  1,  2],\n"
        "       [-3,  4,  5]], dtype=int16, device='native:0')",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int16_t>("array([[[[3]]]], dtype=int16, device='native:0')", {3}, Shape({1, 1, 1, 1}), device);

    // int32
    CheckArrayRepr<int32_t>("array([0], dtype=int32, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<int32_t>("array([-2], dtype=int32, device='native:0')", {-2}, Shape({1}), device);
    CheckArrayRepr<int32_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=int32, device='native:0')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int32_t>(
        "array([[ 0,  1,  2],\n"
        "       [-3,  4,  5]], dtype=int32, device='native:0')",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int32_t>("array([[[[3]]]], dtype=int32, device='native:0')", {3}, Shape({1, 1, 1, 1}), device);

    // int64
    CheckArrayRepr<int64_t>("array([0], dtype=int64, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<int64_t>("array([-2], dtype=int64, device='native:0')", {-2}, Shape({1}), device);
    CheckArrayRepr<int64_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=int64, device='native:0')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int64_t>(
        "array([[ 0,  1,  2],\n"
        "       [-3,  4,  5]], dtype=int64, device='native:0')",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int64_t>("array([[[[3]]]], dtype=int64, device='native:0')", {3}, Shape({1, 1, 1, 1}), device);

    // uint8
    CheckArrayRepr<uint8_t>("array([0], dtype=uint8, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<uint8_t>("array([2], dtype=uint8, device='native:0')", {2}, Shape({1}), device);
    CheckArrayRepr<uint8_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=uint8, device='native:0')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<uint8_t>("array([[[[3]]]], dtype=uint8, device='native:0')", {3}, Shape({1, 1, 1, 1}), device);

    // float32
    CheckArrayRepr<float>("array([0.], dtype=float32, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<float>("array([3.25], dtype=float32, device='native:0')", {3.25}, Shape({1}), device);
    CheckArrayRepr<float>("array([-3.25], dtype=float32, device='native:0')", {-3.25}, Shape({1}), device);
    CheckArrayRepr<float>("array([ inf], dtype=float32, device='native:0')", {std::numeric_limits<float>::infinity()}, Shape({1}), device);
    CheckArrayRepr<float>("array([ -inf], dtype=float32, device='native:0')", {-std::numeric_limits<float>::infinity()}, Shape({1}),
                          device);
    CheckArrayRepr<float>("array([ nan], dtype=float32, device='native:0')", {std::nanf("")}, Shape({1}), device);
    CheckArrayRepr<float>(
        "array([[0., 1., 2.],\n"
        "       [3., 4., 5.]], dtype=float32, device='native:0')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<float>(
        "array([[0.  , 1.  , 2.  ],\n"
        "       [3.25, 4.  , 5.  ]], dtype=float32, device='native:0')",
        {0, 1, 2, 3.25, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<float>(
        "array([[ 0.  ,  1.  ,  2.  ],\n"
        "       [-3.25,  4.  ,  5.  ]], dtype=float32, device='native:0')",
        {0, 1, 2, -3.25, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<float>("array([[[[3.25]]]], dtype=float32, device='native:0')", {3.25}, Shape({1, 1, 1, 1}), device);

    // float64
    CheckArrayRepr<double>("array([0.], dtype=float64, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<double>("array([3.25], dtype=float64, device='native:0')", {3.25}, Shape({1}), device);
    CheckArrayRepr<double>("array([-3.25], dtype=float64, device='native:0')", {-3.25}, Shape({1}), device);
    CheckArrayRepr<double>("array([ inf], dtype=float64, device='native:0')", {std::numeric_limits<double>::infinity()}, Shape({1}),
                           device);
    CheckArrayRepr<double>("array([ -inf], dtype=float64, device='native:0')", {-std::numeric_limits<double>::infinity()}, Shape({1}),
                           device);
    CheckArrayRepr<double>("array([ nan], dtype=float64, device='native:0')", {std::nan("")}, Shape({1}), device);
    CheckArrayRepr<double>(
        "array([[0., 1., 2.],\n"
        "       [3., 4., 5.]], dtype=float64, device='native:0')",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<double>(
        "array([[0.  , 1.  , 2.  ],\n"
        "       [3.25, 4.  , 5.  ]], dtype=float64, device='native:0')",
        {0, 1, 2, 3.25, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<double>(
        "array([[ 0.  ,  1.  ,  2.  ],\n"
        "       [-3.25,  4.  ,  5.  ]], dtype=float64, device='native:0')",
        {0, 1, 2, -3.25, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<double>("array([[[[3.25]]]], dtype=float64, device='native:0')", {3.25}, Shape({1, 1, 1, 1}), device);

    // Single graph
    CheckArrayRepr<int32_t>("array([-2], dtype=int32, device='native:0', graph_ids=['graph_1'])", {-2}, Shape({1}), device, {"graph_1"});

    // Two graphs
    CheckArrayRepr<int32_t>("array([1], dtype=int32, device='native:0', graph_ids=['graph_1', 'graph_2'])", {1}, Shape({1}), device,
                            {"graph_1", "graph_2"});

    // Multiple graphs
    CheckArrayRepr<int32_t>(
        "array([-9], dtype=int32, device='native:0', graph_ids=['graph_1', 'graph_2', 'graph_3', 'graph_4', 'graph_5'])", {-9}, Shape({1}),
        device, {"graph_1", "graph_2", "graph_3", "graph_4", "graph_5"});
}

}  // namespace
}  // namespace xchainer
