#include "xchainer/array_repr.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "xchainer/device.h"

namespace xchainer {
namespace {

template <typename T>
void CheckArrayReprWithCurrentDevice(const std::string& expected, const std::vector<T>& data_vec, const Shape& shape,
                                     const std::vector<GraphId>& graph_ids) {
    // Copy to a contiguous memory block because std::vector<bool> is not packed as a sequence of bool's.
    std::shared_ptr<T> data_ptr = std::make_unique<T[]>(data_vec.size());
    std::copy(data_vec.begin(), data_vec.end(), data_ptr.get());
    Array array = Array::FromBuffer(shape, TypeToDtype<T>, static_cast<std::shared_ptr<void>>(data_ptr));

    for (const GraphId& graph_id) {
        array.RequireGrad(graph_id);
    }

    // std::string version
    EXPECT_EQ(expected, ArrayRepr(array));

    // std::ostream version
    std::ostringstream os;
    os << array;
    EXPECT_EQ(expected, os.str());
}

template <typename T>
void CheckArrayRepr(const std::string& expected, const std::vector<T>& data_vec, Shape shape,
                    const std::vector<GraphId>& graph_ids = std::vector<GraphId>()) {
    {
        DeviceScope ctx{"cpu"};
        CheckArrayReprWithCurrentDevice(expected, data_vec, shape, graph_ids);
    }

    {
#ifdef XCHAINER_ENABLE_CUDA
        DeviceScope ctx{"cuda"};
        CheckArrayReprWithCurrentDevice(expected, data_vec, shape, graph_ids);
#endif  // XCHAINER_ENABLE_CUDA
    }
}

TEST(ArrayReprTest, ArrayRepr) {
    // bool
    CheckArrayRepr<bool>("array([False], dtype=bool)", {false}, Shape({1}));
    CheckArrayRepr<bool>("array([ True], dtype=bool)", {true}, Shape({1}));
    CheckArrayRepr<bool>(
        "array([[False,  True,  True],\n"
        "       [ True, False,  True]], dtype=bool)",
        {false, true, true, true, false, true}, Shape({2, 3}));
    CheckArrayRepr<bool>("array([[[[ True]]]], dtype=bool)", {true}, Shape({1, 1, 1, 1}));

    // int8
    CheckArrayRepr<int8_t>("array([0], dtype=int8)", {0}, Shape({1}));
    CheckArrayRepr<int8_t>("array([-2], dtype=int8)", {-2}, Shape({1}));
    CheckArrayRepr<int8_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=int8)",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<int8_t>(
        "array([[ 0,  1,  2],\n"
        "       [-3,  4,  5]], dtype=int8)",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<int8_t>("array([[[[3]]]], dtype=int8)", {3}, Shape({1, 1, 1, 1}));

    // int16
    CheckArrayRepr<int16_t>("array([0], dtype=int16)", {0}, Shape({1}));
    CheckArrayRepr<int16_t>("array([-2], dtype=int16)", {-2}, Shape({1}));
    CheckArrayRepr<int16_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=int16)",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<int16_t>(
        "array([[ 0,  1,  2],\n"
        "       [-3,  4,  5]], dtype=int16)",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<int16_t>("array([[[[3]]]], dtype=int16)", {3}, Shape({1, 1, 1, 1}));

    // int32
    CheckArrayRepr<int32_t>("array([0], dtype=int32)", {0}, Shape({1}));
    CheckArrayRepr<int32_t>("array([-2], dtype=int32)", {-2}, Shape({1}));
    CheckArrayRepr<int32_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=int32)",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<int32_t>(
        "array([[ 0,  1,  2],\n"
        "       [-3,  4,  5]], dtype=int32)",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<int32_t>("array([[[[3]]]], dtype=int32)", {3}, Shape({1, 1, 1, 1}));

    // int64
    CheckArrayRepr<int64_t>("array([0], dtype=int64)", {0}, Shape({1}));
    CheckArrayRepr<int64_t>("array([-2], dtype=int64)", {-2}, Shape({1}));
    CheckArrayRepr<int64_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=int64)",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<int64_t>(
        "array([[ 0,  1,  2],\n"
        "       [-3,  4,  5]], dtype=int64)",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<int64_t>("array([[[[3]]]], dtype=int64)", {3}, Shape({1, 1, 1, 1}));

    // uint8
    CheckArrayRepr<uint8_t>("array([0], dtype=uint8)", {0}, Shape({1}));
    CheckArrayRepr<uint8_t>("array([2], dtype=uint8)", {2}, Shape({1}));
    CheckArrayRepr<uint8_t>(
        "array([[0, 1, 2],\n"
        "       [3, 4, 5]], dtype=uint8)",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<uint8_t>("array([[[[3]]]], dtype=uint8)", {3}, Shape({1, 1, 1, 1}));

    // float32
    CheckArrayRepr<float>("array([0.], dtype=float32)", {0}, Shape({1}));
    CheckArrayRepr<float>("array([3.25], dtype=float32)", {3.25}, Shape({1}));
    CheckArrayRepr<float>("array([-3.25], dtype=float32)", {-3.25}, Shape({1}));
    CheckArrayRepr<float>("array([ inf], dtype=float32)", {std::numeric_limits<float>::infinity()}, Shape({1}));
    CheckArrayRepr<float>("array([ -inf], dtype=float32)", {-std::numeric_limits<float>::infinity()}, Shape({1}));
    CheckArrayRepr<float>("array([ nan], dtype=float32)", {std::nanf("")}, Shape({1}));
    CheckArrayRepr<float>(
        "array([[0., 1., 2.],\n"
        "       [3., 4., 5.]], dtype=float32)",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<float>(
        "array([[0.  , 1.  , 2.  ],\n"
        "       [3.25, 4.  , 5.  ]], dtype=float32)",
        {0, 1, 2, 3.25, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<float>(
        "array([[ 0.  ,  1.  ,  2.  ],\n"
        "       [-3.25,  4.  ,  5.  ]], dtype=float32)",
        {0, 1, 2, -3.25, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<float>("array([[[[3.25]]]], dtype=float32)", {3.25}, Shape({1, 1, 1, 1}));

    // float64
    CheckArrayRepr<double>("array([0.], dtype=float64)", {0}, Shape({1}));
    CheckArrayRepr<double>("array([3.25], dtype=float64)", {3.25}, Shape({1}));
    CheckArrayRepr<double>("array([-3.25], dtype=float64)", {-3.25}, Shape({1}));
    CheckArrayRepr<double>("array([ inf], dtype=float64)", {std::numeric_limits<double>::infinity()}, Shape({1}));
    CheckArrayRepr<double>("array([ -inf], dtype=float64)", {-std::numeric_limits<double>::infinity()}, Shape({1}));
    CheckArrayRepr<double>("array([ nan], dtype=float64)", {std::nan("")}, Shape({1}));
    CheckArrayRepr<double>(
        "array([[0., 1., 2.],\n"
        "       [3., 4., 5.]], dtype=float64)",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<double>(
        "array([[0.  , 1.  , 2.  ],\n"
        "       [3.25, 4.  , 5.  ]], dtype=float64)",
        {0, 1, 2, 3.25, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<double>(
        "array([[ 0.  ,  1.  ,  2.  ],\n"
        "       [-3.25,  4.  ,  5.  ]], dtype=float64)",
        {0, 1, 2, -3.25, 4, 5}, Shape({2, 3}));
    CheckArrayRepr<double>("array([[[[3.25]]]], dtype=float64)", {3.25}, Shape({1, 1, 1, 1}));
}

TEST(ArrayReprTest, ArrayReprWithGraphIds) {
    // No graph
    CheckArrayRepr<int32_t>("array([3], dtype=int32)", {3}, Shape({1}));

    // Single graph
    CheckArrayRepr<int32_t>("array([-2], dtype=int32, graph_ids=['graph_1'])", {-2}, Shape({1}), {"graph_1"});

    // Single graph, empty string
    CheckArrayRepr<int32_t>("array([-2], dtype=int32, graph_ids=[''])", {-2}, Shape({1}), {""});

    // Two graphs
    CheckArrayRepr<int32_t>("array([1], dtype=int32, graph_ids=['graph_1', 'graph_2'])", {1}, Shape({1}), {"graph_1", "graph_2"});

    // Multiple graphs
    CheckArrayRepr<int32_t>("array([-9], dtype=int32, graph_ids=['graph_1', 'graph_2', 'graph_3', 'graph_4', 'graph_5'])", {-9}, Shape({1}),
                            {"graph_1", "graph_2", "graph_3", "graph_4", "graph_5"});
}

}  // namespace
}  // namespace xchainer
