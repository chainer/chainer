#include "xchainer/array_repr.h"

#include <gmock/gmock.h>
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
std::string AsString(const T& object) {
    std::ostringstream os;
    os << object;
    return os.str();
}

std::string ArrayReprFromTemplate(const std::string& array_data_repr, const std::string& dtype_repr, const std::string& device_repr,
                                  const nonstd::optional<std::string>& graph_ids_repr = nonstd::nullopt) {
    std::ostringstream os;
    os << "array(" + array_data_repr;
    os << ", dtype=" + dtype_repr;
    os << ", device=" << device_repr;
    if (graph_ids_repr) {
        os << ", graph_ids=" << graph_ids_repr.value();
    }
    os << ")";
    return os.str();
}

std::string CreateExpectedArrayRepr(const std::string& array_data_repr, Dtype dtype, const Device& device,
                                    const std::vector<GraphId> graph_ids) {
    // Dtype repr
    std::string dtype_repr = AsString(dtype);

    // Device repr
    std::string device_repr = AsString(device);

    // Graph IDs repr
    nonstd::optional<std::string> graph_ids_repr;
    if (!graph_ids.empty()) {
        std::ostringstream os;
        os << "[";
        for (size_t i = 0; i < graph_ids.size(); ++i) {
            if (i > 0) {
                os << ", ";
            }
            os << '\'' << graph_ids[i] << '\'';
        }
        os << "]";
        graph_ids_repr = nonstd::optional<std::string>(os.str());
    }

    // Expected Array repr
    return ArrayReprFromTemplate(array_data_repr, dtype_repr, device_repr, graph_ids_repr);
}

// Check Array repr without any graph IDs
template <typename T>
void CheckArrayRepr(const std::string& array_data_repr, const std::vector<T>& data_vec, Shape shape, const Device& device,
                    const std::vector<GraphId> graph_ids = {}) {
    std::string expected = CreateExpectedArrayRepr(array_data_repr, TypeToDtype<T>, device, graph_ids);

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

void CheckAllForDevice(const Device& device) {
    // bool
    CheckArrayRepr<bool>("[False]", {false}, Shape({1}), device);
    CheckArrayRepr<bool>("[ True]", {true}, Shape({1}), device);
    CheckArrayRepr<bool>(
        "[[False,  True,  True],\n"
        "       [ True, False,  True]]",
        {false, true, true, true, false, true}, Shape({2, 3}), device);
    CheckArrayRepr<bool>("[False]", {false}, Shape({1}), device);
    CheckArrayRepr<bool>("[ True]", {true}, Shape({1}), device);
    CheckArrayRepr<bool>(
        "[[False,  True,  True],\n"
        "       [ True, False,  True]]",
        {false, true, true, true, false, true}, Shape({2, 3}), device);
    CheckArrayRepr<bool>("[[[[ True]]]]", {true}, Shape({1, 1, 1, 1}), device);

    // int8
    CheckArrayRepr<int8_t>("[0]", {0}, Shape({1}), device);
    CheckArrayRepr<int8_t>("[-2]", {-2}, Shape({1}), device);
    CheckArrayRepr<int8_t>(
        "[[0, 1, 2],\n"
        "       [3, 4, 5]]",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int8_t>(
        "[[ 0,  1,  2],\n"
        "       [-3,  4,  5]]",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int8_t>("[[[[3]]]]", {3}, Shape({1, 1, 1, 1}), device);

    // int16
    CheckArrayRepr<int16_t>("[0]", {0}, Shape({1}), device);
    CheckArrayRepr<int16_t>("[-2]", {-2}, Shape({1}), device);
    CheckArrayRepr<int16_t>(
        "[[0, 1, 2],\n"
        "       [3, 4, 5]]",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int16_t>(
        "[[ 0,  1,  2],\n"
        "       [-3,  4,  5]]",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int16_t>("[[[[3]]]]", {3}, Shape({1, 1, 1, 1}), device);

    // int32
    CheckArrayRepr<int32_t>("[0]", {0}, Shape({1}), device);
    CheckArrayRepr<int32_t>("[-2]", {-2}, Shape({1}), device);
    CheckArrayRepr<int32_t>(
        "[[0, 1, 2],\n"
        "       [3, 4, 5]]",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int32_t>(
        "[[ 0,  1,  2],\n"
        "       [-3,  4,  5]]",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int32_t>("[[[[3]]]]", {3}, Shape({1, 1, 1, 1}), device);

    // int64
    CheckArrayRepr<int64_t>("[0]", {0}, Shape({1}), device);
    CheckArrayRepr<int64_t>("[-2]", {-2}, Shape({1}), device);
    CheckArrayRepr<int64_t>(
        "[[0, 1, 2],\n"
        "       [3, 4, 5]]",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int64_t>(
        "[[ 0,  1,  2],\n"
        "       [-3,  4,  5]]",
        {0, 1, 2, -3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<int64_t>("[[[[3]]]]", {3}, Shape({1, 1, 1, 1}), device);

    // uint8
    CheckArrayRepr<uint8_t>("[0]", {0}, Shape({1}), device);
    CheckArrayRepr<uint8_t>("[2]", {2}, Shape({1}), device);
    CheckArrayRepr<uint8_t>(
        "[[0, 1, 2],\n"
        "       [3, 4, 5]]",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<uint8_t>("[[[[3]]]]", {3}, Shape({1, 1, 1, 1}), device);

    // float32
    CheckArrayRepr<float>("[0.]", {0}, Shape({1}), device);
    CheckArrayRepr<float>("[3.25]", {3.25}, Shape({1}), device);
    CheckArrayRepr<float>("[-3.25]", {-3.25}, Shape({1}), device);
    CheckArrayRepr<float>("[ inf]", {std::numeric_limits<float>::infinity()}, Shape({1}), device);
    CheckArrayRepr<float>("[ -inf]", {-std::numeric_limits<float>::infinity()}, Shape({1}), device);
    CheckArrayRepr<float>("[ nan]", {std::nanf("")}, Shape({1}), device);
    CheckArrayRepr<float>(
        "[[0., 1., 2.],\n"
        "       [3., 4., 5.]]",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<float>(
        "[[0.  , 1.  , 2.  ],\n"
        "       [3.25, 4.  , 5.  ]]",
        {0, 1, 2, 3.25, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<float>(
        "[[ 0.  ,  1.  ,  2.  ],\n"
        "       [-3.25,  4.  ,  5.  ]]",
        {0, 1, 2, -3.25, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<float>("[[[[3.25]]]]", {3.25}, Shape({1, 1, 1, 1}), device);

    // float64
    CheckArrayRepr<double>("[0.]", {0}, Shape({1}), device);
    CheckArrayRepr<double>("[3.25]", {3.25}, Shape({1}), device);
    CheckArrayRepr<double>("[-3.25]", {-3.25}, Shape({1}), device);
    CheckArrayRepr<double>("[ inf]", {std::numeric_limits<double>::infinity()}, Shape({1}), device);
    CheckArrayRepr<double>("[ -inf]", {-std::numeric_limits<double>::infinity()}, Shape({1}), device);
    CheckArrayRepr<double>("[ nan]", {std::nan("")}, Shape({1}), device);
    CheckArrayRepr<double>(
        "[[0., 1., 2.],\n"
        "       [3., 4., 5.]]",
        {0, 1, 2, 3, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<double>(
        "[[0.  , 1.  , 2.  ],\n"
        "       [3.25, 4.  , 5.  ]]",
        {0, 1, 2, 3.25, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<double>(
        "[[ 0.  ,  1.  ,  2.  ],\n"
        "       [-3.25,  4.  ,  5.  ]]",
        {0, 1, 2, -3.25, 4, 5}, Shape({2, 3}), device);
    CheckArrayRepr<double>("[[[[3.25]]]]", {3.25}, Shape({1, 1, 1, 1}), device);

    // Single graph
    CheckArrayRepr<int32_t>("[-2]", {-2}, Shape({1}), device, {"graph_1"});

    // Two graphs
    CheckArrayRepr<int32_t>("[1]", {1}, Shape({1}), device, {"graph_1", "graph_2"});

    // Multiple graphs
    CheckArrayRepr<int32_t>("[-9]", {-9}, Shape({1}), device, {"graph_1", "graph_2", "graph_3", "graph_4", "graph_5"});
}

TEST(ArrayReprNativeTest, ArrayRepr) {
    NativeBackend backend;
    Device device = Device("cpu", &backend);
    CheckAllForDevice(device);
}

#ifdef XCHAINER_ENABLE_CUDA
TEST(ArrayReprCudaTest, ArrayRepr) {
    cuda::CudaBackend backend;
    Device device = Device("cuda", &backend);
    CheckAllForDevice(device);
}
#endif  // XCHAINER_ENABLE_CUDA

}  // namespace
}  // namespace xchainer
