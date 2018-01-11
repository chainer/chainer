#include "xchainer/array_repr.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "xchainer/device.h"

namespace xchainer {
namespace {

template <typename T>
void CheckArrayReprWithCurrentDevice(const std::vector<T>& data_vec, Shape shape, const std::string& expected) {
    // Copy to a contiguous memory block because std::vector<bool> is not packed as a sequence of bool's.
    std::shared_ptr<T> data_ptr = std::make_unique<T[]>(data_vec.size());
    std::copy(data_vec.begin(), data_vec.end(), data_ptr.get());
    Array array = Array::FromBuffer(shape, TypeToDtype<T>, static_cast<std::shared_ptr<void>>(data_ptr));

    // std::string version
    EXPECT_EQ(ArrayRepr(array), expected);

    // std::ostream version
    std::ostringstream os;
    os << array;
    EXPECT_EQ(os.str(), expected);
}

template <typename T>
void CheckArrayRepr(const std::vector<T>& data_vec, Shape shape, const std::string& expected) {
    {
        DeviceScope ctx{"cpu"};
        CheckArrayReprWithCurrentDevice(data_vec, shape, expected);
    }

    {
#ifdef XCHAINER_ENABLE_CUDA
        DeviceScope ctx{"cuda"};
        CheckArrayReprWithCurrentDevice(data_vec, shape, expected);
#endif  // XCHAINER_ENABLE_CUDA
    }
}

TEST(ArrayReprTest, ArrayRepr) {
    // bool
    CheckArrayRepr<bool>({false}, Shape({1}), "array([False], dtype=bool)");
    CheckArrayRepr<bool>({true}, Shape({1}), "array([ True], dtype=bool)");
    CheckArrayRepr<bool>({false, true, true, true, false, true}, Shape({2, 3}),
                         "array([[False,  True,  True],\n"
                         "       [ True, False,  True]], dtype=bool)");
    CheckArrayRepr<bool>({true}, Shape({1, 1, 1, 1}), "array([[[[ True]]]], dtype=bool)");

    // int8
    CheckArrayRepr<int8_t>({0}, Shape({1}), "array([0], dtype=int8)");
    CheckArrayRepr<int8_t>({-2}, Shape({1}), "array([-2], dtype=int8)");
    CheckArrayRepr<int8_t>({0, 1, 2, 3, 4, 5}, Shape({2, 3}),
                           "array([[0, 1, 2],\n"
                           "       [3, 4, 5]], dtype=int8)");
    CheckArrayRepr<int8_t>({0, 1, 2, -3, 4, 5}, Shape({2, 3}),
                           "array([[ 0,  1,  2],\n"
                           "       [-3,  4,  5]], dtype=int8)");
    CheckArrayRepr<int8_t>({3}, Shape({1, 1, 1, 1}), "array([[[[3]]]], dtype=int8)");

    // int16
    CheckArrayRepr<int16_t>({0}, Shape({1}), "array([0], dtype=int16)");
    CheckArrayRepr<int16_t>({-2}, Shape({1}), "array([-2], dtype=int16)");
    CheckArrayRepr<int16_t>({0, 1, 2, 3, 4, 5}, Shape({2, 3}),
                            "array([[0, 1, 2],\n"
                            "       [3, 4, 5]], dtype=int16)");
    CheckArrayRepr<int16_t>({0, 1, 2, -3, 4, 5}, Shape({2, 3}),
                            "array([[ 0,  1,  2],\n"
                            "       [-3,  4,  5]], dtype=int16)");
    CheckArrayRepr<int16_t>({3}, Shape({1, 1, 1, 1}), "array([[[[3]]]], dtype=int16)");

    // int32
    CheckArrayRepr<int32_t>({0}, Shape({1}), "array([0], dtype=int32)");
    CheckArrayRepr<int32_t>({-2}, Shape({1}), "array([-2], dtype=int32)");
    CheckArrayRepr<int32_t>({0, 1, 2, 3, 4, 5}, Shape({2, 3}),
                            "array([[0, 1, 2],\n"
                            "       [3, 4, 5]], dtype=int32)");
    CheckArrayRepr<int32_t>({0, 1, 2, -3, 4, 5}, Shape({2, 3}),
                            "array([[ 0,  1,  2],\n"
                            "       [-3,  4,  5]], dtype=int32)");
    CheckArrayRepr<int32_t>({3}, Shape({1, 1, 1, 1}), "array([[[[3]]]], dtype=int32)");

    // int64
    CheckArrayRepr<int64_t>({0}, Shape({1}), "array([0], dtype=int64)");
    CheckArrayRepr<int64_t>({-2}, Shape({1}), "array([-2], dtype=int64)");
    CheckArrayRepr<int64_t>({0, 1, 2, 3, 4, 5}, Shape({2, 3}),
                            "array([[0, 1, 2],\n"
                            "       [3, 4, 5]], dtype=int64)");
    CheckArrayRepr<int64_t>({0, 1, 2, -3, 4, 5}, Shape({2, 3}),
                            "array([[ 0,  1,  2],\n"
                            "       [-3,  4,  5]], dtype=int64)");
    CheckArrayRepr<int64_t>({3}, Shape({1, 1, 1, 1}), "array([[[[3]]]], dtype=int64)");

    // uint8
    CheckArrayRepr<uint8_t>({0}, Shape({1}), "array([0], dtype=uint8)");
    CheckArrayRepr<uint8_t>({2}, Shape({1}), "array([2], dtype=uint8)");
    CheckArrayRepr<uint8_t>({0, 1, 2, 3, 4, 5}, Shape({2, 3}),
                            "array([[0, 1, 2],\n"
                            "       [3, 4, 5]], dtype=uint8)");
    CheckArrayRepr<uint8_t>({3}, Shape({1, 1, 1, 1}), "array([[[[3]]]], dtype=uint8)");

    // float32
    CheckArrayRepr<float>({0}, Shape({1}), "array([0.], dtype=float32)");
    CheckArrayRepr<float>({3.25}, Shape({1}), "array([3.25], dtype=float32)");
    CheckArrayRepr<float>({-3.25}, Shape({1}), "array([-3.25], dtype=float32)");
    CheckArrayRepr<float>({std::numeric_limits<float>::infinity()}, Shape({1}), "array([ inf], dtype=float32)");
    CheckArrayRepr<float>({-std::numeric_limits<float>::infinity()}, Shape({1}), "array([ -inf], dtype=float32)");
    CheckArrayRepr<float>({std::nanf("")}, Shape({1}), "array([ nan], dtype=float32)");
    CheckArrayRepr<float>({0, 1, 2, 3, 4, 5}, Shape({2, 3}),
                          "array([[0., 1., 2.],\n"
                          "       [3., 4., 5.]], dtype=float32)");
    CheckArrayRepr<float>({0, 1, 2, 3.25, 4, 5}, Shape({2, 3}),
                          "array([[0.  , 1.  , 2.  ],\n"
                          "       [3.25, 4.  , 5.  ]], dtype=float32)");
    CheckArrayRepr<float>({0, 1, 2, -3.25, 4, 5}, Shape({2, 3}),
                          "array([[ 0.  ,  1.  ,  2.  ],\n"
                          "       [-3.25,  4.  ,  5.  ]], dtype=float32)");
    CheckArrayRepr<float>({3.25}, Shape({1, 1, 1, 1}), "array([[[[3.25]]]], dtype=float32)");

    // float64
    CheckArrayRepr<double>({0}, Shape({1}), "array([0.], dtype=float64)");
    CheckArrayRepr<double>({3.25}, Shape({1}), "array([3.25], dtype=float64)");
    CheckArrayRepr<double>({-3.25}, Shape({1}), "array([-3.25], dtype=float64)");
    CheckArrayRepr<double>({std::numeric_limits<double>::infinity()}, Shape({1}), "array([ inf], dtype=float64)");
    CheckArrayRepr<double>({-std::numeric_limits<double>::infinity()}, Shape({1}), "array([ -inf], dtype=float64)");
    CheckArrayRepr<double>({std::nan("")}, Shape({1}), "array([ nan], dtype=float64)");
    CheckArrayRepr<double>({0, 1, 2, 3, 4, 5}, Shape({2, 3}),
                           "array([[0., 1., 2.],\n"
                           "       [3., 4., 5.]], dtype=float64)");
    CheckArrayRepr<double>({0, 1, 2, 3.25, 4, 5}, Shape({2, 3}),
                           "array([[0.  , 1.  , 2.  ],\n"
                           "       [3.25, 4.  , 5.  ]], dtype=float64)");
    CheckArrayRepr<double>({0, 1, 2, -3.25, 4, 5}, Shape({2, 3}),
                           "array([[ 0.  ,  1.  ,  2.  ],\n"
                           "       [-3.25,  4.  ,  5.  ]], dtype=float64)");
    CheckArrayRepr<double>({3.25}, Shape({1, 1, 1, 1}), "array([[[[3.25]]]], dtype=float64)");
}

}  // namespace
}  // namespace xchainer
