#include "xchainer/array_body.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <gtest/gtest.h>

namespace xchainer {
namespace {

class ArrayBodyTest : public ::testing::Test {
 public:
  void SetUp() override {
    //device_ = {DeviceType{"foo"}, 2};
    dtype_ = Dtype::kFloat32;
  }

  ArrayBody MakeArrayBody(std::initializer_list<int64_t> shape) const {
     //return {device_, gsl::make_span(shape.begin(), shape.end()), dtype_};
     return {gsl::make_span(shape.begin(), shape.end()), dtype_};
  }

  //Device device() const {
  //  return device_;
  //}

  Dtype dtype() const {
    return dtype_;
  }

  //static std::shared_ptr<void> MakeAndSetData(ArrayBody& body, size_t n_elems, std::initializer_list<int64_t> strides) {
  //  std::shared_ptr<void> data = std::unique_ptr<float[]>(new float[n_elems]);
  //  body.SetData(data, gsl::make_span(strides.begin(), strides.end()));
  //  return data;
  //}

 private:
  //Device device_;
  Dtype dtype_;
};

TEST_F(ArrayBodyTest, Ctor) {
  ArrayBody x = MakeArrayBody({2, 3, 4});
  //EXPECT_EQ(device(), x.device());
  EXPECT_EQ(dtype(), x.dtype());
  EXPECT_EQ(3, x.ndim());
  //EXPECT_EQ(Shape({2, 3, 4}).cspan(), x.shape().cspan());
  EXPECT_EQ(2 * 3 * 4, x.total_size());
  EXPECT_EQ(4, x.element_bytes());
  EXPECT_EQ(2 * 3 * 4 * 4, x.total_bytes());
  EXPECT_EQ(nullptr, x.data());
  EXPECT_EQ(nullptr, x.raw_data());
}

//TEST_F(ArrayBodyTest, SetData) {
//  ArrayBody x = MakeArrayBody({2, 3, 4});
//  auto data = MakeAndSetData(x, 2 * 3 * 4, {48, 16, 4});
//
//  EXPECT_EQ(data, x.data());
//  EXPECT_EQ(data.get(), x.raw_data());
//  const std::array<int64_t, 3> strides = {48, 16, 4};
//  EXPECT_EQ(gsl::make_span(strides), x.strides().cspan());
//  EXPECT_TRUE(x.is_contiguous());
//  EXPECT_EQ(0, x.offset());
//}

//TEST_F(ArrayBodyTest, SetDataNonContiguous) {
//  ArrayBody x = MakeArrayBody({2, 3, 4});
//  MakeAndSetData(x, 3 * 2 * 4, {16, 32, 4});
//  EXPECT_FALSE(x.is_contiguous());
//}

TEST_F(ArrayBodyTest, SetContiguousData) {
  ArrayBody x = MakeArrayBody({2, 3, 4});
  std::shared_ptr<void> data = std::unique_ptr<float[]>(new float[2 * 3 * 4]);
  x.SetContiguousData(data);

  //const std::array<int64_t, 3> contiguous_strides = {48, 16, 4};

  EXPECT_EQ(data, x.data());
  EXPECT_EQ(data.get(), x.raw_data());
  //EXPECT_EQ(gsl::make_span(contiguous_strides), x.strides().cspan());
  EXPECT_TRUE(x.is_contiguous());
  EXPECT_EQ(0, x.offset());
}

}  // namespace
}  // namespace xchainer
