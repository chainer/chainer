#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"

#include "chainerx/routines/creation.h"

namespace chainerx {
namespace testing {
namespace array_detail {

class ArrayBuilder {
public:
    explicit ArrayBuilder(Shape shape) : shape_{std::move(shape)}, device_{&GetDefaultDevice()} {}

    operator Array() const { return Build(); }  // NOLINT

    Array operator*() const { return Build(); }

    Array Build() const {
        CHAINERX_ASSERT(create_array_ != nullptr);
        return create_array_(*this);
    }

    template <typename T>
    ArrayBuilder& WithData(const std::vector<T>& data) {
        CHAINERX_ASSERT(static_cast<size_t>(shape_.GetTotalSize()) == data.size());
        return WithData<T>(data.begin(), data.end());
    }

    template <typename T, typename InputIter>
    ArrayBuilder& WithData(InputIter first, InputIter last) {
        CHAINERX_ASSERT(create_array_ == nullptr);
        std::vector<T> data(first, last);

        CHAINERX_ASSERT(data.size() == static_cast<size_t>(shape_.GetTotalSize()));

        // Define create_array_ here to type-erase T of `data`.
        // Note: ArrayBuilder must be specified as an argument instead of capturing `this` pointer, because the ArrayBuilder instance could
        // be copied and thus `this` pointer could be invalidated at the moment the function is called.
        create_array_ = [data](const ArrayBuilder& builder) -> Array {
            Dtype dtype = TypeToDtype<T>;
            const Shape& shape = builder.shape_;
            CHAINERX_ASSERT(static_cast<size_t>(shape.GetTotalSize()) == data.size());
            Strides strides = builder.GetStrides<T>();
            int64_t total_size = shape.GetTotalSize();
            size_t n_bytes = internal::GetRequiredBytes(shape, strides, sizeof(T));
            std::shared_ptr<uint8_t> ptr{new uint8_t[n_bytes], std::default_delete<uint8_t[]>()};
            std::fill(ptr.get(), ptr.get() + n_bytes, uint8_t{0xff});

            if (total_size > 0) {
                // Copy the data to buffer, respecting strides
                auto* raw_ptr = ptr.get();
                Shape counter = shape;
                for (const T& value : data) {
                    // Copy a single value
                    CHAINERX_ASSERT((raw_ptr - ptr.get()) < static_cast<ptrdiff_t>(n_bytes));
                    *reinterpret_cast<T*>(raw_ptr) = value;  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                    // Advance the counter and the pointer
                    int8_t i_dim = shape.ndim() - 1;
                    while (i_dim >= 0) {
                        raw_ptr += strides[i_dim];
                        counter[i_dim]--;
                        if (counter[i_dim] > 0) {
                            break;
                        }
                        counter[i_dim] = shape[i_dim];
                        raw_ptr -= strides[i_dim] * shape[i_dim];
                        i_dim--;
                    }
                }
            }
            return internal::FromHostData(shape, dtype, std::move(ptr), strides, 0, *builder.device_);
        };
        return *this;
    }

    template <typename T>
    ArrayBuilder& WithLinearData(T start = T{0}, T step = T{1}) {
        int64_t total_size = shape_.GetTotalSize();
        std::vector<T> data;
        data.reserve(total_size);
        T value = start;
        for (int64_t i = 0; i < total_size; ++i) {
            data.push_back(value);
            value += step;
        }
        return WithData<T>(data.begin(), data.end());
    }

    ArrayBuilder& WithPadding(const std::vector<int64_t>& padding) {
        CHAINERX_ASSERT(padding_.empty());
        CHAINERX_ASSERT(padding.size() == shape_.size());
        padding_ = padding;
        return *this;
    }

    ArrayBuilder& WithPadding(int64_t padding) {
        CHAINERX_ASSERT(padding_.empty());
        std::fill_n(std::back_inserter(padding_), shape_.size(), padding);
        return *this;
    }

    ArrayBuilder& WithDevice(Device& device) {
        device_ = &device;
        return *this;
    }

private:
    template <typename T>
    Strides GetStrides() const {
        std::vector<int64_t> padding = padding_;
        if (padding.empty()) {
            std::fill_n(std::back_inserter(padding), shape_.size(), int64_t{0});
        }

        // Create strides with extra space specified by `padding`.
        CHAINERX_ASSERT(padding.size() == shape_.size());

        Strides strides{};
        strides.resize(shape_.ndim());
        int64_t st = sizeof(T);
        for (int8_t i = shape_.ndim() - 1; i >= 0; --i) {
            st += sizeof(T) * padding[i];  // paddings are multiples of the item size.
            strides[i] = st;
            st *= shape_[i];
        }
        return strides;
    }

    Shape shape_;

    Device* device_{};

    // Padding items (multiplied by sizeof(T) during construction) to each dimension.
    // TODO(niboshi): Support negative strides
    std::vector<int64_t> padding_;

    // Using std::function to type-erase data type T
    std::function<Array(const ArrayBuilder&)> create_array_;
};

}  // namespace array_detail

inline array_detail::ArrayBuilder BuildArray(const Shape& shape) { return array_detail::ArrayBuilder{shape}; }

}  // namespace testing
}  // namespace chainerx
