#include "chainerx/array_repr.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/array_node.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/constant.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/native/data_type.h"
#include "chainerx/numeric.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/shape.h"

namespace chainerx {

namespace {

int GetNDigits(int64_t value) {
    int digits = 0;
    while (value != 0) {
        value /= 10;
        ++digits;
    }
    return digits;
}

void PrintNTimes(std::ostream& os, char c, int n) {
    while (n-- > 0) {
        os << c;
    }
}

class IntFormatter {
public:
    void Scan(int64_t value) {
        int digits = 0;
        if (value < 0) {
            ++digits;
            value = -value;
        }
        digits += GetNDigits(value);
        if (max_digits_ < digits) {
            max_digits_ = digits;
        }
    }

    void Print(std::ostream& os, int64_t value) const { os << std::setw(max_digits_) << std::right << value; }

private:
    int max_digits_ = 1;
};

class FloatFormatter {
public:
    void Scan(Float16 value) { Scan(static_cast<double>(value)); }

    void Scan(double value) {
        int b_digits = 0;
        if (value < 0) {
            has_minus_ = true;
            ++b_digits;
            value = -value;
        }
        if (IsInf(value) || IsNan(value)) {
            b_digits += 3;
            if (digits_before_point_ < b_digits) {
                digits_before_point_ = b_digits;
            }
            return;
        }
        if (value >= 100'000'000) {
            int e_digits = GetNDigits(static_cast<int64_t>(std::log10(value)));
            if (digits_after_e_ < e_digits) {
                digits_after_e_ = e_digits;
            }
        }
        if (digits_after_e_ > 0) {
            return;
        }

        const auto int_frac_parts = IntFracPartsToPrint(value);

        b_digits += GetNDigits(int_frac_parts.first);
        if (digits_before_point_ < b_digits) {
            digits_before_point_ = b_digits;
        }

        const int a_digits = GetNDigits(int_frac_parts.second) - 1;
        if (digits_after_point_ < a_digits) {
            digits_after_point_ = a_digits;
        }
    }

    void Print(std::ostream& os, Float16 value) { Print(os, static_cast<double>(value)); }

    void Print(std::ostream& os, double value) {
        if (digits_after_e_ > 0) {
            int width = 12 + (has_minus_ ? 1 : 0) + digits_after_e_;
            if (has_minus_ && !std::signbit(value)) {
                os << ' ';
                --width;
            }
            os << std::scientific << std::left << std::setw(width) << std::setprecision(8) << value;
        } else {
            if (IsInf(value) || IsNan(value)) {
                os << std::right << std::setw(digits_before_point_ + digits_after_point_ + 1) << value;
                return;
            }
            const auto int_frac_parts = IntFracPartsToPrint(value);
            const int a_digits = GetNDigits(int_frac_parts.second) - 1;
            os << std::fixed << std::right << std::setw(digits_before_point_ + a_digits + 1) << std::setprecision(a_digits)
               << std::showpoint << value;
            PrintNTimes(os, ' ', digits_after_point_ - a_digits);
        }
    }

private:
    // Returns the integral part and fractional part as integers.
    // Note that the fractional part is prefixed by 1 so that the information of preceding zeros is not missed.
    static std::pair<int64_t, int64_t> IntFracPartsToPrint(double value) {
        double int_part;
        const double frac_part = std::modf(value, &int_part);

        auto shifted_frac_part = static_cast<int64_t>((std::abs(frac_part) + 1) * 100'000'000);
        while ((shifted_frac_part % 10) == 0) {
            shifted_frac_part /= 10;
        }

        return {static_cast<int64_t>(int_part), shifted_frac_part};
    }

    int digits_before_point_ = 1;
    int digits_after_point_ = 0;
    int digits_after_e_ = 0;
    bool has_minus_ = false;
};

class BoolFormatter {
public:
    void Scan(bool value) { (void)value; /* unused */ }

    void Print(std::ostream& os, bool value) const {
        os << (value ? " True" : "False");  // NOLINTER
    }
};

template <typename T>
using Formatter = std::
        conditional_t<std::is_same<T, bool>::value, BoolFormatter, std::conditional_t<IsFloatingPointV<T>, FloatFormatter, IntFormatter>>;

template <int8_t Ndim>
struct ArrayReprImpl {
    template <typename T>
    void operator()(const Array& array, std::ostream& os) const {
        Array native_array = array.AsGradStopped().ToNative();
        Formatter<T> formatter;

        // array should be already synchronized
        // Let formatter scan all elements to print.
        Indexer<Ndim> indexer{array.shape()};
        IndexableArray<const T, Ndim> iarray{native_array};
        for (auto it = indexer.It(0); it; ++it) {
            T value = native::StorageToDataType<const T>(iarray[it]);
            formatter.Scan(value);
        }

        os << "array(";
        if (array.GetTotalSize() == 0) {
            os << "[]";
        } else {
            NoBackpropModeScope scope{};
            bool should_abbreviate = array.GetTotalSize() > kThreshold;
            ArrayReprRecursive<T>(native_array, formatter, 7, os, should_abbreviate);
        }

        // Print the footer
        os << ", shape=" << array.shape();
        os << ", dtype=" << array.dtype();
        os << ", device='" << array.device().name() << "'";
        const std::vector<std::shared_ptr<internal::ArrayNode>>& array_nodes = internal::GetArrayBody(array)->nodes();
        if (!array_nodes.empty()) {
            os << ", backprop_ids=[";
            for (size_t i = 0; i < array_nodes.size(); ++i) {
                if (i > 0) {
                    os << ", ";
                }
                os << '\'' << array_nodes[i]->backprop_id() << '\'';
            }
            os << ']';
        }
        os << ')';
    }

private:
    static constexpr int kMaxItemNumPerLine = 10;
    static constexpr int64_t kThreshold = 1000;
    static constexpr int64_t kEdgeItems = 3;

    // The behavior of this function is recursively defined as follows:
    // array.ndim() == 0 => Returns a string represenation of the single scalar.
    // array.ndim() == 1 => Returns a string: space separated scalars.
    // array.ndim() >= 2 => Returns a string: newline separated arrays with one less dimension.
    template <typename T>
    void ArrayReprRecursive(const Array& array, Formatter<T>& formatter, size_t indent, std::ostream& os, bool abbreviate = false) const {
        const uint8_t ndim = array.ndim();
        if (ndim == 0) {
            formatter.Print(os, static_cast<T>(AsScalar(array)));
            return;
        }
        auto print_indent = [ndim, indent, &os](int64_t i) {
            if (i != 0) {
                os << ",";
                if (ndim > 1 || i % kMaxItemNumPerLine == 0) {
                    PrintNTimes(os, '\n', ndim - 1);
                    PrintNTimes(os, ' ', indent);
                } else {
                    os << ' ';
                }
            }
        };
        os << "[";
        int64_t size = array.shape().front();
        if (abbreviate && size > kEdgeItems * 2) {
            for (int64_t i = 0; i < kEdgeItems; ++i) {
                print_indent(i);
                ArrayReprRecursive<T>(internal::At(array, {ArrayIndex{i}}), formatter, indent + 1, os, abbreviate);
            }
            print_indent(1);
            os << "...";
            print_indent(1);
            for (int64_t i = 0; i < kEdgeItems; ++i) {
                print_indent(i);
                ArrayReprRecursive<T>(internal::At(array, {ArrayIndex{i - kEdgeItems}}), formatter, indent + 1, os, abbreviate);
            }
        } else {
            for (int64_t i = 0; i < size; ++i) {
                print_indent(i);
                ArrayReprRecursive<T>(internal::At(array, {ArrayIndex{i}}), formatter, indent + 1, os, abbreviate);
            }
        }
        os << "]";
    }
};

}  // namespace

std::ostream& operator<<(std::ostream& os, const Array& array) {
    // TODO(hvy): We need to determine the output specification of this function, whether or not to align with Python repr specification,
    // and also whether this functionality should be defined in C++ layer or Python layer.
    // TODO(hvy): Consider using a static dimensionality.
    VisitDtype(array.dtype(), [&os, &array](auto pt) { ArrayReprImpl<kDynamicNdim>{}.operator()<typename decltype(pt)::type>(array, os); });
    return os;
}

std::string ArrayRepr(const Array& array) {
    std::ostringstream os;
    os << array;
    return os.str();
}

}  // namespace chainerx
