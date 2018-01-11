#include "xchainer/array_math.h"

#include <cstring>

#include "xchainer/array.h"

namespace xchainer {

void Copy(const Array& src, Array& out) { std::memcpy(out.data().get(), src.data().get(), src.total_bytes()); }

void Add(const Array& lhs, const Array& rhs, Array& out) {
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        int64_t total_size = lhs.total_size();
        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* rdata = static_cast<const T*>(rhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());

        for (int64_t i = 0; i < total_size; i++) {
            odata[i] = ldata[i] + rdata[i];
        }
    });
}

void Mul(const Array& lhs, const Array& rhs, Array& out) {
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        int64_t total_size = lhs.total_size();
        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* rdata = static_cast<const T*>(rhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());

        for (int64_t i = 0; i < total_size; i++) {
            odata[i] = ldata[i] * rdata[i];
        }
    });
}

}  // namespace xchainer
