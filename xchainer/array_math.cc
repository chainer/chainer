#include "xchainer/array_math.h"

#include <cassert>

#include "xchainer/array.h"

namespace xchainer {

void Add(const Array& lhs, const Array& rhs, Array& out) {
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        auto total_size = lhs.shape().total_size();
        const T* ldata = static_cast<const T*>(lhs.data().get());
        const T* rdata = static_cast<const T*>(rhs.data().get());
        T* odata = static_cast<T*>(out.data().get());

        for (decltype(total_size) i = 0; i < total_size; i++) {
            odata[i] = ldata[i] + rdata[i];
        }
    });
}

void Mul(const Array& lhs, const Array& rhs, Array& out) {
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        auto total_size = lhs.shape().total_size();
        const T* ldata = static_cast<const T*>(lhs.data().get());
        const T* rdata = static_cast<const T*>(rhs.data().get());
        T* odata = static_cast<T*>(out.data().get());

        for (decltype(total_size) i = 0; i < total_size; i++) {
            odata[i] = ldata[i] * rdata[i];
        }
    });
}

}  // namespace xchainer
