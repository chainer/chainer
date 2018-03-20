#include "xchainer/python/strides.h"

#include <algorithm>
#include <vector>

namespace xchainer {
namespace internal {

namespace py = pybind11;

Strides ToStrides(const py::tuple& tup) {
    std::vector<int64_t> v;
    v.reserve(tup.size());
    std::transform(tup.begin(), tup.end(), std::back_inserter(v), [](auto& item) { return py::cast<int64_t>(item); });
    return Strides{v};
}

py::tuple ToTuple(const Strides& strides) {
    py::tuple ret{strides.size()};
    for (size_t i = 0; i < strides.size(); ++i) {
        ret[i] = strides[i];
    }
    return ret;
}

}  // namespace internal
}  // namespace xchainer
