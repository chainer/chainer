#include "xchainer/python/strides.h"

#include <algorithm>
#include <vector>

namespace xchainer {

namespace py = pybind11;

Strides ToStrides(const py::tuple& tup) {
    std::vector<int64_t> v;
    v.reserve(tup.size());
    std::transform(tup.begin(), tup.end(), std::back_inserter(v), [](auto& item) { return py::cast<int64_t>(item); });
    return Strides{v};
}

py::tuple ToTuple(const Strides& strides) {
    // TODO(beam2d): Consider casting directly to tuple instead of casting to list.
    py::list ret;
    for (int64_t stride : strides) {
        ret.append(stride);
    }
    return ret;
}

}  // namespace xchainer
