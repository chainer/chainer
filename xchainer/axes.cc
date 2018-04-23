#include "xchainer/axes.h"

#include <ostream>
#include <sstream>
#include <string>

#include "xchainer/axis.h"

namespace xchainer {

std::string Axes::ToString() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const Axes& axes) {
    os << "(";
    for (auto iter = axes.begin(); iter != axes.end(); ++iter) {
        if (iter != axes.begin()) {
            os << ", ";
        }
        os << static_cast<int>(*iter);
    }
    // same as Python tuples with trailing comma in case of length 1
    return os << (axes.ndim() == 1 ? ",)" : ")");
}

}  // namespace xchainer
