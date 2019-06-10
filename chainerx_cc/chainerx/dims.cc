#include "chainerx/dims.h"

#include <cstdint>
#include <ostream>

#include "chainerx/constant.h"
#include "chainerx/stack_vector.h"

namespace chainerx {

void DimsFormatter::Print(std::ostream& os) const {
    os << "[";
    for (auto iter = dims_.begin(); iter != dims_.end(); ++iter) {
        if (iter != dims_.begin()) {
            os << ", ";
        }
        os << *iter;
    }
    os << "]";
}

std::ostream& operator<<(std::ostream& os, const DimsFormatter& formatter) {
    formatter.Print(os);
    return os;
}

}  // namespace chainerx
