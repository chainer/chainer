#include "chainerx/dynamic_lib.h"

#include <string>

#include "chainerx/platform.h"

namespace chainerx {

void* DlOpen(const std::string& filename, int flags) { return platform::DlOpen(filename, flags); }

void DlClose(void* handle) { platform::DlClose(handle); }

}  // namespace chainerx
