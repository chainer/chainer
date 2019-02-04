#pragma once

#include <string>

namespace chainerx {
namespace platform {

void SetEnv(const std::string& name, const std::string& value);

void UnsetEnv(const std::string& name);

void* DlOpen(const std::string& filename, int flags);

void DlClose(void* handle);

}  // namespace platform
}  // namespace chainerx
