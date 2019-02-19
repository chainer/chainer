#pragma once

#include <string>

namespace chainerx {
namespace platform {
namespace windows {

void SetEnv(const std::string& name, const std::string& value);

void UnsetEnv(const std::string& name);

void* DlOpen(const std::string& filename);

void DlClose(void* handle);

void* DlSym(void* handle, const std::string& name);

}  // namespace windows
}  // namespace platform
}  // namespace chainerx
