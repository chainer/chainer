#pragma once

#include <string>

#include <nonstd/optional.hpp>

namespace chainerx {
namespace crossplatform {

nonstd::optional<std::string> GetEnv(const std::string& name);

void SetEnv(const std::string& name, const std::string& value);

void UnsetEnv(const std::string& name);

// TODO(hvy): flags argument might need to be wrapped as well for various platforms.
void* DlOpen(const std::string& filename, int flags);

void DlCloseNoExcept(void* handle) noexcept;

}  // namespace crossplatform
}  // namespace chainerx
