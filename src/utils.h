#pragma once
#include <memory>
#include <string>
#include <stdexcept>
#include <sstream>

// https://stackoverflow.com/a/26221725
template<typename... Args>
std::string strfmt(const std::string& format, Args&&... args) {
    int size_s = std::snprintf(nullptr, 0, format.c_str(), std::forward<Args>(args)...) + 1;
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), std::forward<Args>(args)...);
    return std::string(buf.get(), buf.get() + size - 1);
}

template<typename T>
std::string sizefmt(T bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    int unitIndex = 0;
    while (bytes >= 1024 && unitIndex < 5) {
        bytes /= 1024;
        unitIndex++;
    }
    return std::to_string(bytes) + " " + units[unitIndex];
}
