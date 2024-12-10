cmake_minimum_required(VERSION 3.14)
project("ccwm_cpp" C CXX)

set(INSTALL_DIR "" CACHE STRING "Install directory (e.g. site-packages)")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH "${INSTALL_DIR}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE sources src/*.cpp src/*.h)

add_subdirectory(ggml)

add_subdirectory(pybind11)
pybind11_add_module(ccwm_cpp_ ${sources})

target_include_directories(ccwm_cpp_ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include ${CMAKE_CURRENT_SOURCE_DIR}/json)
target_link_libraries(ccwm_cpp_ PRIVATE ggml)