cmake_minimum_required(VERSION 3.14)
project("ccwm_cpp" C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE sources src/*.cpp src/*.h)

add_subdirectory(ggml)

add_subdirectory(pybind11)
pybind11_add_module(ccwm_cpp ${sources})

target_include_directories(ccwm_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include)
target_link_libraries(ccwm_cpp PRIVATE ggml)