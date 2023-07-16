cmake_minimum_required(VERSION 3.16.3)
project(mope_vec LANGUAGES CXX VERSION 0.0.2)

# copied from https://www.scivision.dev/cmake-auto-gitignore-build-dir/
if(NOT PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    # Git auto-ignore out-of-source build directory
    file(GENERATE OUTPUT .gitignore CONTENT "*")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Catch2 3 REQUIRED)

include(CTest)
include(Catch)
enable_testing()

add_subdirectory(tests)
