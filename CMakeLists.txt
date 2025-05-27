cmake_minimum_required(VERSION 3.10)
project(raif LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Source files
file(GLOB RAIF_SRCS
    src/runtime/*.cpp
    src/ops/*.cpp
)

add_library(raif STATIC ${RAIF_SRCS})

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(raif PRIVATE -mavx2 -mfma)
endif()

# Public headers
# Expose include directory for downstream consumers

# Add this directory for header search path

target_include_directories(raif PUBLIC include)

enable_testing()
add_subdirectory(test)
