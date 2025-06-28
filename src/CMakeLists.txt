file(GLOB RAIF_CORE_SRCS
    core/*.cpp
)
file(GLOB RAIF_OP_SRCS
    ops/*.cpp
)
file(GLOB RAIF_KERNEL_SRCS
    ../kernels/cpu/winograd/*.cpp
    ../kernels/cpu/signal/*.cpp
)
add_library(raif STATIC
    ${RAIF_CORE_SRCS}
    ${RAIF_OP_SRCS}
    ${RAIF_KERNEL_SRCS}
)

target_include_directories(raif PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(raif PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(raif PRIVATE -mavx2 -mfma)
endif()
