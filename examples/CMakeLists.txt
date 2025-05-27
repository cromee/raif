cmake_minimum_required(VERSION 3.10)

find_package(Torch REQUIRED)

add_executable(image_classification
    image_classification/image_classification.cpp
)

target_link_libraries(image_classification PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET image_classification PROPERTY CXX_STANDARD 17)
