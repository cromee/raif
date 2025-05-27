add_executable(raif_test test_main.cpp)
target_link_libraries(raif_test PRIVATE raif)
add_test(NAME raif_test COMMAND raif_test)
