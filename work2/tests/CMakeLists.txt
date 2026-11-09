add_executable(test_matmul
    src/test_matmul.cpp
    src/test_matrix.cpp
)
target_include_directories(test_matmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_matmul PRIVATE matmul_core Eigen3::Eigen GTest::gtest_main)
add_test(NAME test_matmul COMMAND test_matmul)
