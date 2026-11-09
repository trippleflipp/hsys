add_library(add_lib STATIC
    core/kernel.cu
    core/add_cpu.cpp
)
target_include_directories(add_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(add_lib PUBLIC Eigen3::Eigen CUDA::cudart)
cuda_setup_targets(TARGETS add_lib CUDA_MIN_ARCH 75)

# Тесты
add_executable(test_add tests/test_add.cpp)
target_link_libraries(test_add PRIVATE add_lib GTest::gtest_main)
add_test(NAME test_add COMMAND test_add)

# Бенчмарки
add_executable(bench_add benchmarks/bench_add.cu)
target_link_libraries(bench_add PRIVATE add_lib benchmark::benchmark_main)
cuda_setup_targets(TARGETS bench_add CUDA_MIN_ARCH 75)