add_executable(bench_matmul
    src/bench_matmul.cpp
)
target_include_directories(bench_matmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_matmul PRIVATE matmul_core Eigen3::Eigen benchmark::benchmark_main)
