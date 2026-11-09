add_library(matmul_core STATIC
    src/matmul.cu
)
target_include_directories(matmul_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(matmul_core PUBLIC CUDA::cudart)
target_compile_features(matmul_core PUBLIC cxx_std_20 cuda_std_20)
cuda_setup_targets(TARGETS matmul_core CUDA_MIN_ARCH 75)
