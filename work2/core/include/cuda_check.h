#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

// Проверка кода возврата функций CUDA Runtime API: при ошибке выбрасывает исключение
inline void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + expr +
                                 " failed: " + cudaGetErrorString(status));
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)
