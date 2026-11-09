#pragma once

#include "cuda_check.h"

// Таймер на CUDA Events: измеряет время выполнения команд на устройстве между start() и stop()
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        const cudaError_t status = cudaEventCreate(&stop_);
        if (status != cudaSuccess) {
            cudaEventDestroy(start_);
            CUDA_CHECK(status);
        }
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { CUDA_CHECK(cudaEventRecord(start_)); }

    // Возвращает время в секундах
    double stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return static_cast<double>(ms) / 1000.0;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};
