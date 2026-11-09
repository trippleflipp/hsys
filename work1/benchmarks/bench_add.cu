#include <benchmark/benchmark.h>
#include "add.h"
#include <vector>

static void BM_AddCPU(benchmark::State& state) {
    const int n = state.range(0);
    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n);

    for (auto _ : state) {
        add_cpu(a.data(), b.data(), c.data(), n);
    }
}
BENCHMARK(BM_AddCPU)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);

static void BM_AddGPU(benchmark::State& state) {
    const int n = state.range(0);
    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n);
    
    // Выделяем память один раз до цикла
    Vector<float> a_gpu(n), b_gpu(n), c_gpu_vec(n);
    
    for (auto _ : state) {
        // Замеряем полный цикл: передача на GPU -> вычисление -> возврат на CPU
        a_gpu.data().copy_from_host(a.data());
        b_gpu.data().copy_from_host(b.data());
        
        add_gpu(a_gpu, b_gpu, c_gpu_vec);
        
        c_gpu_vec.data().copy_to_host(c.data());
    }
}
BENCHMARK(BM_AddGPU)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);