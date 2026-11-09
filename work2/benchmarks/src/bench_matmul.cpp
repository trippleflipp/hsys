#include "cuda_timer.h"
#include "matrix.h"

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <vector>

namespace {

std::vector<float> random_values(std::size_t count, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> values(count);
    for (auto& value : values) {
        value = dist(gen);
    }
    return values;
}

// operator* для Matrix: выделение памяти и копирование данных выполняются до цикла измерений,
// в цикле замеряется только вычисление (запуск kernel_matmul_naive) с помощью CUDA Events
void BM_MatMulGPU(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));

    Matrix<float> a(n, n);
    Matrix<float> b(n, n);
    Matrix<float> c(n, n);
    a.data().copy_from_host(random_values(n * n, 1).data());
    b.data().copy_from_host(random_values(n * n, 2).data());

    matmul(a, b, c);  // прогрев
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    for (auto _ : state) {
        timer.start();
        matmul(a, b, c);
        state.SetIterationTime(timer.stop());
    }
}

// operator* для Eigen::MatrixXf: матрица результата выделяется до цикла измерений
void BM_MatMulEigen(benchmark::State& state) {
    const auto n = static_cast<Eigen::Index>(state.range(0));

    const Eigen::MatrixXf a = Eigen::MatrixXf::Random(n, n);
    const Eigen::MatrixXf b = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf c(n, n);

    for (auto _ : state) {
        c.noalias() = a * b;
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }
}

}  // namespace

BENCHMARK(BM_MatMulEigen)->RangeMultiplier(2)->Range(16, 1024);
BENCHMARK(BM_MatMulGPU)->RangeMultiplier(2)->Range(16, 1024)->UseManualTime();
