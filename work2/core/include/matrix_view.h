#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

// Представление матрицы на устройстве: указатель и размеры, без владения памятью.
// Элементы хранятся построчно (row-major): элемент (i, j) лежит по индексу i * ncols + j.
template <typename AtomT>
class MatrixView {
public:
    MatrixView(AtomT* data, std::size_t nrows, std::size_t ncols)
        : data_(data), nrows_(nrows), ncols_(ncols) {}

    __host__ __device__ std::size_t size() const { return nrows_ * ncols_; }
    __host__ __device__ std::size_t nrows() const { return nrows_; }
    __host__ __device__ std::size_t ncols() const { return ncols_; }

    __host__ __device__ AtomT& operator[](std::size_t n) { return data_[n]; }
    __host__ __device__ const AtomT& operator[](std::size_t n) const { return data_[n]; }

    __host__ __device__ AtomT& operator()(std::size_t i, std::size_t j) { return data_[i * ncols_ + j]; }
    __host__ __device__ const AtomT& operator()(std::size_t i, std::size_t j) const {
        return data_[i * ncols_ + j];
    }

private:
    AtomT* data_;
    std::size_t nrows_;
    std::size_t ncols_;
};

static_assert(std::is_trivially_copyable_v<MatrixView<float>>,
              "MatrixView must be trivially copyable to be passed to a kernel by value");
