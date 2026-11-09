#include "matrix.h"

#include <stdexcept>

namespace {

constexpr unsigned kBlockSize = 16;

}  // namespace

// Наивное умножение без разделяемой памяти: каждая нить вычисляет один элемент c(row, col).
// Ось x сетки соответствует столбцам, ось y — строкам, поэтому соседние нити варпа
// обращаются к соседним элементам строк матриц b и c.
template <typename AtomT>
__global__ void kernel_matmul_naive(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= c.nrows() || col >= c.ncols()) {
        return;
    }

    AtomT sum{0};
    for (std::size_t p = 0; p < a.ncols(); ++p) {
        sum += a(row, p) * b(p, col);
    }
    c(row, col) = sum;
}

template <typename AtomT>
void matmul(const Matrix<AtomT>& a, const Matrix<AtomT>& b, Matrix<AtomT>& c) {
    if (a.ncols() != b.nrows() || c.nrows() != a.nrows() || c.ncols() != b.ncols()) {
        throw std::invalid_argument("matmul: incompatible matrix sizes");
    }
    if (c.size() == 0) {
        return;
    }

    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid(static_cast<unsigned>((c.ncols() + block.x - 1) / block.x),
                    static_cast<unsigned>((c.nrows() + block.y - 1) / block.y));

    kernel_matmul_naive<<<grid, block>>>(a.view(), b.view(), c.view());
    CUDA_CHECK(cudaGetLastError());
}

template <typename AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& a, const Matrix<AtomT>& b) {
    if (a.ncols() != b.nrows()) {
        throw std::invalid_argument("operator*: incompatible matrix sizes");
    }
    Matrix<AtomT> c(a.nrows(), b.ncols());
    matmul(a, b, c);
    return c;
}

template void matmul<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void matmul<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template Matrix<float> operator*<float>(const Matrix<float>&, const Matrix<float>&);
template Matrix<double> operator*<double>(const Matrix<double>&, const Matrix<double>&);
