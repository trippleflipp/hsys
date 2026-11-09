#pragma once

#include "data.h"
#include "matrix_view.h"

#include <cstddef>
#include <memory>

// Фасад: данные (общие, через shared_ptr) + собственное представление
template <typename AtomT>
class Matrix {
public:
    Matrix(std::size_t nrows, std::size_t ncols)
        : data_(std::make_shared<Data<AtomT>>(nrows * ncols)), view_(data_->data(), nrows, ncols) {}

    std::size_t size() const { return view_.size(); }
    std::size_t nrows() const { return view_.nrows(); }
    std::size_t ncols() const { return view_.ncols(); }

    Data<AtomT>& data() { return *data_; }
    const Data<AtomT>& data() const { return *data_; }

    MatrixView<AtomT>& view() { return view_; }
    const MatrixView<AtomT>& view() const { return view_; }

private:
    std::shared_ptr<Data<AtomT>> data_;
    MatrixView<AtomT> view_;
};

// c = a * b для заранее выделенной матрицы c: запускает кёрнел kernel_matmul_naive.
// Запуск асинхронный; копирование результата на хост (copy_to_host) дожидается его завершения.
template <typename AtomT>
void matmul(const Matrix<AtomT>& a, const Matrix<AtomT>& b, Matrix<AtomT>& c);

// Произведение матриц: выделяет матрицу результата и вызывает matmul
template <typename AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& a, const Matrix<AtomT>& b);
