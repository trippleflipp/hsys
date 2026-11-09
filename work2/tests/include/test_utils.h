#pragma once

#include "matrix.h"

#include <Eigen/Dense>

#include <cstddef>

using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Eigen по умолчанию хранит матрицы по столбцам (column-major), а Matrix — по строкам (row-major),
// поэтому при обмене данными матрица преобразуется в построчный формат

inline Matrix<float> to_device(const Eigen::MatrixXf& host) {
    const RowMajorMatrixXf row_major = host;
    Matrix<float> device(static_cast<std::size_t>(host.rows()), static_cast<std::size_t>(host.cols()));
    device.data().copy_from_host(row_major.data());
    return device;
}

inline Eigen::MatrixXf to_host(const Matrix<float>& device) {
    RowMajorMatrixXf row_major(static_cast<Eigen::Index>(device.nrows()),
                               static_cast<Eigen::Index>(device.ncols()));
    device.data().copy_to_host(row_major.data());
    return row_major;
}
