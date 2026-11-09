#include "test_utils.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {

constexpr float kPrecision = 1e-5f;

// Параметры: m, k, n — размеры матриц A (m x k) и B (k x n)
using MatMulParams = std::tuple<std::size_t, std::size_t, std::size_t>;

class MatMulTest : public ::testing::TestWithParam<MatMulParams> {
protected:
    void SetUp() override { std::srand(42); }
};

TEST_P(MatMulTest, MatchesEigen) {
    const auto [m, k, n] = GetParam();

    const Eigen::MatrixXf a = Eigen::MatrixXf::Random(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(k));
    const Eigen::MatrixXf b = Eigen::MatrixXf::Random(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(n));
    const Eigen::MatrixXf expected = a * b;

    const Matrix<float> a_gpu = to_device(a);
    const Matrix<float> b_gpu = to_device(b);
    const Matrix<float> c_gpu = a_gpu * b_gpu;

    ASSERT_EQ(c_gpu.nrows(), m);
    ASSERT_EQ(c_gpu.ncols(), n);

    const Eigen::MatrixXf actual = to_host(c_gpu);
    EXPECT_TRUE(actual.isApprox(expected, kPrecision))
        << "max |actual - expected| = " << (actual - expected).cwiseAbs().maxCoeff();
}


std::string param_name(const ::testing::TestParamInfo<MatMulParams>& info) {
    const auto [m, k, n] = info.param;
    return "m" + std::to_string(m) + "_k" + std::to_string(k) + "_n" + std::to_string(n);
}

const auto kSizes = ::testing::Values(1, 2, 3, 127, 128, 129, 512);

INSTANTIATE_TEST_SUITE_P(AllSizes, MatMulTest, ::testing::Combine(kSizes, kSizes, kSizes), param_name);

TEST(MatMulErrors, ThrowsOnIncompatibleSizes) {
    const Matrix<float> a(2, 3);
    const Matrix<float> b(4, 5);
    EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(MatMulErrors, MatmulThrowsOnWrongResultSize) {
    const Matrix<float> a(2, 3);
    const Matrix<float> b(3, 4);
    Matrix<float> c(2, 5);
    EXPECT_THROW(matmul(a, b, c), std::invalid_argument);
}

}  // namespace
