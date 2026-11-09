#include "matrix.h"

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>
#include <vector>

namespace {

TEST(MatrixViewTest, IsTriviallyCopyable) {
    EXPECT_TRUE(std::is_trivially_copyable_v<MatrixView<float>>);
    EXPECT_TRUE(std::is_trivially_copyable_v<MatrixView<double>>);
}

TEST(MatrixViewTest, RowMajorIndexing) {
    std::vector<float> values{1, 2, 3, 4, 5, 6};
    MatrixView<float> view(values.data(), 2, 3);

    EXPECT_EQ(view.size(), 6u);
    EXPECT_EQ(view.nrows(), 2u);
    EXPECT_EQ(view.ncols(), 3u);
    EXPECT_EQ(view(0, 0), 1.0f);
    EXPECT_EQ(view(0, 2), 3.0f);
    EXPECT_EQ(view(1, 0), 4.0f);
    EXPECT_EQ(view(1, 2), 6.0f);
    EXPECT_EQ(view[4], 5.0f);

    view(1, 1) = 50.0f;
    EXPECT_EQ(values[4], 50.0f);
}

TEST(DataTest, CopyFromHostAndBack) {
    const std::vector<float> src{1, 2, 3, 4};
    Data<float> data(src.size());
    data.copy_from_host(src.data());

    std::vector<float> dst(src.size());
    data.copy_to_host(dst.data());
    EXPECT_EQ(dst, src);
}

TEST(DataTest, CopyConstructorIsDeep) {
    const std::vector<float> src{1, 2, 3, 4};
    Data<float> original(src.size());
    original.copy_from_host(src.data());

    Data<float> copy(original);
    EXPECT_NE(copy.data(), original.data());
    EXPECT_EQ(copy.size(), original.size());

    std::vector<float> dst(src.size());
    copy.copy_to_host(dst.data());
    EXPECT_EQ(dst, src);
}

TEST(DataTest, CopyAssignmentSameSizeKeepsBuffer) {
    const std::vector<float> src{1, 2, 3};
    Data<float> a(3);
    Data<float> b(3);
    a.copy_from_host(src.data());
    float* const b_ptr = b.data();

    b = a;
    EXPECT_EQ(b.data(), b_ptr);

    std::vector<float> dst(src.size());
    b.copy_to_host(dst.data());
    EXPECT_EQ(dst, src);
}

TEST(DataTest, CopyAssignmentDifferentSize) {
    const std::vector<float> src{1, 2, 3, 4, 5};
    Data<float> a(src.size());
    a.copy_from_host(src.data());
    Data<float> b(2);

    b = a;
    EXPECT_EQ(b.size(), src.size());
    EXPECT_NE(b.data(), a.data());

    std::vector<float> dst(src.size());
    b.copy_to_host(dst.data());
    EXPECT_EQ(dst, src);
}

TEST(DataTest, MoveConstructorTransfersOwnership) {
    Data<float> original(8);
    float* const ptr = original.data();

    Data<float> moved(std::move(original));
    EXPECT_EQ(moved.data(), ptr);
    EXPECT_EQ(moved.size(), 8u);
    EXPECT_EQ(original.data(), nullptr);
    EXPECT_EQ(original.size(), 0u);
}

TEST(DataTest, MoveAssignmentTransfersOwnership) {
    Data<float> a(8);
    Data<float> b(4);
    float* const ptr = a.data();

    b = std::move(a);
    EXPECT_EQ(b.data(), ptr);
    EXPECT_EQ(b.size(), 8u);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0u);
}

TEST(DataTest, ZeroSize) {
    Data<float> data(0);
    EXPECT_EQ(data.size(), 0u);
    EXPECT_EQ(data.data(), nullptr);
}

TEST(MatrixTest, Dimensions) {
    const Matrix<float> m(3, 5);
    EXPECT_EQ(m.nrows(), 3u);
    EXPECT_EQ(m.ncols(), 5u);
    EXPECT_EQ(m.size(), 15u);
    EXPECT_EQ(m.data().size(), 15u);
    EXPECT_EQ(m.view().size(), 15u);
}

TEST(MatrixTest, CopySharesDataButHasOwnView) {
    Matrix<float> a(2, 3);
    Matrix<float> b = a;

    EXPECT_EQ(&b.data(), &a.data());
    EXPECT_NE(&b.view(), &a.view());
    EXPECT_EQ(b.view().nrows(), a.view().nrows());
}

}  // namespace
