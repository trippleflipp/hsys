#include <gtest/gtest.h>
#include "add.h"
#include <vector>

TEST(AddTest, SmallSize) {
    const int n = 1024;
    std::vector<float> a(n), b(n), c_cpu(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    add_cpu(a.data(), b.data(), c_cpu.data(), n);

    // Создаем векторы на GPU и копируем данные с хоста
    Vector<float> a_gpu(n), b_gpu(n), c_gpu_vec(n);
    a_gpu.data().copy_from_host(a.data());
    b_gpu.data().copy_from_host(b.data());
    
    add_gpu(a_gpu, b_gpu, c_gpu_vec);
    
    // Копируем результат обратно на хост
    std::vector<float> c_gpu(n);
    c_gpu_vec.data().copy_to_host(c_gpu.data());

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(c_cpu[i], c_gpu[i]) << "at index " << i;
    }
}

TEST(AddTest, LargeSize) {
    const int n = 1 << 20;
    std::vector<float> a(n), b(n), c_cpu(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 100);
        b[i] = static_cast<float>((i * 3) % 100);
    }

    add_cpu(a.data(), b.data(), c_cpu.data(), n);

    Vector<float> a_gpu(n), b_gpu(n), c_gpu_vec(n);
    a_gpu.data().copy_from_host(a.data());
    b_gpu.data().copy_from_host(b.data());
    
    add_gpu(a_gpu, b_gpu, c_gpu_vec);
    
    std::vector<float> c_gpu(n);
    c_gpu_vec.data().copy_to_host(c_gpu.data());

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(c_cpu[i], c_gpu[i]) << "at index " << i;
    }
}