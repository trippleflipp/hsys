#include "add.h"

template <typename AtomT>
__global__ void add_kernel(VectorView<AtomT> a, VectorView<AtomT> b, VectorView<AtomT> c) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < a.size()) c[i] = a[i] + b[i];
}

template <typename AtomT>
void add_gpu(Vector<AtomT>& a, Vector<AtomT>& b, Vector<AtomT>& c) {
    std::size_t length = a.size();
    int threads = 256;
    int blocks = (length + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a.view(), b.view(), c.view());
    cudaDeviceSynchronize();
}

template void add_gpu<float>(Vector<float>& a, Vector<float>& b, Vector<float>& c);