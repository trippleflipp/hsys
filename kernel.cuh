#pragma once

template <class T>
__global__ void add(const T* a, const T* b, T* c, int length) {
  uint i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < length) c[i] = a[i] + b[i];
}