#pragma once
#include "vector.h"

void add_cpu(const float* a, const float* b, float* c, int length);

template <typename AtomT>
void add_gpu(Vector<AtomT>& a, Vector<AtomT>& b, Vector<AtomT>& c);