#include "add.h"
#include <Eigen/Dense>

void add_cpu(const float* a, const float* b, float* c, int length) {
    Eigen::Map<const Eigen::VectorXf> va(a, length);
    Eigen::Map<const Eigen::VectorXf> vb(b, length);
    Eigen::Map<Eigen::VectorXf> vc(c, length);
    vc = va + vb;
}