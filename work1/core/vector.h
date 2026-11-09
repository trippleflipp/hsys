#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <cstddef>
#include <stdexcept>

template <typename AtomT>
class VectorView {
private:
    AtomT* data_;
    std::size_t size_;

public:
    VectorView(AtomT* data, std::size_t size) : data_(data), size_(size) {}

    __host__ __device__ std::size_t size() const { return size_; }

    __host__ __device__ AtomT& operator[](std::size_t n) { return data_[n]; }
    __host__ __device__ const AtomT& operator[](std::size_t n) const { return data_[n]; }

    __host__ __device__ AtomT& operator()(std::size_t i) { return data_[i]; }
    __host__ __device__ const AtomT& operator()(std::size_t i) const { return data_[i]; }
};

template <typename AtomT>
class Data {
private:
    std::size_t size_;
    AtomT* data_;

public:
    Data(std::size_t size) : size_(size) {
        if (cudaMalloc(&data_, size_ * sizeof(AtomT)) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
    }

    Data(const Data& other) : size_(other.size_) {
        if (cudaMalloc(&data_, size_ * sizeof(AtomT)) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
        cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
    }

    Data(Data&& other) noexcept : size_(other.size_), data_(other.data_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    Data& operator=(const Data& other) {
        if (this != &other) {
            if (data_) cudaFree(data_);
            size_ = other.size_;
            if (cudaMalloc(&data_, size_ * sizeof(AtomT)) != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed");
            }
            cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
        }
        return *this;
    }

    Data& operator=(Data&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
            size_ = other.size_;
            data_ = other.data_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    AtomT* data() { return data_; }
    const AtomT* data() const { return data_; }
    std::size_t size() const { return size_; }

    void copy_to_host(AtomT* host_ptr) const {
        cudaMemcpy(host_ptr, data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToHost);
    }

    void copy_from_host(const AtomT* host_ptr) {
        cudaMemcpy(data_, host_ptr, size_ * sizeof(AtomT), cudaMemcpyHostToDevice);
    }

    ~Data() {
        if (data_) cudaFree(data_);
    }
};

template <typename AtomT>
class Vector {
private:
    std::shared_ptr<Data<AtomT>> data_;
    VectorView<AtomT> view_;

public:
    Vector(std::size_t size) 
        : data_(std::make_shared<Data<AtomT>>(size)), 
          view_(data_->data(), size) {}

    std::size_t size() const { return data_->size(); }
    
    Data<AtomT>& data() { return *data_; }
    const Data<AtomT>& data() const { return *data_; }

    VectorView<AtomT>& view() { return view_; }
    const VectorView<AtomT>& view() const { return view_; }
};