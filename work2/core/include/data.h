#pragma once

#include "cuda_check.h"

#include <cstddef>
#include <utility>

// Владелец массива в глобальной памяти устройства (идиома RAII)
template <typename AtomT>
class Data {
public:
    explicit Data(std::size_t size) : size_(size) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(AtomT)));
        }
    }

    Data(const Data& other) : Data(other.size_) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice));
        }
    }

    Data(Data&& other) noexcept
        : size_(std::exchange(other.size_, 0)), data_(std::exchange(other.data_, nullptr)) {}

    Data& operator=(const Data& other) {
        if (this == &other) {
            return *this;
        }
        if (size_ == other.size_) {
            // Размер совпадает: копируем в существующий буфер, указатель не меняется
            if (size_ > 0) {
                CUDA_CHECK(cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice));
            }
            return *this;
        }
        Data copy(other);  // при ошибке выделения *this остаётся без изменений
        swap(copy);
        return *this;
    }

    Data& operator=(Data&& other) noexcept {
        if (this != &other) {
            Data moved(std::move(other));
            swap(moved);
        }
        return *this;
    }

    ~Data() {
        if (data_) {
            cudaFree(data_);
        }
    }

    AtomT* data() noexcept { return data_; }
    const AtomT* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

    void copy_to_host(AtomT* host_ptr) const {
        if (size_ > 0) {
            CUDA_CHECK(cudaMemcpy(host_ptr, data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToHost));
        }
    }

    void copy_from_host(const AtomT* host_ptr) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMemcpy(data_, host_ptr, size_ * sizeof(AtomT), cudaMemcpyHostToDevice));
        }
    }

private:
    void swap(Data& other) noexcept {
        std::swap(size_, other.size_);
        std::swap(data_, other.data_);
    }

    std::size_t size_ = 0;
    AtomT* data_ = nullptr;
};
