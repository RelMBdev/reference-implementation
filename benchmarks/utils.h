#ifndef UTILS_H                                                                                                                                                                                                                                      
#define UTILS_H

#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/std/cmath>
#include <float.h>
#include <time.h>
#include <random>
#include <iostream>
#include <cublas_v2.h>
#include <complex>

//For cuRAND Host API calls (curandCreateGenerator, etc.)
#define CURAND_CHECK(x) do { \
    curandStatus_t err = x; \
    if (err != CURAND_STATUS_SUCCESS) { \
        std::cerr << "cuRAND Error at " << __FILE__ << ":" << __LINE__ \
                  << " -> Status Code: " << err << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Macro for checking CUDA runtime API calls
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA Error at %s:%d -> %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Macro for checking cuBLAS API calls (Uses modern CUDA status strings)
#define CUBLAS_CHECK(call)                                               \
    do {                                                                 \
        cublasStatus_t stat = (call);                                    \
        if (stat != CUBLAS_STATUS_SUCCESS) {                             \
            fprintf(stderr, "cuBLAS Error at %s:%d -> %s\n",             \
                    __FILE__, __LINE__, cublasGetStatusString(stat));    \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)


// Generate random matrices with cuRAND according to this paper: 10.48550/arXiv.2306.11975
template<typename T, typename TC>
void generate_random_matrix(TC* devA, size_t sizeA, T phi, int seed){
	
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

    if(sizeof(T)==sizeof(float)) CURAND_CHECK(curandGenerateUniform      (gen, (float*) devA, 2*sizeA));
    if(sizeof(T)==sizeof(double)) CURAND_CHECK(curandGenerateUniformDouble(gen, (double*) devA, 2*sizeA));

    CURAND_CHECK(curandDestroyGenerator(gen));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    T factor = static_cast<T>(std::exp(phi));
    if constexpr (sizeof(T)==sizeof(float)) CUBLAS_CHECK(cublasSscal(handle, 2*sizeA, &factor, (float*) devA, 1));
    if constexpr (sizeof(T)==sizeof(double)) CUBLAS_CHECK(cublasDscal(handle, 2*sizeA, &factor, (double*) devA, 1));

    CUBLAS_CHECK(cublasDestroy(handle));
}

// Compute error metrics with thrust
template <typename T>
struct max_rel_error_op {
    __host__ __device__ T operator()(const T a, const T b) const {
        // 'a' is the reference value, 'b' is the test value
        T abs_a = std::fabs(a);
        if (abs_a < 1e-16) {
            // Avoid division by zero. If both are zero, error is 0.
            // If b is non-zero, return its absolute difference.
            return 0.0;
        }
        return std::fabs(a - b) / abs_a;
    }
};

template <typename T>
T get_max_relative_error(T* d_A, T* d_B, size_t num_elements) {
    thrust::device_ptr<T> ptr_A(d_A);
    thrust::device_ptr<T> ptr_B(d_B);

    T max_relative_error = thrust::inner_product(
        ptr_A, ptr_A + num_elements,
        ptr_B,
        0.0,                          // Initial value for maximum
        thrust::maximum<T>(),     // Finds the highest one
        max_rel_error_op<T>()           // Calculates relative error per element
    );

    return max_relative_error;
}

template <typename T>
struct max_abs_error_op {
    __host__ __device__ T operator()(const T a, const T b) const {
        return std::fabs(a - b);
    }
};

template <typename T>
T get_max_absolute_error(T* d_A, T* d_B, size_t num_elements) {
    thrust::device_ptr<T> ptr_A(d_A);
    thrust::device_ptr<T> ptr_B(d_B);

    T max_absolute_error = thrust::inner_product(
        ptr_A, ptr_A + num_elements,
        ptr_B,
        0.0,                          // Initial value for maximum
        thrust::maximum<T>(),     // Finds the highest one
        max_abs_error_op<T>()           // Calculates absolute error per element
    );

    return max_absolute_error;
}

#endif

