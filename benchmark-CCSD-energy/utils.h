#ifndef UTILS_H                                                                                                                                                                                                                                      
#define UTILS_H

#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
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
#include <cutensor.h>
#include "H5Cpp.h"

using namespace H5;

#define CUTENSOR_CHECK(x) do { \
    cutensorStatus_t err = x; \
    if (err != CUTENSOR_STATUS_SUCCESS) { \
        std::cerr << "cuTENSOR Error at " << __FILE__ << ":" << __LINE__ \
                  << " -> Status Code: " << err \
	          << "\nMessage : " <<  cutensorGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

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


// Kernels assumed by the routine (if not already defined elsewhere)
template <typename T>
__global__ void exponential_scaler_kernel(T* d_data, T phi, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] = exp(phi * d_data[idx]);
    }
}

template <typename T>
__global__ void elemwise_mul_kernel(T* d_out, const T* d_in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] *= d_in[idx];
    }
}

/*
 * Generates random tensors with distribution: random(-1, 1) * exp(phi * gaussian(0, 1))
 * Supports both real and complex types (via template parameters T and TC).
 */
template <typename T=double, typename TC=cuDoubleComplex>
void generate_random_matrix(TC* devA, size_t sizeA, T phi, int seed, bool ONLY_REAL_PART = false) {
    // Total scalar elements (accounts for real/imag pairs if TC is a complex type)
    size_t total_elements = sizeof(TC) > sizeof(T) ? 2 * sizeA : sizeA;

    // 1. Create cuRAND generator
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

    // 2. Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if constexpr (sizeof(T) == sizeof(float)) {
        // Generate uniform [0, 1) and scale to [-1, 1] via 2 * U - 1
        CURAND_CHECK(curandGenerateUniform(gen, (float*)devA, total_elements));

        float two = 2.0f;
        float minus_one = -1.0f;
        CUBLAS_CHECK(cublasSscal(handle, total_elements, &two, (float*)devA, 1));

        float* d_ones = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ones, total_elements * sizeof(float)));
        std::vector<float> h_ones(total_elements, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_ones, h_ones.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSaxpy(handle, total_elements, &minus_one, d_ones, 1, (float*)devA, 1));

        // Generate Gaussian N(0,1) for exponential scaling factor: exp(phi * G)
        CURAND_CHECK(curandGenerateNormal(gen, d_ones, total_elements, 0.0f, 1.0f));
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        exponential_scaler_kernel<<<blocks, threads>>>(d_ones, (float)phi, total_elements);
        elemwise_mul_kernel<<<blocks, threads>>>((float*)devA, d_ones, total_elements);
        cudaFree(d_ones);

    } else if constexpr (sizeof(T) == sizeof(double)) {
        // Generate uniform [0, 1) and scale to [-1, 1] via 2 * U - 1
        CURAND_CHECK(curandGenerateUniformDouble(gen, (double*)devA, total_elements));

        double two = 2.0;
        double minus_one = -1.0;
        CUBLAS_CHECK(cublasDscal(handle, total_elements, &two, (double*)devA, 1));

        double* d_ones = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ones, total_elements * sizeof(double)));
        std::vector<double> h_ones(total_elements, 1.0);
        CUDA_CHECK(cudaMemcpy(d_ones, h_ones.data(), total_elements * sizeof(double), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasDaxpy(handle, total_elements, &minus_one, d_ones, 1, (double*)devA, 1));

        // Generate Gaussian N(0,1) for exponential scaling factor: exp(phi * G)
        CURAND_CHECK(curandGenerateNormalDouble(gen, d_ones, total_elements, 0.0, 1.0));
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        exponential_scaler_kernel<<<blocks, threads>>>(d_ones, (double)phi, total_elements);
        elemwise_mul_kernel<<<blocks, threads>>>((double*)devA, d_ones, total_elements);
        cudaFree(d_ones);
    }

    // 3. Handle ONLY_REAL_PART constraint for complex types
    if (ONLY_REAL_PART && sizeof(TC) > sizeof(T)) {
        T zero = (T)0.0;
        if constexpr (sizeof(T) == sizeof(float)) {
            CUBLAS_CHECK(cublasSscal(handle, sizeA, &zero, ((float*)devA) + 1, 2));
        } else if constexpr (sizeof(T) == sizeof(double)) {
            CUBLAS_CHECK(cublasDscal(handle, sizeA, &zero, ((double*)devA) + 1, 2));
        }
    }

    // Cleanup resources
    CURAND_CHECK(curandDestroyGenerator(gen));
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

template <typename T=double>
T get_max_relative_error(double* d_A, T* d_B, size_t num_elements) {
    thrust::device_ptr<double> ptr_A(d_A);
    thrust::device_ptr<T> ptr_B(d_B);

    thrust::device_vector<double> ptr_B_double(num_elements);

    thrust::transform(
        thrust::device,
        ptr_B,
        ptr_B + num_elements,
        ptr_B_double.begin(),
        [] __device__ (T val) {
            return static_cast<double>(val);
        }
    );

    double max_relative_error = thrust::inner_product(
        ptr_A, ptr_A + num_elements,
        ptr_B_double.begin(),
        0.0,                          // Initial value for maximum
        thrust::maximum<double>(),     // Finds the highest one
        max_rel_error_op<double>()           // Calculates relative error per element
    );

    return max_relative_error;
}

template <typename T>
struct max_abs_error_op {
    __host__ __device__ T operator()(const T a, const T b) const {
        return std::fabs(a - b);
    }
};

template <typename T=double>
T get_max_absolute_error(double* d_A, T* d_B, size_t num_elements) {
    thrust::device_ptr<double> ptr_A(d_A);
    thrust::device_ptr<T> ptr_B(d_B);

    thrust::device_vector<double> ptr_B_double(num_elements);

    thrust::transform(
        thrust::device,
        ptr_B,
        ptr_B + num_elements,
        ptr_B_double.begin(),
        [] __device__ (T val) {
            return static_cast<double>(val);
        }
    );


    double max_absolute_error = thrust::inner_product(
        ptr_A, ptr_A + num_elements,
        ptr_B_double.begin(),
        0.0,                          // Initial value for maximum
        thrust::maximum<double>(),     // Finds the highest one
        max_abs_error_op<double>()           // Calculates absolute error per element
    );

    return max_absolute_error;
}

inline void complex_double_to_float(const cuDoubleComplex* devA, cuComplex* devA_T, size_t nA) {
    thrust::device_ptr<const cuDoubleComplex> ptr_A(devA);
    thrust::device_ptr<cuComplex> ptr_A_T(devA_T);

    thrust::transform(
        thrust::device,
        ptr_A,
        ptr_A + nA,
        ptr_A_T,
        [] __device__ (cuDoubleComplex val) {
            return make_cuComplex(
                static_cast<float>(val.x),
                static_cast<float>(val.y)
            );
        }
    );
}

inline void complex_float_to_double(const cuComplex* devA, cuDoubleComplex* devA_T, size_t nA) {
    thrust::device_ptr<const cuComplex> ptr_A(devA);
    thrust::device_ptr<cuDoubleComplex> ptr_A_double(devA_T);

    thrust::transform(
        thrust::device,
        ptr_A,
        ptr_A + nA,
        ptr_A_double,
        [] __device__ (cuComplex val) {
            return make_cuDoubleComplex(
                static_cast<double>(val.x),
                static_cast<double>(val.y)
            );
        }
    );
}

void read_hdf5(cuDoubleComplex* T, size_t n, std::string filename) {
    printf("Reading data from file %s with n = %i\n", filename.c_str(), n);
    const H5std_string FILE_NAME(filename);
    try {
        // Turn off auto-printing for error handling
        H5::Exception::dontPrint();

        // Open the HDF5 file
        H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);

        // Open the group "data"
        H5::Group data_group = file.openGroup("data");

        // 1. Read the "real" dataset
        H5::DataSet real_dataset = data_group.openDataSet("real");
        std::vector<double> real_buf(n);
        real_dataset.read(real_buf.data(), H5::PredType::NATIVE_DOUBLE, H5::DataSpace::ALL, H5::DataSpace::ALL);

        // 2. Read the "aimag" dataset
        H5::DataSet aimag_dataset = data_group.openDataSet("aimag");
        std::vector<double> aimag_buf(n);
        aimag_dataset.read(aimag_buf.data(), H5::PredType::NATIVE_DOUBLE, H5::DataSpace::ALL, H5::DataSpace::ALL);

        // 3. Interleave real and imaginary parts into cuDoubleComplex format
        for (size_t i = 0; i < n; ++i) {
            T[i].x = real_buf[i]; // Real part
            T[i].y = aimag_buf[i]; // Imaginary part
        }

    }
    catch (H5::FileIException error) {
	printf("Failure in read_hdf5\n");
        error.printErrorStack();
    }
    catch (H5::DataSetIException error) {
	printf("Failure in read_hdf5\n");
        error.printErrorStack();
    }
    catch (H5::GroupIException error) {
	printf("Failure in read_hdf5\n");
        error.printErrorStack();
    }

    return;
}

#endif

