
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "C:\Users\crazy\source\repos\AI\AI\Matrix.cuh"

int main()
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Create two matrices
    Matrix A(2, 2);
    Matrix B(2, 2);

    // Fill the matrices with some values
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;

    B(0, 0) = 5.0f; B(0, 1) = 6.0f;
    B(1, 0) = 7.0f; B(1, 1) = 8.0f;

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        // Handle error...
    }



    // Multiply the matrices
    Matrix C = A + B;

     err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        // Handle error...
    }



    A.print();
    B.print();

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}



