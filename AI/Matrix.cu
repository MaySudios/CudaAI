#include "matrix.cuh"

__global__ void addKernel(Matrix A, Matrix B, Matrix C) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A.rows && col < A.cols) {
        C(row, col) = A(row, col) + B(row, col);
        printf("%f\t", C(row, col));
    }
}

__global__ void subtractKernel(Matrix A, Matrix B, Matrix C) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A.rows && col < A.cols)
        C(row, col) = A(row, col) - B(row, col);
}

__global__ void scalarMultiplyKernel(Matrix A, float scalar, Matrix B) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A.rows && col < A.cols)
        B(row, col) = A(row, col) * scalar;
}

__global__ void matrixMultiplyKernel(Matrix A, Matrix B, Matrix C) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    printf("\n", A(0, 0));
    if (row < A.rows && col < B.cols) {
        float sum = 0;
        for (int i = 0; i < A.cols; ++i)
            sum += A(row, i) * B(i, col);
        C(row, col) = sum;
    }
}

__global__ void transposeKernel(Matrix A, Matrix B) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A.rows && col < A.cols)
        B(col, row) = A(row, col);
}



Matrix Matrix::operator+(const Matrix& other) {
    Matrix result(rows, cols);
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    addKernel << <gridSize, blockSize >> > (*this, other, result);
    cudaDeviceSynchronize();
    result.print(); // prints the result immediately after the kernel execution
    cudaError_t err = cudaGetLastError(); // checks if there was any error during the kernel execution
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(err) << std::endl;
    }
    return result;
}


Matrix Matrix::operator-(const Matrix& other) {
    Matrix result(rows, cols);
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    subtractKernel << <gridSize, blockSize >> > (*this, other, result);
    cudaDeviceSynchronize();
    return result;
}

Matrix Matrix::operator*(const float& alpha) {
    Matrix result(rows, cols);
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    scalarMultiplyKernel << <gridSize, blockSize >> > (*this, alpha, result);
    cudaDeviceSynchronize();
    return result;
}

Matrix Matrix::transpose() {
    Matrix result(cols, rows);
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    transposeKernel << <gridSize, blockSize >> > (*this, result);
    cudaDeviceSynchronize();
    return result;
}
/*
void Matrix::operator=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        cudaFree(data);
        rows = other.rows;
        cols = other.cols;
        cudaMallocManaged(&data, rows * cols * sizeof(float));
    }
    cudaMemcpy(data, other.data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
}
*/
/*
void Matrix::operator+=(const Matrix& other) {
    *this = *this + other;
}
*/
void Matrix::operator-=(const Matrix& other) {
    *this = *this - other;
}

void Matrix::operator*=(const Matrix& other) {
    *this = *this * other;
}

Matrix Matrix::operator*(const Matrix& other) {
    Matrix result(rows, other.cols);
    dim3 blockSize(16, 16);
    dim3 gridSize((other.cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    matrixMultiplyKernel << <gridSize, blockSize >> > (*this, other, result);
    cudaDeviceSynchronize();
    return result;
}