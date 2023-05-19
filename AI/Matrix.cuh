
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

class Matrix {
private:


public:

    int rows, cols;
    float* data;

    // Constructor
    Matrix(int rows, int cols) : rows(rows), cols(cols) {
        cudaError_t err = cudaMallocManaged(&data, rows * cols * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Fehler bei cudaMallocManaged: " << cudaGetErrorString(err) << std::endl;
            // Handle error...
        }
        else {
            std::cout << "Erfolg bei cudaMallocManaged" << std::endl;
            for (int i = 0; i < rows * cols; ++i) {
                data[i] = 0.0f;
            }
        }

    }

    // Kopierkonstruktor
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        cudaError_t err = cudaMallocManaged(&data, rows * cols * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Fehler bei cudaMallocManaged: " << cudaGetErrorString(err) << std::endl;
        }
        else {
            cudaMemcpy(data, other.data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }

    // Move-Konstruktor
    Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(other.data) {
        other.data = nullptr;
    }

    // Move-Zuweisungsoperator
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            cudaFree(data);
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }

    // Überschreibe den Zuweisungsoperator
    Matrix& operator=(const Matrix& other) {
        if (this != &other) { // Schützt vor Selbstzuweisung
            rows = other.rows;
            cols = other.cols;

            cudaFree(data); // Alten Speicher freigeben

            cudaError_t err = cudaMallocManaged(&data, rows * cols * sizeof(float));
            if (err != cudaSuccess) {
                std::cerr << "Fehler bei cudaMallocManaged: " << cudaGetErrorString(err) << std::endl;
            }
            else {
                memcpy(data, other.data, rows * cols * sizeof(float));
            }
        }

        return *this;
    }

    // Destructor
    ~Matrix() {
        cudaFree(data);
    }

    // Getters
    int getRows() const { return rows; }
    int getCols() const { return cols; }

    // Set an element
    void set(int row, int col, float value) {
        data[row * cols + col] = value;
    }

    // Get an element
    float get(int row, int col) const {
        return data[row * cols + col];
    }

    // Überschreibe den Operator '()' für direkten Zugriff auf die Elemente
    __host__ __device__ float& operator()(int row, int col) {
        return data[row * cols + col];
    }

    __host__ __device__ const float& operator()(int row, int col) const {
        return data[row * cols + col];
    }

    Matrix transpose();

    Matrix operator+(const Matrix& other);

    Matrix operator-(const Matrix& other);

    Matrix operator*(const Matrix& other);

    Matrix operator*(const float& alpha);

    void operator+=(const Matrix& other);

    void operator-=(const Matrix& other);

    void operator*=(const Matrix& other);

    void print() const {
        std::cout << std::endl;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << data[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

};