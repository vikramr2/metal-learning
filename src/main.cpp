#include "../include/MetalMatrixVec.h"
#include <iostream>
#include <vector>
#include <chrono>

// Helper function to print a vector
void printVector(const std::vector<float>& vec, const std::string& name) {
    std::cout << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Helper function to print a matrix
void printMatrix(const std::vector<std::vector<float>>& matrix, const std::string& name) {
    std::cout << name << ":" << std::endl;
    for (const auto& row : matrix) {
        std::cout << "  [";
        for (size_t i = 0; i < row.size(); ++i) {
            std::cout << row[i];
            if (i < row.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

// CPU implementation for comparison
std::vector<float> cpuMatrixVectorMultiply(
    const std::vector<std::vector<float>>& matrix,
    const std::vector<float>& vec) {
    
    size_t numRows = matrix.size();
    std::vector<float> result(numRows, 0.0f);
    
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    
    return result;
}

int main() {
    // Create a sample matrix and vector
    std::vector<std::vector<float>> matrix = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f},
        {10.0f, 11.0f, 12.0f}
    };
    
    std::vector<float> vector = {2.0f, 3.0f, 4.0f};
    
    // Print inputs
    printMatrix(matrix, "Matrix");
    printVector(vector, "Vector");
    
    try {
        // Initialize Metal
        MetalMatrixVec metalMultiplier;
        if (!metalMultiplier.initialize()) {
            std::cerr << "Failed to initialize Metal" << std::endl;
            return 1;
        }
        
        std::cout << "Metal initialized successfully" << std::endl;
        
        // Measure GPU performance
        auto startGPU = std::chrono::high_resolution_clock::now();
        std::vector<float> gpuResult = metalMultiplier.multiply(matrix, vector);
        auto endGPU = std::chrono::high_resolution_clock::now();
        
        // Measure CPU performance
        auto startCPU = std::chrono::high_resolution_clock::now();
        std::vector<float> cpuResult = cpuMatrixVectorMultiply(matrix, vector);
        auto endCPU = std::chrono::high_resolution_clock::now();
        
        // Print results
        printVector(gpuResult, "GPU Result");
        printVector(cpuResult, "CPU Result");
        
        // Print timing info
        auto gpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endGPU - startGPU).count();
        auto cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU).count();
        
        std::cout << "GPU Time: " << gpuTime << " microseconds" << std::endl;
        std::cout << "CPU Time: " << cpuTime << " microseconds" << std::endl;
        std::cout << "Speedup: " << (cpuTime > 0 ? (float)cpuTime / gpuTime : 0) << "x" << std::endl;
        
        // Validate results
        bool correct = true;
        for (size_t i = 0; i < cpuResult.size(); ++i) {
            if (std::abs(cpuResult[i] - gpuResult[i]) > 1e-5) {
                std::cout << "Error at index " << i << ": CPU = " << cpuResult[i] 
                          << ", GPU = " << gpuResult[i] << std::endl;
                correct = false;
            }
        }
        
        if (correct) {
            std::cout << "Results match! The GPU computation is correct." << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
