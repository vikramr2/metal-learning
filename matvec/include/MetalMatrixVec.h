#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <vector>
#include <iostream>

class MetalMatrixVec {
public:
    MetalMatrixVec();
    ~MetalMatrixVec();
    
    // Initialize Metal
    bool initialize();
    
    // Multiply matrix by vector: result = matrix * vector
    std::vector<float> multiply(
        const std::vector<std::vector<float>>& matrix, 
        const std::vector<float>& vec);
    
private:
    // Metal objects
    MTL::Device* _device;
    MTL::CommandQueue* _commandQueue;
    MTL::Library* _library;
    MTL::Function* _function;
    MTL::ComputePipelineState* _pipelineState;
    
    // Helper method to create a Metal buffer
    MTL::Buffer* createBuffer(const void* data, size_t size);
};
