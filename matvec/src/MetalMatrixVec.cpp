#include "../include/MetalMatrixVec.h"

MetalMatrixVec::MetalMatrixVec() : 
    _device(nullptr),
    _commandQueue(nullptr),
    _library(nullptr),
    _function(nullptr),
    _pipelineState(nullptr) {
}

MetalMatrixVec::~MetalMatrixVec() {
    // Release Metal objects
    if (_pipelineState) _pipelineState->release();
    if (_function) _function->release();
    if (_library) _library->release();
    if (_commandQueue) _commandQueue->release();
    if (_device) _device->release();
}

bool MetalMatrixVec::initialize() {
    // Get default Metal device
    _device = MTL::CreateSystemDefaultDevice();
    if (!_device) {
        std::cerr << "Failed to create Metal device" << std::endl;
        return false;
    }
    
    // Create command queue
    _commandQueue = _device->newCommandQueue();
    if (!_commandQueue) {
        std::cerr << "Failed to create command queue" << std::endl;
        return false;
    }
    
    // Load the Metal shader library
    NS::Error* error = nullptr;
    
    // First try to load the metallib file directly from the current directory
    NS::String* libraryPath = NS::String::string("./shader.metallib", 
                                              NS::StringEncoding::UTF8StringEncoding);
    
    _library = _device->newLibrary(libraryPath, &error);
    
    // If that fails, try loading from the bin directory
    if (!_library) {
        libraryPath = NS::String::string("./bin/shader.metallib", 
                                      NS::StringEncoding::UTF8StringEncoding);
        _library = _device->newLibrary(libraryPath, &error);
    }
    
    // If both fail, try the build directory
    if (!_library) {
        libraryPath = NS::String::string("./build/shader.metallib", 
                                     NS::StringEncoding::UTF8StringEncoding);
        _library = _device->newLibrary(libraryPath, &error);
    }
    
    // If all the above fail, try to get the default library
    if (!_library) {
        _library = _device->newDefaultLibrary();
    }
    
    // If we still don't have a library, report error and exit
    if (!_library) {
        std::cerr << "Failed to load Metal library. Make sure shader.metallib is in the current, bin, or build directory" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return false;
    }
    
    // Get the kernel function
    _function = _library->newFunction(NS::String::string("matrix_vector_multiply", 
                                                        NS::StringEncoding::UTF8StringEncoding));
    if (!_function) {
        std::cerr << "Failed to find the kernel function" << std::endl;
        return false;
    }
    
    // Create compute pipeline
    _pipelineState = _device->newComputePipelineState(_function, &error);
    if (!_pipelineState) {
        std::cerr << "Failed to create pipeline state: ";
        if (error) {
            std::cerr << error->localizedDescription()->utf8String();
        }
        std::cerr << std::endl;
        return false;
    }
    
    return true;
}

MTL::Buffer* MetalMatrixVec::createBuffer(const void* data, size_t size) {
    MTL::Buffer* buffer = _device->newBuffer(size, MTL::ResourceStorageModeShared);
    if (data) {
        memcpy(buffer->contents(), data, size);
    }
    return buffer;
}

std::vector<float> MetalMatrixVec::multiply(
    const std::vector<std::vector<float>>& matrix, 
    const std::vector<float>& vec) {
    
    if (matrix.empty() || vec.empty() || matrix[0].size() != vec.size()) {
        throw std::invalid_argument("Invalid matrix or vector dimensions");
    }
    
    size_t numRows = matrix.size();
    size_t numCols = vec.size();
    
    // Flatten the matrix for Metal (row-major order)
    std::vector<float> flatMatrix;
    flatMatrix.reserve(numRows * numCols);
    for (const auto& row : matrix) {
        flatMatrix.insert(flatMatrix.end(), row.begin(), row.end());
    }
    
    // Create output vector
    std::vector<float> result(numRows, 0.0f);
    
    // Create Metal buffers
    MTL::Buffer* matrixBuffer = createBuffer(flatMatrix.data(), flatMatrix.size() * sizeof(float));
    MTL::Buffer* vectorBuffer = createBuffer(vec.data(), vec.size() * sizeof(float));
    MTL::Buffer* resultBuffer = createBuffer(nullptr, result.size() * sizeof(float));
    
    // Create a buffer for numCols (as constant uint)
    uint32_t numColsValue = static_cast<uint32_t>(numCols);
    MTL::Buffer* numColsBuffer = createBuffer(&numColsValue, sizeof(uint32_t));
    
    // Create command buffer and compute encoder
    MTL::CommandBuffer* commandBuffer = _commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    
    // Configure the compute encoder
    computeEncoder->setComputePipelineState(_pipelineState);
    computeEncoder->setBuffer(matrixBuffer, 0, 0);
    computeEncoder->setBuffer(vectorBuffer, 0, 1);
    computeEncoder->setBuffer(resultBuffer, 0, 2);
    computeEncoder->setBuffer(numColsBuffer, 0, 3);
    
    // Calculate grid size and launch the kernel
    MTL::Size gridSize = MTL::Size(numRows, 1, 1);
    
    // Calculate threadgroup size
    NS::UInteger threadGroupSize = _pipelineState->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > numRows) {
        threadGroupSize = numRows;
    }
    MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);
    
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();
    
    // Execute and wait
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // Copy result back
    memcpy(result.data(), resultBuffer->contents(), result.size() * sizeof(float));
    
    // Clean up
    matrixBuffer->release();
    vectorBuffer->release();
    resultBuffer->release();
    numColsBuffer->release();
    
    return result;
}
