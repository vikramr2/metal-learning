#include <metal_stdlib>
using namespace metal;

kernel void matrix_vector_multiply(
    device const float* matrix [[buffer(0)]],
    device const float* vector [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    constant uint& num_columns [[buffer(3)]])
{
    // Each thread computes one element of the result vector
    // id represents the row in the matrix
    float sum = 0.0;
    for (uint j = 0; j < num_columns; j++) {
        sum += matrix[id * num_columns + j] * vector[j];
    }
    result[id] = sum;
}
