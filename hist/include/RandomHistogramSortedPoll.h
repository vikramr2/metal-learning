#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <vector>
#include <atomic>
#include <mutex>
#include <functional>

// Constants
const int MAX_STATES = 16;
const int NIL_NODE = -1;
const int NIL_TRANSITION = -1;
const int MAX_NODE_TABLE_ENTRIES = 1000000;

const int INTERNAL = 1;
const int DUPLICATE = 2;

// Node table entry structure
struct NodeTableEntry {
    int from_node;
    int event;
    int node_type;
    int dup_node;
    int sorted_poll_dup_node;
    int sorted_poll_node_type;
    int histogram_state[MAX_STATES];
    int sorted_poll[MAX_STATES];
};

// Structure to hold trial results
struct TrialResult {
    int histogram_gap;
    int sorted_poll_gap;
    std::vector<std::vector<int>> histogram_transition_matrix;
    std::vector<int> histogram_initial_state;
    std::vector<std::vector<int>> sorted_poll_transition_matrix;
    std::vector<int> sorted_poll_initial_state;
};

// Structures for Metal (GPU) computation
struct MetalTrialParams {
    int num_states;
    int num_events;
    int num_agents;
    unsigned int seed;
};

struct MetalTrialResult {
    int histogram_gap;
    int sorted_poll_gap;
    int histogram_transition_matrix[MAX_STATES * MAX_STATES];  // Flattened matrix
    int histogram_initial_state[MAX_STATES];
    int sorted_poll_transition_matrix[MAX_STATES * MAX_STATES]; // Flattened matrix
    int sorted_poll_initial_state[MAX_STATES];
};

class RandomHistogramSortedPoll {
private:
    // Configuration parameters
    int num_states, num_events, num_agents, num_trials;
    int num_threads;
    
    // Metal objects
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::Library* library;
    MTL::Function* trialFunction;
    MTL::ComputePipelineState* pipelineState;
    
    // Synchronization
    std::mutex best_result_mutex;
    std::atomic<bool> found_negative_gap;
    std::atomic<bool> found_zero_gap_for_both;
    
    // Methods
    TrialResult runSingleTrialCPU(unsigned int seed);
    std::vector<MetalTrialResult> runTrialsGPU(int numTrials, unsigned int base_seed);
    int bellNumber(int n);
    void initializeMetalResources();
    
public:
    RandomHistogramSortedPoll(int states, int events, int agents, int trials, int threads);
    ~RandomHistogramSortedPoll();
    void run();
};
