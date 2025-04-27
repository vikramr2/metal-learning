#include "../include/RandomHistogramSortedPoll.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <thread>
#include <fstream>
#include <algorithm>

// External file streams declared in main.cpp
extern std::ofstream histogram_output_file;
extern std::ofstream sorted_poll_output_file;

// Helper function to create Metal device
MTL::Device* createMetalDevice() {
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to create Metal device" << std::endl;
        exit(1);
    }
    return device;
}

RandomHistogramSortedPoll::RandomHistogramSortedPoll(int states, int events, int agents, int trials, int threads)
    : num_states(states), num_events(events), num_agents(agents), num_trials(trials),
      num_threads(threads), found_negative_gap(false), found_zero_gap_for_both(false) {
    
    // Initialize Metal resources
    device = createMetalDevice();
    commandQueue = device->newCommandQueue();
    
    // Create Metal library and function
    initializeMetalResources();
    
    std::cout << "Using Metal GPU acceleration for trials" << std::endl;
}

RandomHistogramSortedPoll::~RandomHistogramSortedPoll() {
    // Clean up Metal resources
    pipelineState->release();
    trialFunction->release();
    library->release();
    commandQueue->release();
    device->release();
}

void RandomHistogramSortedPoll::initializeMetalResources() {
    // Try to load the Metal shader library
    NS::Error* error = nullptr;
    
    // First try to load the metallib file directly from the current directory
    NS::String* libraryPath = NS::String::string("./shader.metallib", 
                                              NS::StringEncoding::UTF8StringEncoding);
    
    library = device->newLibrary(libraryPath, &error);
    
    // If that fails, try loading from the bin directory
    if (!library) {
        libraryPath = NS::String::string("./bin/shader.metallib", 
                                      NS::StringEncoding::UTF8StringEncoding);
        library = device->newLibrary(libraryPath, &error);
    }
    
    // If both fail, try the build directory
    if (!library) {
        libraryPath = NS::String::string("./build/shader.metallib", 
                                     NS::StringEncoding::UTF8StringEncoding);
        library = device->newLibrary(libraryPath, &error);
    }
    
    // If all the above fail, report error and exit
    if (!library) {
        std::cerr << "Failed to load Metal library. Make sure shader.metallib is in the current, bin, or build directory" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        exit(1);
    }
    
    // Get the kernel function
    trialFunction = library->newFunction(NS::String::string("runTrial", 
                                                        NS::StringEncoding::UTF8StringEncoding));
    if (!trialFunction) {
        std::cerr << "Failed to find the kernel function 'runTrial'" << std::endl;
        exit(1);
    }
    
    // Create compute pipeline
    pipelineState = device->newComputePipelineState(trialFunction, &error);
    if (!pipelineState) {
        std::cerr << "Failed to create pipeline state: ";
        if (error) {
            std::cerr << error->localizedDescription()->utf8String();
        }
        std::cerr << std::endl;
        exit(1);
    }
}

std::vector<MetalTrialResult> RandomHistogramSortedPoll::runTrialsGPU(int numTrials, unsigned int base_seed) {
    // Create buffers for parameters and results
    MetalTrialParams params;
    params.num_states = num_states;
    params.num_events = num_events;
    params.num_agents = num_agents;
    params.seed = base_seed;
    
    MTL::Buffer* paramsBuffer = device->newBuffer(&params, sizeof(MetalTrialParams), MTL::ResourceStorageModeShared);
    MTL::Buffer* resultsBuffer = device->newBuffer(numTrials * sizeof(MetalTrialResult), MTL::ResourceStorageModeShared);
    
    // Create command buffer and encoder
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    
    // Set pipeline state and buffers
    computeEncoder->setComputePipelineState(pipelineState);
    computeEncoder->setBuffer(paramsBuffer, 0, 0);
    computeEncoder->setBuffer(resultsBuffer, 0, 1);
    
    // Calculate grid size
    NS::UInteger grid_size = numTrials;
    NS::UInteger threadgroup_size = pipelineState->maxTotalThreadsPerThreadgroup();
    if (threadgroup_size > grid_size) {
        threadgroup_size = grid_size;
    }
    
    // Dispatch threads
    computeEncoder->dispatchThreads(MTL::Size(grid_size, 1, 1), MTL::Size(threadgroup_size, 1, 1));
    
    // End encoding and submit command buffer
    computeEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // Copy results from GPU
    MetalTrialResult* results = static_cast<MetalTrialResult*>(resultsBuffer->contents());
    std::vector<MetalTrialResult> resultVector(results, results + numTrials);
    
    // Release Metal resources
    paramsBuffer->release();
    resultsBuffer->release();
    
    return resultVector;
}

TrialResult RandomHistogramSortedPoll::runSingleTrialCPU(unsigned int seed) {
    // Initialize random number generator with given seed
    std::srand(seed);
    
    std::vector<NodeTableEntry> node_table(MAX_NODE_TABLE_ENTRIES);
    std::vector<int> initial_state(num_states);
    int next_node_table_entry = 0;
    
    // Initialize transition matrices
    std::vector<std::vector<int>> transition_matrix(num_states, std::vector<int>(num_events));
    TrialResult result;
    
    // Initialize result structures
    result.histogram_transition_matrix.resize(num_states, std::vector<int>(num_events));
    result.sorted_poll_transition_matrix.resize(num_states, std::vector<int>(num_events));
    result.histogram_initial_state.resize(num_states);
    result.sorted_poll_initial_state.resize(num_states);
    result.histogram_gap = INT_MAX;
    result.sorted_poll_gap = INT_MAX;
    
    // Generate random transitions
    for (int i = 0; i < num_states; i++) {
        for (int j = 0; j < num_events; j++) {
            transition_matrix[i][j] = rand() % num_states;
        }
    }
    
    // Generate random initial state
    for (int i = 0; i < num_states; i++) {
        initial_state[i] = 0;
    }
    
    for (int i = 0; i < num_agents; i++) {
        initial_state[rand() % num_states]++;
    }
    
    // Local function to check for duplicates
    auto dup_check = [&node_table, this, &next_node_table_entry](const std::vector<int>& current_state) -> int {
        for (int i = 0; i < next_node_table_entry; i++) {
            if (node_table[i].node_type != DUPLICATE) {
                bool same = true;
                int j = 0;
                while ((j < num_states) && (same)) {
                    same = (node_table[i].histogram_state[j] == current_state[j]);
                    j += 1;
                }
                if (same)
                    return i;
            }
        }
        return NIL_NODE;
    };
    
    // Local function to check for sorted poll duplicates
    auto sorted_poll_dup_check = [&node_table, this, &next_node_table_entry](const int* current_sorted_poll_state) -> int {
        for (int i = 0; i < next_node_table_entry; i++) {
            if (node_table[i].node_type != DUPLICATE) {
                bool same = true;
                int j = 0;
                while ((j < num_states) && (same)) {
                    same = (node_table[i].sorted_poll[j] == current_sorted_poll_state[j]);
                    j += 1;
                }
                if (same)
                    return i;
            }
        }
        return NIL_NODE;
    };
    
    // Local function to fire an event
    auto fire_event = [&transition_matrix, this](int event, const std::vector<int>& current_state, std::vector<int>& new_state) {
        for (int i = 0; i < num_states; i++) {
            new_state[i] = 0;
        }
        
        for (int i = 0; i < num_states; i++) {
            new_state[transition_matrix[i][event]] += current_state[i];
        }
    };
    
    // Function to process new state (recursive)
    std::function<void(const std::vector<int>&, int, int)> process_new_state;
    process_new_state = [&](const std::vector<int>& current_state, int event, int from_node) {
        std::vector<int> new_state(MAX_STATES);
        
        if (dup_check(current_state) == NIL_NODE) {
            // This state/node is unique
            int new_from = next_node_table_entry;
            node_table[new_from].from_node = from_node;
            node_table[new_from].event = event;
            node_table[new_from].node_type = INTERNAL;
            
            for (int i = 0; i < num_states; i++) {
                node_table[new_from].histogram_state[i] = current_state[i];
                node_table[new_from].sorted_poll[i] = current_state[i];
            }
            
            // Sort the poll
            std::sort(node_table[new_from].sorted_poll, node_table[new_from].sorted_poll + num_states);
            
            // Check if the sorted_poll is unique as well
            if ((node_table[new_from].sorted_poll_dup_node = sorted_poll_dup_check(node_table[new_from].sorted_poll)) == NIL_NODE) {
                node_table[new_from].sorted_poll_node_type = INTERNAL;
                node_table[new_from].sorted_poll_dup_node = new_from;
            } else {
                node_table[new_from].sorted_poll_node_type = DUPLICATE;
            }
            
            if ((next_node_table_entry += 1) >= MAX_NODE_TABLE_ENTRIES) {
                std::cout << "\n Random Histogram: more than " << MAX_NODE_TABLE_ENTRIES << " node table entries generated\n\n";
                return;
            }
            
            // Now check all events
            for (int new_event = 0; new_event < num_events; new_event++) {
                fire_event(new_event, current_state, new_state);
                // Recursively process the new marking
                process_new_state(new_state, new_event, new_from);
            }
        }
    };
    
    // Process the initial state
    process_new_state(initial_state, NIL_TRANSITION, NIL_NODE);
    
    // Count node types
    int internal_count = 0, duplicate_count = 0, sorted_internal_count = 0;
    for (int i = 1; i < next_node_table_entry; i++) {
        switch (node_table[i].node_type) {
            case INTERNAL:
                internal_count += 1;
                break;
            case DUPLICATE:
                duplicate_count += 1;
                break;
        }
        
        switch (node_table[i].sorted_poll_node_type) {
            case INTERNAL:
                sorted_internal_count += 1;
                break;
        }
    }
    
    // Calculate gaps
    int histogram_gap = static_cast<int>(pow(num_states, num_states)) - (internal_count + 1);
    int sorted_poll_gap = bellNumber(num_states) - (sorted_internal_count + 1);
    if ((histogram_gap == 0) && (sorted_poll_gap == 0))
        found_zero_gap_for_both = true;
    
    // Check for negative gaps (error condition)
    if (histogram_gap < 0 || sorted_poll_gap < 0) {
        found_negative_gap = true;
        
        std::lock_guard<std::mutex> lock(best_result_mutex);
        if (histogram_gap < 0) {
            std::cout << "Thread found histogram gap negative: " << histogram_gap << "\n";
            for (int i = 0; i < num_states; i++) {
                for (int j = 0; j < num_events; j++) {
                    std::cout << i << " " << j << " " << transition_matrix[i][j] << std::endl;
                }
            }
            for (int i = 0; i < num_states; i++) {
                std::cout << initial_state[i] << " ";
            }
            std::cout << std::endl;
        }
        
        if (sorted_poll_gap < 0) {
            std::cout << "Thread found sorted poll gap negative: " << sorted_poll_gap << "\n";
            std::cout << "Bell number(" << num_states << ") = " << bellNumber(num_states)
                      << "\tSorted-internal-count = " << sorted_internal_count + 1 << std::endl;
            for (int i = 0; i < num_states; i++) {
                for (int j = 0; j < num_events; j++) {
                    std::cout << i << " " << j << " " << transition_matrix[i][j] << std::endl;
                }
            }
            for (int i = 0; i < num_states; i++) {
                std::cout << initial_state[i] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // Store results
    result.histogram_gap = histogram_gap;
    result.sorted_poll_gap = sorted_poll_gap;
    
    for (int i = 0; i < num_states; i++) {
        result.histogram_initial_state[i] = initial_state[i];
        result.sorted_poll_initial_state[i] = initial_state[i];
        for (int j = 0; j < num_events; j++) {
            result.histogram_transition_matrix[i][j] = transition_matrix[i][j];
            result.sorted_poll_transition_matrix[i][j] = transition_matrix[i][j];
        }
    }
    
    return result;
}

int RandomHistogramSortedPoll::bellNumber(int n) {
    if (n == 0 || n == 1) {
        return 1;
    }
    
    std::vector<std::vector<int>> bellTriangle(n + 1);
    for (int i = 0; i <= n; i++) {
        bellTriangle[i].resize(i + 1);
    }
    
    bellTriangle[0][0] = 1;
    
    for (int i = 1; i <= n; i++) {
        bellTriangle[i][0] = bellTriangle[i - 1][i - 1];
        for (int j = 1; j <= i; j++) {
            bellTriangle[i][j] = bellTriangle[i - 1][j - 1] + bellTriangle[i][j - 1];
        }
    }
    
    return bellTriangle[n][0];
}

void RandomHistogramSortedPoll::run() {
    // Best results across all trials
    TrialResult best_result;
    best_result.histogram_gap = INT_MAX;
    best_result.sorted_poll_gap = INT_MAX;
    best_result.histogram_transition_matrix.resize(num_states, std::vector<int>(num_events));
    best_result.sorted_poll_transition_matrix.resize(num_states, std::vector<int>(num_events));
    best_result.histogram_initial_state.resize(num_states);
    best_result.sorted_poll_initial_state.resize(num_states);
    
    // Progress tracking
    std::atomic<int> trials_completed(0);
    
    // Use Metal GPU when possible
    if (num_states <= MAX_STATES && num_states > 0 && num_events <= MAX_STATES && num_events > 0) {
        // Determine number of GPU batches
        const int max_trials_per_batch = 10000; // Adjust based on GPU memory
        int num_batches = (num_trials + max_trials_per_batch - 1) / max_trials_per_batch;
        
        unsigned int base_seed = static_cast<unsigned int>(time(NULL));
        int trials_remaining = num_trials;
        
        std::cout << "Running " << num_trials << " trials in " << num_batches << " GPU batches" << std::endl;
        
        for (int batch = 0; batch < num_batches && !found_negative_gap && !found_zero_gap_for_both; batch++) {
            int trials_this_batch = std::min(max_trials_per_batch, trials_remaining);
            
            // Run batch on GPU
            std::vector<MetalTrialResult> batch_results = runTrialsGPU(trials_this_batch, base_seed + batch * max_trials_per_batch);
            
            // Process batch results
            for (const auto& result : batch_results) {
                // Update best results
                std::lock_guard<std::mutex> lock(best_result_mutex);
                
                if (result.histogram_gap < best_result.histogram_gap) {
                    best_result.histogram_gap = result.histogram_gap;
                    
                    // Copy initial state
                    for (int i = 0; i < num_states; i++) {
                        best_result.histogram_initial_state[i] = result.histogram_initial_state[i];
                    }
                    
                    // Copy transition matrix
                    for (int i = 0; i < num_states; i++) {
                        for (int j = 0; j < num_events; j++) {
                            best_result.histogram_transition_matrix[i][j] = result.histogram_transition_matrix[i * num_states + j];
                        }
                    }
                }
                
                if (result.sorted_poll_gap < best_result.sorted_poll_gap) {
                    best_result.sorted_poll_gap = result.sorted_poll_gap;
                    
                    // Copy initial state
                    for (int i = 0; i < num_states; i++) {
                        best_result.sorted_poll_initial_state[i] = result.sorted_poll_initial_state[i];
                    }
                    
                    // Copy transition matrix
                    for (int i = 0; i < num_states; i++) {
                        for (int j = 0; j < num_events; j++) {
                            best_result.sorted_poll_transition_matrix[i][j] = result.sorted_poll_transition_matrix[i * num_states + j];
                        }
                    }
                }
                
                // Check for special conditions
                if (result.histogram_gap < 0 || result.sorted_poll_gap < 0) {
                    found_negative_gap = true;
                }
                
                if (result.histogram_gap == 0 && result.sorted_poll_gap == 0) {
                    found_zero_gap_for_both = true;
                }
                
                trials_completed++;
            }
            
            // Update progress
            std::cout << "Completed " << trials_completed << " of " << num_trials
                      << " trials (" << (trials_completed * 100 / num_trials) << "%)" << std::endl;
            
            trials_remaining -= trials_this_batch;
        }
    } else {
        // Fallback to CPU implementation for larger problems
        std::cout << "Problem size too large for GPU, falling back to CPU implementation" << std::endl;
        
        // Function to handle a batch of trials on CPU
        auto processBatch = [this, &best_result, &trials_completed](int start_trial, int end_trial, unsigned int base_seed) {
            for (int trial = start_trial; trial < end_trial && !found_negative_gap && !found_zero_gap_for_both; trial++) {
                // Use a different seed for each trial
                unsigned int trial_seed = base_seed + trial;
                TrialResult result = runSingleTrialCPU(trial_seed);
                
                // Update the best results if better
                {
                    std::lock_guard<std::mutex> lock(best_result_mutex);
                    
                    if (result.histogram_gap < best_result.histogram_gap) {
                        best_result.histogram_gap = result.histogram_gap;
                        best_result.histogram_initial_state = result.histogram_initial_state;
                        best_result.histogram_transition_matrix = result.histogram_transition_matrix;
                    }
                    
                    if (result.sorted_poll_gap < best_result.sorted_poll_gap) {
                        best_result.sorted_poll_gap = result.sorted_poll_gap;
                        best_result.sorted_poll_initial_state = result.sorted_poll_initial_state;
                        best_result.sorted_poll_transition_matrix = result.sorted_poll_transition_matrix;
                    }
                    
                    // Update progress counter
                    trials_completed++;
                    if (trials_completed % 100 == 0) {
                        std::cout << "Completed " << trials_completed << " of " << num_trials
                                  << " trials (" << (trials_completed * 100 / num_trials) << "%)" << std::endl;
                    }
                }
            }
        };
        
        // Calculate batch size
        int base_trials_per_thread = num_trials / num_threads;
        int remaining_trials = num_trials % num_threads;
        
        // Create and start threads
        std::vector<std::thread> threads;
        unsigned int base_seed = static_cast<unsigned int>(time(NULL));
        
        int start_trial = 0;
        for (int t = 0; t < num_threads; t++) {
            // Calculate the range of trials for this thread
            int trials_for_thread = base_trials_per_thread + (t < remaining_trials ? 1 : 0);
            int end_trial = start_trial + trials_for_thread;
            
            // Start thread
            threads.emplace_back(processBatch, start_trial, end_trial, base_seed);
            
            start_trial = end_trial;
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    // Output the final results
    std::cout << "\nBest Histogram Gap = " << best_result.histogram_gap
              << " (#Histogram states = " << ((int)pow(num_states, num_states)) - best_result.histogram_gap << ")" << std::endl;
    histogram_output_file << "\nBest Histogram Gap = " << best_result.histogram_gap
              << " (#Histogram states = " << ((int)pow(num_states, num_states)) - best_result.histogram_gap << ")" << std::endl;
    for (int i = 0; i < num_states; i++) {
        for (int j = 0; j < num_events; j++) {
            histogram_output_file  << i << " " << j << " " << best_result.histogram_transition_matrix[i][j] << std::endl;
        }
    }
    for (int i = 0; i < num_states; i++) {
        histogram_output_file  << best_result.histogram_initial_state[i] << " ";
    }
    histogram_output_file << std::endl;
    
    std::cout << "\nBest Sorted Poll Gap = " << best_result.sorted_poll_gap
              << " (#Sorted Poll States = " << bellNumber(num_states) - best_result.sorted_poll_gap << ")" << std::endl;
    sorted_poll_output_file << "\nBest Sorted Poll Gap = " << best_result.sorted_poll_gap
              << " (#Sorted Poll States = " << bellNumber(num_states) - best_result.sorted_poll_gap << ")" << std::endl;
    for (int i = 0; i < num_states; i++) {
        for (int j = 0; j < num_events; j++) {
            sorted_poll_output_file << i << " " << j << " " << best_result.sorted_poll_transition_matrix[i][j] << std::endl;
        }
    }
    for (int i = 0; i < num_states; i++) {
        sorted_poll_output_file << best_result.sorted_poll_initial_state[i] << " ";
    }
    sorted_poll_output_file << std::endl;
    std::cout << std::endl;
}
