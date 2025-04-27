#include "../include/RandomHistogramSortedPoll.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <string>

// Define global output file streams
std::ofstream histogram_output_file;
std::ofstream sorted_poll_output_file;

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <num_states> <num_events> <num_agents> <num_trials> [num_threads]" << std::endl;
        std::cout << "If num_threads is not specified, all available cores will be used." << std::endl;
        std::cout << "Note: GPU acceleration will be used for problems with states <= 16 and events <= 16." << std::endl;
        return 1;
    }
    
    int num_states = std::stoi(argv[1]);
    int num_events = std::stoi(argv[2]);
    int num_agents = std::stoi(argv[3]);
    int num_trials = std::stoi(argv[4]);
    
    // Generate output filenames based on parameters
    std::string histogram_filename = "best_histogram_"+std::to_string(num_states)+"_"+std::to_string(num_events)+".txt";
    std::string sorted_poll_filename = "best_sorted_poll_"+std::to_string(num_states)+"_"+std::to_string(num_events)+".txt";
    
    // Open the output files
    histogram_output_file.open(histogram_filename);
    sorted_poll_output_file.open(sorted_poll_filename);
    
    if (!histogram_output_file.is_open() || !sorted_poll_output_file.is_open()) {
        std::cerr << "Error: Could not open output files." << std::endl;
        return 1;
    }
    
    std::cout << "Randomized Histogram- and Sorted-Poll trials (Metal GPU/Multithreaded, C++ version)" << std::endl;
    std::cout << "Number of States = " << num_states << std::endl;
    std::cout << "Number of Events = " << num_events << std::endl;
    std::cout << "Number of Agents = " << num_agents << std::endl;
    std::cout << "Number of Trials = " << num_trials << std::endl;
    
    // Use specified number of threads or all available cores
    int num_threads = (argc > 5) ? std::stoi(argv[5]) : std::thread::hardware_concurrency();
    std::cout << "Using " << num_threads << " CPU threads if needed" << std::endl;
    
    // Create and run the simulation
    try {
        RandomHistogramSortedPoll simulator(num_states, num_events, num_agents, num_trials, num_threads);
        simulator.run();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        // Close output files before exiting
        histogram_output_file.close();
        sorted_poll_output_file.close();
        return 1;
    }
    
    // Close output files
    histogram_output_file.close();
    sorted_poll_output_file.close();
    
    std::cout << "Results written to " << histogram_filename << " and " << sorted_poll_filename << std::endl;
    
    return 0;
}
