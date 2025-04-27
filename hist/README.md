# Histogram and Sorted Poll Simulation

This project implements a Metal-accelerated simulation for calculating histogram and sorted poll gaps in state transition systems. It uses Apple's Metal framework to perform GPU-accelerated simulations when possible, with a CPU fallback for larger problem sizes.

## Project Structure

```
.
├── CMakeLists.txt           # Build configuration
├── include/
│   └── RandomHistogramSortedPoll.h    # Class declaration
├── src/
│   ├── RandomHistogramSortedPoll.cpp  # Class implementation
│   ├── HistogramMain.cpp              # Main entry point for histogram app
│   ├── MetalCppImpl.cpp               # Metal implementation details
│   ├── main.cpp                       # Matrix-vector multiply demo
│   └── MetalMatrixVec.cpp             # Matrix-vector multiply implementation
└── shader.metal              # Metal shader code for GPU acceleration
```

## Requirements

- macOS with Apple Silicon or Intel Mac with Metal support
- Xcode Command Line Tools
- CMake 3.20 or later

## Building the Project

```bash
make
```

# Run the histogram simulation
```
./hist <num_states> <num_events> <num_agents> <num_trials> [num_threads]
```

## Usage

The histogram simulation accepts the following parameters:

- `num_states`: Number of states in the transition system
- `num_events`: Number of events/transitions
- `num_agents`: Number of agents
- `num_trials`: Number of random trials to perform
- `num_threads` (optional): Number of CPU threads to use if fallback is needed. Defaults to all available cores.

Example:

```bash
./hist 4 3 10 1000
```

This will run 1000 trials with a system having 4 states, 3 events, and 10 agents.

## Performance Notes

- The system automatically uses GPU acceleration for problems where the number of states and events are both ≤ 16.
- For larger problems, it falls back to a multithreaded CPU implementation.
- Results are written to text files named according to the parameters:
  - `best_histogram_<num_states>_<num_events>.txt`
  - `best_sorted_poll_<num_states>_<num_events>.txt`
