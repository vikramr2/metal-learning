CXX = clang++
CXXFLAGS = -std=c++17 -stdlib=libc++ -O2 -I./metal-cpp
FRAMEWORKS = -framework Metal -framework Foundation -framework MetalKit
METAL_COMPILER = xcrun -sdk macosx metal
METAL_LINKER = xcrun -sdk macosx metallib

# Paths
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Files
SOURCES = $(SRC_DIR)/MetalCppImpl.cpp $(SRC_DIR)/main.cpp $(SRC_DIR)/MetalMatrixVec.cpp
HEADERS = $(INCLUDE_DIR)/MetalMatrixVec.h
METAL_SRC = $(SRC_DIR)/shader.metal
METAL_LIB = $(BUILD_DIR)/shader.metallib
EXECUTABLE = $(BIN_DIR)/matrix_vector_multiply

# Default target
all: directories $(EXECUTABLE)

# Create build directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Compile Metal shader to metallib
$(METAL_LIB): $(METAL_SRC)
	$(METAL_COMPILER) -c -o $(BUILD_DIR)/shader.air $(METAL_SRC)
	$(METAL_LINKER) -o $@ $(BUILD_DIR)/shader.air

# Build executable
$(EXECUTABLE): $(SOURCES) $(HEADERS) $(METAL_LIB)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -fno-objc-arc $(SOURCES) -o $@

# Copy metallib to binary directory
	@cp $(METAL_LIB) $(BIN_DIR)/

# Clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all clean directories
