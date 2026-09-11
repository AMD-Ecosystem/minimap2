# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# GNUmakefile - CMake wrapper for minimap2
# 
# Usage:
#   make                                    # Default AMD GPU build
#   make GPU=AMD DEBUG=analyze              # AMD GPU with debug analyze mode
#   make GPU=AMD GPUARCH=gfx942 DEBUG=analyze  # Specific architecture
#   make GPU=NONE                           # CPU-only build
#   make clean                              # Clean build directory
#   make distclean                          # Remove all build artifacts
#
# Variables:
#   GPU      - GPU type: AMD, NONE (default: AMD)
#   GPUARCH  - GPU architecture: gfx942, gfx1030, etc. (optional, auto-detect if not set)
#   DEBUG    - Debug mode: info, analyze, verbose (optional)
#   BUILD_TYPE - CMake build type: Debug, Release, RelWithDebInfo (default: RelWithDebInfo)
#   JOBS     - Number of parallel jobs (default: auto)

# Default values
GPU ?= AMD
BUILD_TYPE ?= RelWithDebInfo
BUILD_DIR := build

# Determine number of jobs
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# CMake configuration options
CMAKE_OPTS := -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
CMAKE_OPTS += -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# GPU configuration
ifneq ($(GPU),)
    CMAKE_OPTS += -DGPU=$(GPU)
endif

# GPU architecture (optional)
ifneq ($(GPUARCH),)
    CMAKE_OPTS += -DGPUARCH=$(GPUARCH)
endif

# Debug mode (optional)
ifneq ($(DEBUG),)
    CMAKE_OPTS += -DDEBUG_MODE=$(DEBUG)
endif

# Additional options
ifdef BUILD_TESTING
    CMAKE_OPTS += -DBUILD_TESTING=$(BUILD_TESTING)
endif

ifdef ENABLE_COVERAGE
    CMAKE_OPTS += -DENABLE_COVERAGE=$(ENABLE_COVERAGE)
endif

ifdef ENABLE_ASAN
    CMAKE_OPTS += -DENABLE_ASAN=$(ENABLE_ASAN)
endif

ifdef ENABLE_TSAN
    CMAKE_OPTS += -DENABLE_TSAN=$(ENABLE_TSAN)
endif

ifdef SSE2_ONLY
    CMAKE_OPTS += -DSSE2_ONLY=$(SSE2_ONLY)
endif

ifdef ARM_NEON
    CMAKE_OPTS += -DARM_NEON=$(ARM_NEON)
endif

ifdef BUILD_EXTRA
    CMAKE_OPTS += -DBUILD_EXTRA=$(BUILD_EXTRA)
endif

# Phony targets
.PHONY: all configure build clean distclean install test help

# Default target
all: build

# Configure CMake
configure: $(BUILD_DIR)/Makefile

$(BUILD_DIR)/Makefile: CMakeLists.txt
	@echo "Configuring with: cmake -S . -B $(BUILD_DIR) $(CMAKE_OPTS)"
	@mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) $(CMAKE_OPTS)

# Build target - always reconfigure to pick up variable changes
build:
	@echo "Configuring with: cmake -S . -B $(BUILD_DIR) $(CMAKE_OPTS)"
	@mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) $(CMAKE_OPTS)
	@echo "Building with $(JOBS) parallel jobs..."
	cmake --build $(BUILD_DIR) -j $(JOBS)

# Install target
install: build
	cmake --install $(BUILD_DIR)

# Run tests
test: build
	cd $(BUILD_DIR) && ctest --output-on-failure

# Clean build artifacts
clean:
	@if [ -d "$(BUILD_DIR)" ]; then \
		cmake --build $(BUILD_DIR) --target clean 2>/dev/null || true; \
	fi

# Remove build directory entirely
distclean:
	rm -rf $(BUILD_DIR)
	rm -rf out

# Help target
help:
	@echo "minimap2 CMake Build System"
	@echo ""
	@echo "Usage: make [TARGET] [OPTIONS]"
	@echo ""
	@echo "Targets:"
	@echo "  all (default)  - Configure and build"
	@echo "  configure      - Configure CMake only"
	@echo "  build          - Build the project"
	@echo "  install        - Install to prefix"
	@echo "  test           - Run tests"
	@echo "  clean          - Clean build artifacts"
	@echo "  distclean      - Remove build directory"
	@echo "  help           - Show this help"
	@echo ""
	@echo "Options:"
	@echo "  GPU=AMD|NONE           - GPU type (default: AMD)"
	@echo "  GPUARCH=<arch>         - GPU architecture (e.g., gfx942, gfx1030)"
	@echo "  DEBUG=info|analyze|verbose - Debug mode"
	@echo "  BUILD_TYPE=<type>      - CMake build type (default: RelWithDebInfo)"
	@echo "  SSE2_ONLY=ON           - Build x86 SSE kernels for SSE2 only (no SSE4.1)"
	@echo "  ARM_NEON=ON            - Build SSE kernels for ARM via sse2neon (auto-detected)"
	@echo "  JOBS=<n>               - Parallel jobs (default: auto)"
	@echo ""
	@echo "Examples:"
	@echo "  make                              # Default AMD GPU build"
	@echo "  make GPU=AMD DEBUG=analyze        # AMD with debug analyze"
	@echo "  make GPU=AMD GPUARCH=gfx942 DEBUG=analyze"
	@echo "  make GPU=NONE                     # CPU-only build"
	@echo "  make BUILD_TYPE=Release           # Release build"
	@echo "  make distclean && make            # Clean rebuild"
