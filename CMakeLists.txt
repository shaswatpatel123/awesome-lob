cmake_minimum_required(VERSION 3.18)
project(CudaOrderbook CUDA CXX)

# CUDA and C++ standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
# -O3: Maximum optimization
# --use_fast_math: Faster math operations (good for A100)
# -Xptxas -v: Show register usage and occupancy info
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# CUDA architectures - Auto-detect GPU or use multiple architectures
# If CMAKE_CUDA_ARCHITECTURES is not set, auto-detect from available GPU
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "Auto-detecting GPU architecture...")
    
    # Try to detect using our detection script
    find_program(BASH_EXECUTABLE bash)
    if(BASH_EXECUTABLE)
        set(DETECT_SCRIPT "${CMAKE_SOURCE_DIR}/detect_gpu_arch.sh")
        if(EXISTS "${DETECT_SCRIPT}")
            execute_process(
                COMMAND ${BASH_EXECUTABLE} "${DETECT_SCRIPT}"
                OUTPUT_VARIABLE DETECTED_ARCH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(DETECTED_ARCH AND DETECTED_ARCH MATCHES "^[0-9]+$")
                message(STATUS "Detected GPU architecture: SM ${DETECTED_ARCH}")
                set(CMAKE_CUDA_ARCHITECTURES "${DETECTED_ARCH}")
            else
                message(STATUS "Auto-detection failed, using multiple architectures for compatibility")
                set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89")
            endif()
        else
            message(STATUS "Detection script not found, using multiple architectures for compatibility")
            set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89")
        endif()
    else
        message(STATUS "Bash not found, using multiple architectures for compatibility")
        set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89")
    endif()
else()
    message(STATUS "Using manually specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================================================
# Core Library
# ============================================================================

# Source files for the orderbook library
set(CUDA_ORDERBOOK_SOURCES
    # Team 1 & 2 implemented:
    src/operations.cu    # Device functions: add, cancel, matching
    src/kernels.cu       # CUDA kernels: all operations
    src/utils.cu         # Utility functions: memory management, init
    # src/queries.cu       # Team 3: Days 2-4 (TODO)
    # src/orderbook.cu     # Team 3: Days 5-8 (TODO)
)

# Create static library with CUDA sources
add_library(cuda_orderbook STATIC ${CUDA_ORDERBOOK_SOURCES})

# ENABLE CUDA SEPARATE COMPILATION (ADD THESE LINES)
set_target_properties(cuda_orderbook PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Include directories
target_include_directories(cuda_orderbook PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ============================================================================
# Examples (Team 3: Days 7-8)
# ============================================================================

# Uncomment when examples are implemented
# add_executable(single_example examples/single_orderbook.cpp)
# target_link_libraries(single_example cuda_orderbook)

# add_executable(batch_example examples/batch_orderbooks.cpp)
# target_link_libraries(batch_example cuda_orderbook)

# ============================================================================
# Tests (Team 3: Days 7-10)
# ============================================================================

# Option to build tests
option(BUILD_TESTS "Build test executables" ON)

if(BUILD_TESTS)
    # Uncomment when tests are implemented
    # add_executable(test_operations tests/test_operations.cpp)
    # target_link_libraries(test_operations cuda_orderbook)
    
    # add_executable(test_batch tests/test_batch.cpp)
    # target_link_libraries(test_batch cuda_orderbook)
    
    # If using Google Test (optional)
    # find_package(GTest)
    # if(GTest_FOUND)
    #     add_executable(test_orderbook_gtest tests/test_gtest.cpp)
    #     target_link_libraries(test_orderbook_gtest cuda_orderbook GTest::GTest GTest::Main)
    #     enable_testing()
    #     add_test(NAME OrderbookTests COMMAND test_orderbook_gtest)
    # endif()
endif()

# ============================================================================
# Installation
# ============================================================================

# Install library
install(TARGETS cuda_orderbook
    EXPORT CudaOrderbookTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

# Export targets for find_package
install(EXPORT CudaOrderbookTargets
    FILE CudaOrderbookTargets.cmake
    NAMESPACE CudaOrderbook::
    DESTINATION lib/cmake/CudaOrderbook
)

# ============================================================================
# Print Configuration
# ============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "CUDA Orderbook Configuration")
message(STATUS "========================================")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "========================================")
message(STATUS "")

# ============================================================================
# Team Instructions
# ============================================================================

# TO ADD YOUR SOURCE FILES:
# 1. Uncomment the appropriate file in CUDA_ORDERBOOK_SOURCES
# 2. Change cuda_orderbook from INTERFACE to STATIC library
# 3. Uncomment the target_link_libraries line
# 4. Run: cmake --build . --clean-first
#
# EXAMPLE USAGE:
# mkdir build && cd build
# cmake ..
# make -j$(nproc)
#
# TO BUILD IN DEBUG MODE:
# cmake -DCMAKE_BUILD_TYPE=Debug ..
#
# CUDA ARCHITECTURE AUTO-DETECTION:
# By default, CMake will auto-detect your GPU architecture.
# To override auto-detection, specify manually:
# cmake -DCMAKE_CUDA_ARCHITECTURES="52" ..  # SM 52 (Maxwell)
# cmake -DCMAKE_CUDA_ARCHITECTURES="75" ..  # RTX 20xx
# cmake -DCMAKE_CUDA_ARCHITECTURES="86" ..  # RTX 30xx
# cmake -DCMAKE_CUDA_ARCHITECTURES="52;75;86" ..  # Multiple architectures (fat binary)

