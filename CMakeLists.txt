cmake_minimum_required(VERSION 3.18)
project(qwen3-infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Jetson Orin Nano: SM 8.7
set(CMAKE_CUDA_ARCHITECTURES 87)

option(USE_CUBLAS "Use cuBLAS for GEMM (recommended)" ON)

find_package(CUDAToolkit REQUIRED)

set(SOURCES
    src/model_loader.cpp
    src/tokenizer.cpp
    src/weight_store.cu
    src/activation_pool.cu
    src/kv_cache.cu
    src/model.cu
    src/sampler.cu
    src/engine.cpp
)

set(KERNELS
    kernels/embedding.cu
    kernels/rmsnorm.cu
    kernels/rope.cu
    kernels/attention.cu
    kernels/matmul.cu
    kernels/fused_swiglu.cu
    kernels/softmax.cu
)

# Main executable
add_executable(qwen3-infer main.cpp ${SOURCES} ${KERNELS})
target_include_directories(qwen3-infer PRIVATE include)
target_link_libraries(qwen3-infer PRIVATE CUDA::cudart)

if(USE_CUBLAS)
    target_link_libraries(qwen3-infer PRIVATE CUDA::cublas)
    target_compile_definitions(qwen3-infer PRIVATE USE_CUBLAS=1)
endif()

# Tests (no CUDA dependency, run on host)
add_executable(test_loader tests/test_loader.cpp src/model_loader.cpp)
target_include_directories(test_loader PRIVATE include)

add_executable(test_tokenizer tests/test_tokenizer.cpp src/model_loader.cpp src/tokenizer.cpp)
target_include_directories(test_tokenizer PRIVATE include)
