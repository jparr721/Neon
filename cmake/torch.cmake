include(FetchContent)
FetchContent_Declare(
        pytorch
        GIT_REPOSITORY https://github.com/pytorch/pytorch.git
        GIT_TAG v1.9.0
)

# Configure for minimal, static, openblas-enabled GPU library.
set(BUILD_SHARED_LIBS OFF)
set(USE_MKL OFF) # poor perf vs. openblas on older CPUs, ignores threading limits, cpuid-isms
set(USE_MKLDNN OFF) # ditto
set(USE_CUDA ON)
set(USE_DISTRIBUTED OFF)
set(BUILD_PYTHON OFF)
set(USE_ROCM OFF)
set(USE_NCCL OFF)
set(BUILD_CAFFE2_OPS OFF)
set(BLAS_INFO "open")
set(WITH_BLAS "open")
set(USE_NUMA OFF)
FetchContent_MakeAvailable(pytorch)
