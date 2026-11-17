include(CheckSymbolExists)
# Find OneDNN ukernel library
find_package(dnnl CONFIG)
if (dnnl_FOUND)
  get_target_property(dnnl_include DNNL::dnnl INTERFACE_INCLUDE_DIRECTORIES)
  # currently used only in triton_cpu.cc and in ConvertDotToOneDNN
  include_directories(${dnnl_include})
  set (dnnl_header ${dnnl_include}/oneapi/dnnl/dnnl_config.h)
  check_symbol_exists(DNNL_EXPERIMENTAL_UKERNEL ${dnnl_header}  HAS_UKERNELS)
  if (HAS_UKERNELS)
    message(STATUS "Found OneDNN/DNNL with Ukernels support")
    add_compile_definitions(ONEDNN_AVAILABLE)
  else ()
    message(STATUS "SKIPPING found OneDNN/DNNL because Ukernels support is missing")
  endif()
else ()
  message(STATUS "Could NOT find OneDNN/DNNL")
endif()

# Find XSMM ukernel library
find_library(LIBXSMM xsmm
  PATHS ENV XSMM_LIBRARY_DIRS
)
find_path(LIBXSMM_INCLUDE_DIR
  NAMES libxsmm.h
  PATHS ENV XSMM_INCLUDE_DIRS
)
if (LIBXSMM AND LIBXSMM_INCLUDE_DIR)
  set(xsmm_FOUND True)
  message(STATUS "Found XSMM: " ${LIBXSMM})
  add_compile_definitions(XSMM_AVAILABLE)
  include_directories(${LIBXSMM_INCLUDE_DIR})
else ()
  message(STATUS "Could NOT find XSMM")
endif()

# Build Triton-CPU plugin
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonCPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_cpu.cc LINK_LIBS TritonCPUToLLVM TritonCPUTransforms)
  target_link_libraries(TritonCPU PUBLIC MLIRVectorToSCF MLIRAffineToStandard MLIRMathToLibm MLIRMemRefTransforms MLIRReconcileUnrealizedCasts PRIVATE Python3::Module pybind11::headers)
endif()

# Configure and build Triton-CPU runtime
set(TRITON_CPU_RUNTIME_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/runtime/cpu_runtime.cpp)
set(TRITON_CPU_RUNTIME_LIBS LLVMSupport)
if (dnnl_FOUND AND HAS_UKERNELS)
  set(TRITON_CPU_RUNTIME_SOURCES ${TRITON_CPU_RUNTIME_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_onednn.cpp)
  set(TRITON_CPU_RUNTIME_LIBS ${TRITON_CPU_RUNTIME_LIBS} DNNL::dnnl)
endif()
if(xsmm_FOUND)
  set(TRITON_CPU_RUNTIME_SOURCES ${TRITON_CPU_RUNTIME_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_xsmm.cpp)
  set(TRITON_CPU_RUNTIME_LIBS ${TRITON_CPU_RUNTIME_LIBS} ${LIBXSMM})
endif()

# TLE-CPU NEON runtime for ARM64 (route C: raw_call micro-kernels)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(TRITON_CPU_RUNTIME_SOURCES ${TRITON_CPU_RUNTIME_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_sdot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_swiglu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_flash_attn_decode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_fused_mlp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_transformer_layer.cpp)
  # The micro-kernels use NEON SDOT (vdotq_s32 -> +dotprod) and INT8 MMLA
  # (vmmla -> +i8mm); without these the guarded real implementations fall
  # back to empty stubs. Linux/O6N keeps the arm64-backend armv9-a+sve2 ISA;
  # macOS arm64 (Apple clang) uses an armv8.6-a fixed-mode equivalent.
  if(APPLE)
    set(TLE_CPU_RUNTIME_ARCH "armv8.6-a+dotprod+i8mm")
  else()
    set(TLE_CPU_RUNTIME_ARCH "armv9-a+sve2+i8mm+bf16+fp16+dotprod")
  endif()
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_sdot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_swiglu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_flash_attn_decode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_fused_mlp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_transformer_layer.cpp
    PROPERTIES COMPILE_FLAGS "-march=${TLE_CPU_RUNTIME_ARCH}")
endif()

add_library(TritonCPURuntime SHARED ${TRITON_CPU_RUNTIME_SOURCES})
target_link_libraries(TritonCPURuntime PRIVATE ${TRITON_CPU_RUNTIME_LIBS})
# OpenMP for SDOT runtime parallelization (kernels call omp_get_thread_num).
if(APPLE)
  # Apple clang ships no OpenMP. PyTorch already loads its own libomp; linking a
  # second copy (e.g. Homebrew) triggers OMP Error #15 (duplicate runtime) and a
  # segfault. Instead compile against the OpenMP header but leave the omp symbols
  # undefined, resolving them at load time from torch's libomp via dynamic_lookup.
  find_path(LIBOMP_INCLUDE_DIR omp.h
    PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
  if(LIBOMP_INCLUDE_DIR)
    target_include_directories(TritonCPURuntime PRIVATE ${LIBOMP_INCLUDE_DIR})
    target_compile_options(TritonCPURuntime PRIVATE -Xclang -fopenmp)
    target_link_options(TritonCPURuntime PRIVATE "-Wl,-undefined,dynamic_lookup")
  else()
    # The runtime sources unconditionally #include <omp.h>, so a missing header
    # is a hard compile failure, not a graceful "OpenMP disabled" fallback.
    message(FATAL_ERROR "omp.h not found; install libomp (brew install libomp) to build the TLE-CPU runtime")
  endif()
else()
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(TritonCPURuntime PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# Build and link sleef
set(SLEEF_BUILD_SHARED_LIBS ON CACHE BOOL "Build sleef shared lib" FORCE)
set(SLEEF_BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
set(SLEEF_BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
set(SLEEF_BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)
set(SLEEF_BUILD_SCALAR_LIB OFF CACHE BOOL "libsleefscalar will not be built." FORCE)
add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/sleef" sleef)
# Override sleef's output directory with our own
set_target_properties(sleef PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
