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

# TLE-CPU NEON runtime for ARM64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(TRITON_CPU_RUNTIME_SOURCES ${TRITON_CPU_RUNTIME_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_sdot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_swiglu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_flash_attn_decode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_fused_mlp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_transformer_layer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_decode_step.cpp)
endif()

add_library(TritonCPURuntime SHARED ${TRITON_CPU_RUNTIME_SOURCES})
target_link_libraries(TritonCPURuntime PRIVATE ${TRITON_CPU_RUNTIME_LIBS})
# OpenMP for SDOT runtime parallelization
if(APPLE)
  # Homebrew libomp is keg-only, so CMake's FindOpenMP cannot locate it without
  # hints. Point the OpenMP flags/libs at the brew libomp prefix so the runtime
  # .cpp files (which #include <omp.h>) compile and link against -lomp.
  set(_BREW_LIBOMP "/opt/homebrew/opt/libomp")
  if(EXISTS "${_BREW_LIBOMP}/include/omp.h")
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${_BREW_LIBOMP}/include" CACHE STRING "" FORCE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${_BREW_LIBOMP}/include" CACHE STRING "" FORCE)
    set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
    set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
    set(OpenMP_omp_LIBRARY "${_BREW_LIBOMP}/lib/libomp.dylib" CACHE STRING "" FORCE)
    # Make omp.h visible to all runtime translation units regardless of how
    # FindOpenMP resolves the imported target.
    include_directories("${_BREW_LIBOMP}/include")
  endif()
endif()
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(TritonCPURuntime PRIVATE OpenMP::OpenMP_CXX)
elseif(APPLE AND EXISTS "/opt/homebrew/opt/libomp/lib/libomp.dylib")
  # Fallback: link brew libomp directly if the imported target was not created.
  target_link_libraries(TritonCPURuntime PRIVATE "/opt/homebrew/opt/libomp/lib/libomp.dylib")
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
