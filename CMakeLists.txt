cmake_minimum_required(VERSION 3.18)

project("mfem" LANGUAGES CXX CUDA)

if (NOT CMAKE_BUILD_TYPE)
  message (STATUS "No build type selected, default to Release")
  set (CMAKE_BUILD_TYPE "Release")
endif ()

# Maybe stop from CMAKEing in the wrong place
if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Source and build directories cannot be the same. Go use the /build directory.")
endif()

### Configure output locations
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

### Compiler options
set(CMAKE_EXPORT_COMPILE_COMMANDS 1 ) # Emit a compile flags file to support completion engines 

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using Clang (linux or apple) or GCC
  message("Using clang/gcc compiler flags")
  SET(BASE_CXX_FLAGS "-std=c++11")# -g3")
  SET(DISABLED_WARNINGS " -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-deprecated-declarations -Wno-missing-braces")
  SET(TRACE_INCLUDES " -H -Wno-error=unused-command-line-argument")

  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    message("Setting clang-specific options")
    SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -ferror-limit=5 -fcolor-diagnostics")
    SET(CMAKE_CXX_FLAGS_DEBUG          "-fsanitize=address -fno-limit-debug-info")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -fmax-errors=5")
    message("Setting gcc-specific options")
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} -Wno-maybe-uninitialized -Wreorder \
      -Wuninitialized -Wunused-result -Wunused-local-typedefs -Wsign-compare  \
      -Wno-format-zero-length -Wno-unused-but-set-parameter -Wno-unused-but-set-variable \
      -Wno-pedantic")
  endif()


  SET(CMAKE_C_FLAGS "${DISABLED_WARNINGS} -Wno-pedantic")
  SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
  #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TRACE_INCLUDES}") # uncomment if you need to track down where something is getting included from
  SET(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -g3")
  SET(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
  SET(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG")
  SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
  message("Using Visual Studio compiler flags")
  set(BASE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  set(BASE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP") # parallel build
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4267\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4244\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4305\"")  # ignore truncation on initialization
  SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

  add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
  add_definitions(-DNOMINMAX)
  add_definitions(-D_USE_MATH_DEFINES)

else()
  # unrecognized
  message( FATAL_ERROR "Unrecognized compiler [${CMAKE_CXX_COMPILER_ID}]" )
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Some Flags for OMP to work on OSx
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp /opt/homebrew/opt/libomp/lib/libomp.dylib -I/opt/homebrew/opt/libomp/include -Wunused-command-line-argument ")
  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
      set(OpenMP_C "${CMAKE_C_COMPILER}")
      set(OpenMP_C_FLAGS "-Wno-unused-command-line-argument")
      set(OpenMP_C_LIB_NAMES "")
      set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
      set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
      set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    set(OpenMP_CXX_FLAGS "-Wno-unused-command-line-argument")
    set(OpenMP_CXX_LIB_NAMES "")
    set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
  endif()
endif()
    
# Find Eigen
find_package(Eigen3 3.4.0 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS)
find_package(LAPACK)
find_package(CHOLMOD)
find_package(SuiteSparse)
find_package(Ginkgo REQUIRED)

# CMAKE IS MY NIGHTMARE
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -Xcompiler -fPIC -arch=sm_86)
set(CUDA_NVCC_FLAGS -Xcudafe "--diag_suppress=cc_clobber_ignored " ;
    -Xcudafe -use_fast_math " --diag_suppress=set_but_not_used " ;
    ${CUDA_NVCC_FLAGS} " --extended-lambda --disable-warnings")
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--extended-lambda" )

# Add polyscope
message("\n\n == CMAKE recursively building Polyscope\n")
add_subdirectory("deps/polyscope")
add_subdirectory("deps/amgcl")

file(GLOB SOURCES 
    ${PROJECT_SOURCE_DIR}/src/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/svd/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/boundary_conditions/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/energies/*.cpp
    ${PROJECT_SOURCE_DIR}/src/optimizers/*.cpp
    ${PROJECT_SOURCE_DIR}/src/variables/*.cpp
    ${PROJECT_SOURCE_DIR}/src/linear_solvers/*.cpp
    ${PROJECT_SOURCE_DIR}/src/time_integrators/*.cpp
    ${PROJECT_SOURCE_DIR}/src/factories/*.cpp
    ${PROJECT_SOURCE_DIR}/src/utils/*.cpp
    ${PROJECT_SOURCE_DIR}/src/utils/collision/*.cpp
    ${PROJECT_SOURCE_DIR}/src/mesh/*.cpp
    ${PROJECT_SOURCE_DIR}/src/svd/*.cu 
    ${PROJECT_SOURCE_DIR}/src/time_integrators/*.cu
    ${PROJECT_SOURCE_DIR}/src/linear_solvers/*.cu
    ${PROJECT_SOURCE_DIR}/src/energies/*.cu
    ${PROJECT_SOURCE_DIR}/src/optimizers/*.cu
    ${PROJECT_SOURCE_DIR}/src/variables/*.cu
    ${PROJECT_SOURCE_DIR}/src/utils/*.cu)

add_library(mixed_fem_lib STATIC ${SOURCES})

include(FetchContent)
FetchContent_Declare(
    finite-diff
    GIT_REPOSITORY https://github.com/zfergus/finite-diff.git
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(finite-diff)

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.x
)
FetchContent_MakeAvailable(Catch2)

set(GPUTI_WITH_DOUBLE_PRECISION ON CACHE BOOL "Enable Double Precision" FORCE)
set(IPC_TOOLKIT_WITH_CUDA ON CACHE BOOL "Enable CUDA Implementation" FORCE)
include(FetchContent)
FetchContent_Declare(
    ipc_toolkit
    GIT_REPOSITORY https://github.com/tytrusty/ipc-toolkit
    GIT_SHALLOW FALSE
    GIT_TAG main
)
FetchContent_MakeAvailable(ipc_toolkit)

target_link_libraries(mixed_fem_lib ipc::toolkit)

set(CMAKE_CXX_STANDARD 17)

set_target_properties(mixed_fem_lib PROPERTIES 
      POSITION_INDEPENDENT_CODE ON 
      CUDA_SEPARABLE_COMPILATION ON)

find_library(CUDA_RUNTIME_LIBS_DYNAMIC cudart
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})   

target_link_libraries(mixed_fem_lib ${CUDA_RUNTIME_LIBS} 
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_LIBRARIES}
    )
set(CMAKE_CUDA_ARCHITECTURES 86)
set_target_properties(mixed_fem_lib PROPERTIES CUDA_ARCHITECTURES "86")

# Add OMP library and includes
if(OpenMP_CXX_FOUND)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        execute_process(
            COMMAND brew --prefix libomp 
            RESULT_VARIABLE BREW_OMP
            OUTPUT_VARIABLE BREW_OMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        include_directories(${BREW_OMP_PREFIX}/include)
        target_link_libraries(mixed_fem_lib OpenMP::OpenMP_CXX)
        target_compile_definitions(mixed_fem_lib PUBLIC -DSIM_USE_OPENMP)

    else()
        include_directories(${OpenMP_CXX_INCLUDE_DIRS})
        target_link_libraries(mixed_fem_lib OpenMP::OpenMP_CXX)
        target_compile_definitions(mixed_fem_lib PUBLIC -DSIM_USE_OPENMP)
    endif()
endif()

# Blas, lapack, and cholmod
if(BLAS_FOUND)
  target_link_libraries(mixed_fem_lib ${BLAS_LIBRARIES})
endif()
if(LAPACK_FOUND)
  target_link_libraries(mixed_fem_lib ${LAPACK_LIBRARIES})
endif()
if(CHOLMOD_FOUND AND BLAS_FOUND AND LAPACK_FOUND)
  include_directories(${CHOLMOD_INCLUDE_DIR})
  target_link_libraries(mixed_fem_lib ${CHOLMOD_LIBRARIES})
  target_compile_definitions(mixed_fem_lib PUBLIC -DSIM_USE_CHOLMOD)
endif()

TRY_COMPILE(COMPILER_SUPPORTS_ARM_NEON ${CMAKE_BINARY_DIR}
   ${PROJECT_SOURCE_DIR}/cmake/check_arm_neon.cpp)
TRY_COMPILE(COMPILER_SUPPORTS_AVX    ${CMAKE_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}//cmake/check_isa.cpp COMPILE_DEFINITIONS ${FLAGS_AVX})

if (COMPILER_SUPPORTS_ARM_NEON)
  target_compile_definitions(mixed_fem_lib PUBLIC -DUSE_NEON_IMPLEMENTATION)
elseif(COMPILER_SUPPORTS_AVX)  
  target_compile_definitions(mixed_fem_lib PUBLIC -DUSE_AVX_IMPLEMENTATION)
elseif ()
  target_compile_definitions(mixed_fem_lib PUBLIC -DUSE_SSE_IMPLEMENTATION)
ENDIF()

# Include settings
target_include_directories(mixed_fem_lib SYSTEM PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/deps/Bartels/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/deps/libigl/include" 
  "${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/args"
  "${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/json/include")
target_include_directories(mixed_fem_lib SYSTEM PUBLIC "/usr/local/cuda/include")

target_compile_options(mixed_fem_lib
  PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${GINKGO_CUDA_ARCH_FLAGS}>")
  #--generate-code=arch=compute_86,code=sm_86
# Link settings
target_link_libraries(mixed_fem_lib polyscope Eigen3::Eigen amgcl::amgcl Ginkgo::ginkgo)

# Create an executable
add_executable(convergence apps/convergence.cpp)
target_link_libraries(convergence mixed_fem_lib)

add_executable(convergence_iterative apps/convergence_iterative.cpp )
target_link_libraries(convergence_iterative mixed_fem_lib)

add_executable(sim2d apps/sim2d.cpp)
target_link_libraries(sim2d PUBLIC mixed_fem_lib)

add_executable(sim3d apps/sim3d.cpp)
set_target_properties(sim3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(sim3d mixed_fem_lib)

add_executable(sim3d_gpu apps/sim3d_gpu.cpp)
set_target_properties(sim3d_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(sim3d_gpu mixed_fem_lib)

add_subdirectory(tests)
