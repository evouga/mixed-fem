set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}) 

#file(GLOB_RECURSE TEST_SOURCES ${PROJECT_SOURCE_DIR}/tests *.cpp)
file(GLOB_RECURSE TEST_SOURCES ${PROJECT_SOURCE_DIR}/tests test_dsvd.cpp)

add_executable(all_tests ${TEST_SOURCES})
set_target_properties(all_tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_link_libraries(all_tests mixed_fem_lib Catch2::Catch2 finitediff::finitediff) 