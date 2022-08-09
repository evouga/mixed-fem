cmake_minimum_required(VERSION 3.1)
project(collisiondetection)

find_package(Eigen3 3.4.0 REQUIRED)

set(CCD_ROOT "${CMAKE_CURRENT_LIST_DIR}/../deps/collisiondetection")
set(CCD_SOURCE_DIR ${CCD_ROOT}/src)
set(CCD_INCLUDE_DIR ${CCD_ROOT}/include)
message("CCD SOURCE " ${CCD_SOURCE_DIR})

file(GLOB LIBFILES ${CCD_SOURCE_DIR}/*.cpp)
add_library(${PROJECT_NAME} STATIC ${LIBFILES})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

target_include_directories(${PROJECT_NAME} PRIVATE ${CCD_INCLUDE_DIR})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
