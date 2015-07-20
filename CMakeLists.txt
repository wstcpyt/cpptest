cmake_minimum_required(VERSION 3.2)
project(cpptest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(cpptest ${SOURCE_FILES})

find_package(Boost 1.32 REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(cpptest ${Boost_LIBRARIES})


find_package(gsl REQUIRED COMPONENTS program_options)
include_directories(${gsl_INCLUDE_DIRS})
target_link_libraries( cpptest ${gsl_LIBRARIES})