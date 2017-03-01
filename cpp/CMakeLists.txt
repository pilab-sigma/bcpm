cmake_minimum_required(VERSION 2.8)

project(bcpm)

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )

SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Werror -std=c++11" )

add_subdirectory(test/)
