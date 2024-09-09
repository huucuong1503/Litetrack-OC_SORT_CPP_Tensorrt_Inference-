cmake_minimum_required(VERSION 3.10)
project(utility_functions)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/home/rtr/cuong/LiteTrackInferenceCPP/myCPP/libs/common/util/include)
# Include directories for CUDA and OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_library(utility_func SHARED src/utils.cpp)  # Change "main.cpp" to your actual source file name if different


# Link the TensorRT and CUDA libraries
target_link_libraries(utility_func
    # cudart
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)