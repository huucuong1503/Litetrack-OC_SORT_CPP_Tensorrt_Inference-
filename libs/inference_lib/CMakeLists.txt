cmake_minimum_required(VERSION 3.10)
project(TensorRTInference)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)
# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/home/rtr/cuong/LiteTrackInferenceCPP/myCPP/libs/common/temp_folder/template_feature/include)
include_directories(/home/rtr/cuong/LiteTrackInferenceCPP/myCPP/libs/common/temp_folder/forward_hann/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Link directories
link_directories(${TensorRT_LIBRARY_DIR})

# Add the executable
# add_executable(tensorrt_inference_forZ src/forwardZ.cpp)  # Change "main.cpp" to your actual source file name if different
add_library(tensorrt_inference SHARED src/forward.cpp)  # Change "main.cpp" to your actual source file name if different


# Link the TensorRT and CUDA libraries
target_link_libraries(tensorrt_inference
    nvinfer
    nvonnxparser
    # cudart
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)