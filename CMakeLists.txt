cmake_minimum_required(VERSION 3.10)
project(TensorRTInference)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Find the Eigen3 package
find_package(Eigen3 REQUIRED)


# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/home/rtr/cuong/LiteTrackInferenceCPP/myCPP/libs/inference_lib/include)
include_directories(/home/rtr/cuong/LiteTrackInferenceCPP/myCPP/libs/common/util/include)
include_directories(/home/rtr/cuong/LiteTrackInferenceCPP/myCPP/libs/OCSort/include)


# Include directories for CUDA and OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Link directories
link_directories(${TensorRT_LIBRARY_DIR})


add_subdirectory(/home/rtr/cuong/LiteTrackInferenceCPP/myCPP/libs/common/util)
add_subdirectory(/home/rtr/cuong/LiteTrackInferenceCPP/myCPP/libs/inference_lib)

file(GLOB SRC_LIST /home/rtr/cuong/LiteTrackInferenceCPP/myCPP/libs/OCSort/src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SRC_LIST})
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

# Add the executable
add_executable(litetrack_infer src/main.cpp)  # Change "main.cpp" to your actual source file name if different


# Link the TensorRT and CUDA libraries
target_link_libraries(litetrack_infer
    nvinfer
    nvonnxparser
    # cudart
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    tensorrt_inference
    utility_func
    Eigen3::Eigen
    ${PROJECT_NAME} 
)