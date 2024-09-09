cmake_minimum_required(VERSION 3.10)
project(libocsort)

# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} /O2") 
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_RELEASE} /O2")

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(CMAKE_CXX_STANDARD 17)

# Linker external library
# SET(Eigen3_DIR "C:/eigen-3.4.0/build")
find_package(Eigen3 REQUIRED)
# set(OpenCV_DIR "C:/opencv")
find_package(OpenCV REQUIRED)

include_directories(/home/rtr/cuong/LiteTrackInferenceCPP/myCPP/libs/OCSort/include)

# add_subdirectory(src)
file(GLOB SRC_LIST src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SRC_LIST})
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

# note：test with yolo inference
add_executable(test mutilthread.cpp)
target_link_libraries(test PUBLIC Eigen3::Eigen ${PROJECT_NAME} ${OpenCV_LIBS})


