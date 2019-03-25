cmake_minimum_required(VERSION 2.6)
project(face_landmark_detection)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC")

set(SOURCE_FILES src/main.cpp src/face_landmark_detection.cpp include/face_landmark_detection.h src/facenet_caffe.cpp include/facenet_caffe.h include/caffe_register.h)

set(LIBPATH_MIDDLE ${PROJECT_SOURCE_DIR}/lib/middle)
file(GLOB bLIBS "${LIBPATH_MIDDLE}/libboost*.so")
file(GLOB oLIBS "${LIBPATH_MIDDLE}/libopencv*.so")
file(GLOB hLIBS "${LIBPATH_MIDDLE}/libhdf5*.so")
file(GLOB olibs_point "${PROJECT_SOURCE_DIR}/lib/*.a")

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/middle)

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib/middle)

#############################################################

add_executable(face_landmark ${SOURCE_FILES})

#ADD_LIBRARY(face_landmark SHARED ${SOURCE_FILES})
#SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

target_link_libraries(face_landmark ${olibs_point} ${bLIBS} ${oLIBS} ${hLIBS} dlib caffe glog gflags protobuf openblas opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
