
if(NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release)
   message(STATUS "CMAKE_BUILD_TYPE Release")
else()
   SET(CMAKE_BUILD_TYPE DEBUG)
   message(STATUS "CMAKE_BUILD_TYPE DEBUG")
endif()

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

#SET(TARGET_COMPILER "x64")
if(TARGET_COMPILER STREQUAL "x64")
    message(STATUS "TARGET_COMPILE STREQUAL X64")
    SET(NCNN_INSTALL_DIR /home/nnn/external/ncnn/build/x64/src/)
    SET(ZBAR_INSTALL_DIR /home/nnn/external/zbar-0.10/build/x64/)
    SET(ZBAR_INSTALL_DIR /home/nnn/external/zbar-0.10/build/x64/)
    SET(OpenCV_DIR /home/nnn/external/opencv/build/x64)
else()
    message(STATUS "TARGET_COMPILE STREQUAL arm")
    SET(CMAKE_CXX_COMPILER arm-none-linux-gnueabihf-g++)
    SET(CMAKE_C_COMPILER arm-none-linux-gnueabihf-gcc)
    SET(NCNN_INSTALL_DIR /home/nnn/external/ncnn/install/arm/lib)
    SET(ZBAR_INSTALL_DIR /home/nnn/external/zbar-0.10/build/arm)
    SET(OpenCV_DIR /home/nnn/external/opencv/build/arm)
endif()


# project name
PROJECT(example)

# requirement of cmake version
cmake_minimum_required(VERSION 3.5)


# set the directory of executable files
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
if(TARGET_COMPILER STREQUAL "x64")
    message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY X64")
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/x64)
else()
    message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY arm")
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/arm)
endif()

set(Target ${TARGET})

# find required opencv
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})

# name of executable file and path of source file
add_executable(${Target} main/${Target}.cpp)

# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})

# includes
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/ncnn)

# opencv libraries
target_link_libraries(${Target} ${OpenCV_LIBRARIES})
target_link_libraries(${Target} ${NCNN_INSTALL_DIR}/libncnn.a -fopenmp)

# .so link
target_link_libraries(${Target} ${ZBAR_INSTALL_DIR}/lib/libzbar.so)


FILE(GLOB_RECURSE PRJ_SRC
        "${PROJECT_SOURCE_DIR}/src/*.c"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        )


add_library(mainlib ${PRJ_SRC})
target_link_libraries(mainlib ${OpenCV_LIBRARIES})
target_link_libraries(mainlib ${NCNN_INSTALL_DIR}/libncnn.a -fopenmp)

target_link_libraries(${Target} mainlib)


