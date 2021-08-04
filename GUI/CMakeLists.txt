cmake_minimum_required(VERSION 3.5)

project(cofusion-gui)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "Evaluating GUI/CMAKE")

option(ROSBAG "rosbag reader" ON)

option(ROSNODE "read images live as ROS node" ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(ZLIB REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(OpenCV REQUIRED )
find_package(Boost REQUIRED)
find_package(JPEG REQUIRED)

if(ROSBAG)
    find_package(rosbag REQUIRED)
    find_package(tf2 REQUIRED)
    find_package(tf2_msgs REQUIRED)
endif()

if(ROSNODE)
    find_package(roscpp REQUIRED)
    find_package(image_transport REQUIRED)
    find_package(tf2_ros REQUIRED)
endif()

if(ROSBAG OR ROSNODE)
    find_package(cv_bridge REQUIRED)
    find_package(tf2_eigen REQUIRED)
endif()

include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Core)
include_directories(${OPENNI2_INCLUDE_DIR})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${BOOST_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "Include-directories: ")
message(STATUS "ZLIB_INCLUDE_DIR: ${ZLIB_INCLUDE_DIR}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "OPENNI2_INCLUDE_DIR: ${OPENNI2_INCLUDE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR/../Core: ${${CMAKE_CURRENT_SOURCE_DIR}/../Core}")

file(GLOB srcs *.cpp)
file(GLOB tools_srcs Tools/*.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -msse -msse2 -msse3")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Debug build.")
else(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Release build.")
    add_compile_options(-O3)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

add_definitions(-Dlinux=1)

add_executable(CoFusion
               ${srcs}
)

target_link_libraries(CoFusion
                      cofusion
                      CoFusionTools
                      gSLICr_lib # this should be included by target 'cofusion'
                      ${JPEG_LIBRARIES}
)

add_library(CoFusionTools SHARED ${tools_srcs})

target_link_libraries(CoFusionTools
                      ${ZLIB_LIBRARY}
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${OPENNI2_LIBRARY}
                      ${OpenCV_LIBRARIES}
                      boost_filesystem
                      boost_system
                      pthread
)

target_include_directories(CoFusionTools PUBLIC
                            ${CMAKE_CURRENT_SOURCE_DIR}
                            ${ZLIB_INCLUDE_DIR}
                            ${Pangolin_INCLUDE_DIRS}
                            ${CUDA_INCLUDE_DIRS}
                            ${OPENNI2_INCLUDE_DIR}
                            ${OpenCV_INCLUDE_DIRS}
                            ${BOOST_INCLUDE_DIRS}
)

if(ROSBAG)
    target_include_directories(CoFusionTools PUBLIC
        ${rosbag_INCLUDE_DIRS}
        ${tf2_INCLUDE_DIRS}
        ${tf2_msgs_INCLUDE_DIRS}
    )
    target_link_libraries(CoFusionTools
        ${rosbag_LIBRARIES}
        ${tf2_LIBRARIES}
        ${tf2_msgs_LIBRARIES}
    )
    add_definitions(-DROSBAG)
endif()

if(ROSNODE)
    target_include_directories(CoFusionTools PUBLIC
        ${roscpp_INCLUDE_DIRS}
        ${image_transport_INCLUDE_DIRS}
        ${tf2_ros_INCLUDE_DIRS}
    )
    target_link_libraries(CoFusionTools
        ${roscpp_LIBRARIES}
        ${image_transport_LIBRARIES}
        ${tf2_ros_LIBRARIES}
    )
    add_definitions(-DROSNODE)
endif()

if(ROSBAG OR ROSNODE)
    target_include_directories(CoFusionTools PUBLIC
        ${cv_bridge_INCLUDE_DIRS}
        ${tf2_eigen_INCLUDE_DIRS}
    )
    target_link_libraries(CoFusionTools
        ${cv_bridge_LIBRARIES}
        ${tf2_eigen_LIBRARIES}
    )
    add_definitions(-DROSCOMMON)
endif()

include(GNUInstallDirs)

install(TARGETS CoFusionTools
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# install executable
install(TARGETS CoFusion EXPORT cofusion
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
