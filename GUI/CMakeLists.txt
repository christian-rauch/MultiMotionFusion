cmake_minimum_required(VERSION 3.16)

project(multimotionfusion-gui LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Werror=deprecated-declarations>
)

message(STATUS "Evaluating GUI/CMAKE")

if($ENV{ROS_VERSION})
set(HASROS ON)
if($ENV{ROS_VERSION} EQUAL 1)
    set(HASROS1 ON)
    add_compile_options(-DROS1)
elseif($ENV{ROS_VERSION} EQUAL 2)
    set(HASROS2 ON)
    add_compile_options(-DROS2)
endif()
endif()

option(ROSBAG "rosbag reader" ${HASROS1})

option(ROSNODE "read images live as ROS node" ${HASROS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(ZLIB REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(OpenCV REQUIRED )
find_package(Boost REQUIRED filesystem)
find_package(JPEG REQUIRED)

# TODO: Why do we have to do this here again?
find_package(super_point_inference REQUIRED)

if(ROSBAG)
    find_package(rosbag REQUIRED)
    find_package(tf2 REQUIRED)
    find_package(tf2_msgs REQUIRED)
endif()

if(ROSNODE)
    if(HASROS1)
        find_package(roscpp REQUIRED)
    endif()
    add_compile_options(-DROSUI)
    add_compile_options(-DROSSTATE)
    add_compile_options(-DROSREADER)
    find_package(image_transport REQUIRED)
    find_package(tf2_ros REQUIRED)
    find_package(cob_srvs REQUIRED)
    find_package(std_srvs REQUIRED)
endif()

if(ROSBAG OR ROSNODE)
    find_package(cv_bridge REQUIRED)
    find_package(tf2_eigen REQUIRED)
    find_package(image_geometry REQUIRED)
endif()

include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Core)
include_directories(${OPENNI2_INCLUDE_DIR})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${BOOST_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "Include-directories: ")
message(STATUS "ZLIB_INCLUDE_DIR: ${ZLIB_INCLUDE_DIR}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message(STATUS "OPENNI2_INCLUDE_DIR: ${OPENNI2_INCLUDE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR/../Core: ${${CMAKE_CURRENT_SOURCE_DIR}/../Core}")

file(GLOB srcs *.cpp)
file(GLOB tools_srcs Tools/*.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Debug build.")
else(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Release build.")
    add_compile_options(-O3)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

add_definitions(-Dlinux=1)

add_executable(MultiMotionFusion
               ${srcs}
)

target_link_libraries(MultiMotionFusion
                      multimotionfusion
                      MultiMotionFusionTools
                      gSLICr_lib # this should be included by target 'multimotionfusion'
                      ${JPEG_LIBRARIES}
                      super_point_inference
)

add_library(MultiMotionFusionTools SHARED ${tools_srcs})

target_link_libraries(MultiMotionFusionTools
                      ${ZLIB_LIBRARY}
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${OPENNI2_LIBRARY}
                      ${OpenCV_LIBRARIES}
                      Boost::filesystem
                      pthread
)

target_include_directories(MultiMotionFusionTools PUBLIC
                            ${CMAKE_CURRENT_SOURCE_DIR}
                            ${ZLIB_INCLUDE_DIR}
                            ${Pangolin_INCLUDE_DIRS}
                            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                            ${OPENNI2_INCLUDE_DIR}
                            ${OpenCV_INCLUDE_DIRS}
                            ${BOOST_INCLUDE_DIRS}
)

set_property(TARGET MultiMotionFusionTools PROPERTY CXX_STANDARD 17)

if(ROSBAG)
    target_include_directories(MultiMotionFusionTools PUBLIC
        ${rosbag_INCLUDE_DIRS}
        ${tf2_INCLUDE_DIRS}
        ${tf2_msgs_INCLUDE_DIRS}
    )
    target_link_libraries(MultiMotionFusionTools
        ${rosbag_LIBRARIES}
        ${tf2_LIBRARIES}
        ${tf2_msgs_LIBRARIES}
    )
    add_definitions(-DROSBAG)
endif()

if(ROSNODE)
    target_include_directories(MultiMotionFusionTools PUBLIC
        ${roscpp_INCLUDE_DIRS}
        ${image_transport_INCLUDE_DIRS}
        ${tf2_ros_INCLUDE_DIRS}
        ${cob_srvs_INCLUDE_DIRS}
        ${std_srvs_INCLUDE_DIRS}
    )
    target_link_libraries(MultiMotionFusionTools
        ${roscpp_LIBRARIES}
        ${image_transport_LIBRARIES}
        ${tf2_ros_LIBRARIES}
        ${cob_srvs_LIBRARIES}
        ${std_srvs_LIBRARIES}
    )
if(HASROS2)
    target_link_libraries(MultiMotionFusionTools
        image_geometry::image_geometry
        image_transport::image_transport
    )
endif()
    # to find "gSLICr.h" and "super_point_inference.hpp"
    target_link_libraries(MultiMotionFusionTools gSLICr_lib super_point_inference)
    add_definitions(-DROSNODE)
endif()

if(ROSBAG OR ROSNODE)
    target_include_directories(MultiMotionFusionTools PUBLIC
        ${cv_bridge_INCLUDE_DIRS}
        ${tf2_eigen_INCLUDE_DIRS}
        ${image_geometry_INCLUDE_DIRS}
    )
    target_link_libraries(MultiMotionFusionTools
        ${cv_bridge_LIBRARIES}
        ${tf2_eigen_LIBRARIES}
        ${image_geometry_LIBRARIES}
    )
if(HASROS2)
    target_link_libraries(MultiMotionFusionTools
        image_geometry::image_geometry
        cv_bridge::cv_bridge
    )
endif()
    add_definitions(-DROSCOMMON)
endif()

include(GNUInstallDirs)

install(TARGETS MultiMotionFusionTools
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# install executable
install(TARGETS MultiMotionFusion EXPORT multimotionfusion
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
