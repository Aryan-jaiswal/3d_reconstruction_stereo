cmake_minimum_required(VERSION 2.8.3) 
PROJECT (elas)

set(CMAKE_BUILD_TYPE RELEASE)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  geometry_msgs
  message_generation
  roscpp
  sensor_msgs
  std_msgs
  mavros_msgs
  octomap_msgs
  octomap_ros
  tf
  dynamic_reconfigure
)
find_package(OpenCV REQUIRED )
find_package(octomap REQUIRED)
find_package(Boost 1.54.0 COMPONENTS system filesystem program_options REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-msse3 -std=c++1y")
SET(CMAKE_CXX_FLAGS "-O3 -mfpu=neon -std=c++1y")
set( SRC
	descriptor.cpp
	elas.cpp
	filter.cpp
	matrix.cpp
	triangle.cpp   
)

set( HEADERS       
     descriptor.h
     filter.h
     image.h
     matrix.h
     StereoEfficientLargeScale.h
     triangle.h
     popt_pp.h
     timer.h
     sse2neon.h
)
catkin_package(
#  INCLUDE_DIRS include
  CATKIN_DEPENDS cv_bridge image_transport message_generation octomap_msgs octomap_ros roscpp sensor_msgs std_msgs nav_msgs geometry_msgs roscpp message_runtime
  DEPENDS eigen3
)
include_directories(${catkin_INCLUDE_DIRECTORIES} ${OCTOMAP_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
#INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )


add_executable( demo main2.cpp ${SRC} ${HEADERS} )
target_compile_options(demo PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries( demo PRIVATE ${catkin_LIBRARIES} ${OpenCV_LIBS} popt ${OpenMP_CXX_FLAGS})

