cmake_minimum_required(VERSION 2.8.3) 
PROJECT (elas)



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
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-msse3 -std=c++1y")
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
)
catkin_package(
#  INCLUDE_DIRS include
  CATKIN_DEPENDS cv_bridge image_transport message_generation octomap_msgs octomap_ros roscpp sensor_msgs std_msgs nav_msgs geometry_msgs roscpp message_runtime
  DEPENDS eigen3
)
include_directories(${catkin_INCLUDE_DIRECTORIES} ${OCTOMAP_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
#INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )


add_executable( demo main2.cpp ${SRC} ${HEADERS} )
target_link_libraries( demo ${catkin_LIBRARIES} ${OpenCV_LIBS} ${OpenMP_CXX_FLAGS}  popt)

add_executable( image_publisher publish.cpp)
target_link_libraries( image_publisher ${catkin_LIBRARIES} ${OpenCV_LIBS} )
