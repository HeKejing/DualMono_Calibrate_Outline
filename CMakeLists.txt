cmake_minimum_required(VERSION 2.8.11)

project(GrayCalibration_Outline)

#---------------------------------------------------------------------------------------
# OpenCV
#---------------------------------------------------------------------------------------
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})



#---------------------------------------------------------------------------------------
# ${PROJECT_NAME}
#---------------------------------------------------------------------------------------
include_directories("${CMAKE_SOURCE_DIR}/src")
AUX_SOURCE_DIRECTORY(${CMAKE_SOURCE_DIR}/src DIR_SRCS)
FILE(GLOB DIR_HEADERS ${CMAKE_SOURCE_DIR}/src/*.h)

MESSAGE(STATUS ${DIR_SRCS})

add_executable(${PROJECT_NAME} ${DIR_SRCS} ${DIR_HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


#---------------------------------------------------------------------------------------
# set groups
#---------------------------------------------------------------------------------------
source_group("Header Files" FILES ${DIR_HEADERS})


    
    