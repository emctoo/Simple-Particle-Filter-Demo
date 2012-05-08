cmake_minimum_required(VERSION 2.8)

project(ParticleFilter)

# set(CMAKE_CXX_COMPILER clang)
set(source_files 
	main.cc
)

INCLUDE_DIRECTORIES(
        ${PROJECT_SOURCE_DIR}/src
        /usr/include/opencv
)

# add_definitions(-g -fshow-source-location)
add_definitions(-g )

add_executable(pf
    ${source_files}
)
target_link_libraries(pf
    opencv_core opencv_imgproc opencv_highgui opencv_ml opencv_video
    opencv_features2d opencv_calib3d opencv_objdetect opencv_contrib
    opencv_legacy opencv_flann
)

add_executable(fback fback.cpp)
target_link_libraries(fback
    opencv_core opencv_imgproc opencv_highgui opencv_ml opencv_video
    opencv_features2d opencv_calib3d opencv_objdetect opencv_contrib
    opencv_legacy opencv_flann
)

add_executable(lkdemo lkdemo.cpp)
target_link_libraries(lkdemo
    opencv_core opencv_imgproc opencv_highgui opencv_ml opencv_video
    opencv_features2d opencv_calib3d opencv_objdetect opencv_contrib
    opencv_legacy opencv_flann
)

