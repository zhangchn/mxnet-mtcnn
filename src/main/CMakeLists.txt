message(STATUS "============main module===========")
# include头文件目录
include_directories( ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include
                   )

# library目录及name名称
link_directories(
${PROJECT_SOURCE_DIR}/lib 
${PROJECT_SOURCE_DIR}/3rdparty/lib
    ${PROJECT_SOURCE_DIR}/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/mxnet/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/mxnet/build
    ${PROJECT_SOURCE_DIR}/3rdparty/mxnet/3rdparty/bin
)


# test picture execute file
add_executable(test_picture test_picture.cpp)
target_link_libraries(test_picture mtcnn ${OpenCV_LIBS})
set_target_properties(test_picture
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

# test camera execute file
add_executable(test_camera test_camera.cpp)
target_link_libraries(test_camera mtcnn ${OpenCV_LIBS})
set_target_properties(test_camera
        PROPERTIES 
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

# test stream with multithreading
add_executable(stream_process stream_process.cpp)
target_link_libraries(stream_process mtcnn ${OpenCV_LIBS})
set_target_properties(stream_process
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_library(detect SHARED detect.cpp)

target_link_libraries(detect mtcnn ${OpenCV_LIBS})

# test detect.dll
add_executable(test_detect test_detect.cpp)
target_link_libraries(test_detect detect)
set_target_properties(test_detect
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
