message(STATUS "===========module libmtcnn===========")
message(STATUS "Project source Directory: ${PROJECT_SOURCE_DIR}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# include头文件目录
include_directories( ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rdparty/include
        )

# library目录及name名称
link_directories(
    ${PROJECT_SOURCE_DIR}/3rdparty/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/mxnet/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/mxnet/build
    ${PROJECT_SOURCE_DIR}/3rdparty/mxnet/3rdparty/bin
    )

list(APPEND FACE_DETECT_LINKER_LIBS mxnet)

# source directory源文件目录
file(GLOB MTCNN_SRC *.cpp)

set(MTCNN_COMPILE_CODE ${MTCNN_SRC})

message(STATUS "mtcnn code: ${MTCNN_SRC}")

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# generate shared library
add_library(mtcnn SHARED ${MTCNN_COMPILE_CODE})

# generate static library
#add_library(mtcnn_static STATIC ${MTCNN_COMPILE_CODE})

# 指定静态库的输出名称
#set_target_properties(mtcnn_static PROPERTIES OUTPUT_NAME "mtcnn")
if(WIN32)
        target_link_libraries(mtcnn libmxnet ${OpenCV_LIBS})
        #target_link_libraries(mtcnn_static libmxnet ${OpenCV_LIBS})
else()
        target_link_libraries(mtcnn mxnet ${OpenCV_LIBS})
        #target_link_libraries(mtcnn_static mxnet ${OpenCV_LIBS})
endif()
