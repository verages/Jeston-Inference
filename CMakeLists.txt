cmake_minimum_required(VERSION 3.10)
project(yolo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# CUDA & TensorRT (仅支持aarch64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    set(CUDA_INCLUDE_DIR /usr/local/cuda/targets/aarch64-linux/include)
    set(CUDA_LIB_DIR /usr/local/cuda/targets/aarch64-linux/lib)
    set(TENSORRT_ROOT "/usr/lib/aarch64-linux-gnu")
else()
    message(FATAL_ERROR "Only aarch64 platform is supported by this CMakeLists.txt")
endif()
include_directories(${CUDA_INCLUDE_DIR})
link_directories(${CUDA_LIB_DIR})

# TensorRT
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 工程头文件
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# 源文件
set(SRC_LIST
    ${PROJECT_SOURCE_DIR}/main.cpp
    ${PROJECT_SOURCE_DIR}/yolo.cpp
    ${PROJECT_SOURCE_DIR}/warp_affine.cu
)

# 插件库
add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/warp_affine.cu)
target_link_libraries(myplugins PRIVATE nvinfer cudart)

# 主程序
add_executable(yolo ${SRC_LIST})
target_link_libraries(yolo PRIVATE nvinfer cudart nvinfer_plugin myplugins ${OpenCV_LIBS})

# 可选：安装规则
# install(TARGETS yolo RUNTIME DESTINATION bin)