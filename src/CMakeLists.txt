# 添加可执行文件
add_executable(erode_dilation ${CMAKE_SOURCE_DIR}/src/erode-dilation/erode_dilation.cu)
add_executable(integral_image ${CMAKE_SOURCE_DIR}/src/integral-image/integral_image.cu)

# 包含目录
target_include_directories(erode_dilation PRIVATE g:/cuda-samples/common)
target_include_directories(integral_image PRIVATE ${CMAKE_SOURCE_DIR}/src/integral-image)
