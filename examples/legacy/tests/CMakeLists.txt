# 早期学习用例（不属于主库；如需构建，把本目录作为子目录 add_subdirectory 即可）
# 添加可执行文件
add_executable(test_loop_unrolling ${CMAKE_CURRENT_SOURCE_DIR}/test_loop_unrolling.cpp)
add_executable(test_stream_asynchronize ${CMAKE_CURRENT_SOURCE_DIR}/test_stream_asynchronize.cu)

# 添加测试
add_test(NAME TestLoopUnrolling COMMAND test_loop_unrolling)
add_test(NAME TestStreamAsynchronize COMMAND test_stream_asynchronize)
