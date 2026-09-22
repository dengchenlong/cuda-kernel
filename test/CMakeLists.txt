# 添加可执行文件
add_executable(test_loop_unrolling ${CMAKE_SOURCE_DIR}/test/test_loop_unrolling.cpp)
add_executable(test_stream_asynchronize ${CMAKE_SOURCE_DIR}/test/test_stream_asynchronize.cu)

# 添加测试
add_test(NAME TestLoopUnrolling COMMAND test_loop_unrolling)
add_test(NAME TestStreamAsynchronize COMMAND test_stream_asynchronize)
