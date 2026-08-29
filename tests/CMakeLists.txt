# 用辅助函数复用“测试程序 + 库依赖 + CTest 注册”这三个步骤。
function(add_sturdy_guide_test target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE SturdyGuide::sturdy_guide)
  add_test(NAME ${target} COMMAND ${target})
endfunction()

# 构建完成后，CTest 会按这里注册的名称运行三个测试程序。
add_sturdy_guide_test(greeting_test greeting_test.cpp)
add_sturdy_guide_test(hello_test hello_test.cpp)
add_sturdy_guide_test(text_test text_test.cpp)
