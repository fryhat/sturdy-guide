# 起始版本只有一个单线程可执行程序。作业要求学生把它重构为库、应用和测试。
find_package(OpenCV 4 REQUIRED COMPONENTS core)
find_package(Threads REQUIRED)
add_subdirectory(src)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
