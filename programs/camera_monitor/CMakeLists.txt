# 作业把单线程起始程序重构为三个独立 target：核心库、可执行应用和无硬件测试。
find_package(OpenCV 4 REQUIRED COMPONENTS core videoio highgui imgproc imgcodecs)
find_package(Threads REQUIRED)
add_subdirectory(src)
add_subdirectory(app)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
