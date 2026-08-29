# src/ 负责可复用库 target 及其构建要求。
add_library(sturdy_guide
  greeting.cpp
  hello.cpp
  text.cpp
)
add_library(SturdyGuide::sturdy_guide ALIAS sturdy_guide)

# PUBLIC 表示 C++17 和头文件路径会传递给链接本库的应用。
target_compile_features(sturdy_guide PUBLIC cxx_std_17)
target_include_directories(
  sturdy_guide
  PUBLIC
    # 源码构建读取项目 include/；安装后则读取安装前缀中的 include/。
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(STURDY_GUIDE_ENABLE_WARNINGS)
  if(MSVC)
    target_compile_options(sturdy_guide PRIVATE /W4)
  else()
    target_compile_options(sturdy_guide PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

# 导出 target 后，安装包才能为使用者提供 SturdyGuide::sturdy_guide。
install(
  TARGETS sturdy_guide
  EXPORT SturdyGuideTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
