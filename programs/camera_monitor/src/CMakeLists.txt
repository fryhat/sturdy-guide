add_library(sturdy_guide_camera_core
  camera_session.cpp
  fake_frame_source.cpp
  open_cv_camera.cpp
)
target_compile_features(sturdy_guide_camera_core PUBLIC cxx_std_17)
target_include_directories(sturdy_guide_camera_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(sturdy_guide_camera_core PUBLIC ${OpenCV_LIBS} Threads::Threads)

if(STURDY_GUIDE_ENABLE_WARNINGS)
  if(MSVC)
    target_compile_options(sturdy_guide_camera_core PRIVATE /W4)
  else()
    target_compile_options(
      sturdy_guide_camera_core
      PRIVATE -Wall -Wextra -Wpedantic
    )
  endif()
endif()
