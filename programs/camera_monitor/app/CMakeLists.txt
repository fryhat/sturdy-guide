add_executable(sturdy_guide_camera_app main.cpp preview_application.cpp)
target_compile_features(sturdy_guide_camera_app PRIVATE cxx_std_17)
target_link_libraries(
  sturdy_guide_camera_app
  PRIVATE sturdy_guide_camera_core
)

if(STURDY_GUIDE_ENABLE_WARNINGS)
  if(MSVC)
    target_compile_options(sturdy_guide_camera_app PRIVATE /W4)
  else()
    target_compile_options(
      sturdy_guide_camera_app
      PRIVATE -Wall -Wextra -Wpedantic
    )
  endif()
endif()

set_target_properties(
  sturdy_guide_camera_app
  PROPERTIES
    OUTPUT_NAME sturdy-guide-camera
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)
