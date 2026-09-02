add_library(sturdy_guide_camera_core camera_session.cpp fake_frame_source.cpp)
target_compile_features(sturdy_guide_camera_core PUBLIC cxx_std_17)
target_include_directories(sturdy_guide_camera_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(sturdy_guide_camera_core PUBLIC ${OpenCV_LIBS} Threads::Threads)
