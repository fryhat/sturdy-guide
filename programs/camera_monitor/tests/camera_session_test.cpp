#include "camera/camera_session.hpp"
#include "camera/fake_frame_source.hpp"
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace std::chrono_literals;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
int main() {
  try {
    auto source = std::make_unique<camera::FakeFrameSource>(8, 2ms);
    camera::CameraSession session(std::move(source));
    session.start(); require(session.wait_for_frame(500ms), "no frame produced");
    require(session.buffered_frames() <= 2, "buffer exceeded two frames"); session.stop(); session.stop(); session.join();
    require(session.latest_frame().has_value(), "latest frame missing");
    bool repeated_start_threw = false;
    {
      auto running_source = std::make_unique<camera::FakeFrameSource>(3, 2ms);
      camera::CameraSession running(std::move(running_source));
      running.start();
      try { running.start(); } catch (const std::logic_error&) { repeated_start_threw = true; }
      running.stop(); running.join();
    }
    require(repeated_start_threw, "repeated start while running must throw");
    // After stop()+join() the session is idle again and may be restarted.
    bool restart_ok = true;
    try { session.start(); } catch (const std::logic_error&) { restart_ok = false; }
    require(restart_ok, "restart after join should be allowed");
    session.stop(); session.join();
    auto failing_source = std::make_unique<camera::FakeFrameSource>(0, 1ms);
    failing_source->set_fail_after(2);
    camera::CameraSession failed(std::move(failing_source));
    failed.start(); failed.join(); require(failed.exception() != nullptr, "background exception not observable");
    auto empty_source = std::make_unique<camera::FakeFrameSource>(1);
    empty_source->set_empty_frame(true);
    camera::CameraSession empty(std::move(empty_source));
    empty.start(); empty.join(); require(empty.buffered_frames() <= 1, "invalid buffer state");
    require(empty.buffered_frames() == 0, "empty frame was published");
    std::cout << "camera session tests passed\n"; return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
