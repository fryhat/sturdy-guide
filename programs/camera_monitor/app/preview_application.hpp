#pragma once

#include "camera/camera_session.hpp"

#include <cstddef>
#include <string>

namespace camera {

// Runs the display loop on the calling (main) thread. Only this class calls
// cv::imshow / cv::waitKey; frame acquisition stays on the CameraSession
// worker thread, and screenshots are written without holding any lock.
class PreviewApplication {
 public:
  explicit PreviewApplication(std::string window_name);

  // Displays frames until Q or window close. Throws std::runtime_error when
  // the worker stops unexpectedly or a background exception is observed.
  void run(CameraSession& session);

 private:
  void save_screenshot(const cv::Mat& frame);

  std::string window_name_;
  std::size_t capture_number_ = 0;
};

}  // namespace camera
