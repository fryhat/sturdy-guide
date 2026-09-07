#include "camera/open_cv_camera.hpp"

#include <stdexcept>

namespace camera {

OpenCvCamera::OpenCvCamera(const int device, const int width, const int height)
    : capture_{device, cv::CAP_ANY} {
  if (!capture_.isOpened()) {
    throw std::runtime_error(
        "cannot open camera; check the device index, permissions, and whether "
        "another program is using it");
  }
  capture_.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(width));
  capture_.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(height));
}

bool OpenCvCamera::read(cv::Mat& frame) { return capture_.read(frame); }

}  // namespace camera
