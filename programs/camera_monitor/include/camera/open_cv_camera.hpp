#pragma once

#include "camera/frame_source.hpp"

#include <opencv2/videoio.hpp>

namespace camera {

// FrameSource backed by a real cv::VideoCapture device. The device is opened
// eagerly so an unusable camera fails fast, before any worker thread starts.
class OpenCvCamera final : public FrameSource {
 public:
  OpenCvCamera(int device, int width, int height);
  bool read(cv::Mat& frame) override;

 private:
  cv::VideoCapture capture_;
};

}  // namespace camera
