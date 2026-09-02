#pragma once
#include <opencv2/core/mat.hpp>
namespace camera {
class FrameSource {
 public:
  virtual ~FrameSource() = default;
  virtual bool read(cv::Mat& frame) = 0;
};
}  // namespace camera
