#pragma once
#include "camera/frame_source.hpp"
#include <chrono>
#include <cstddef>
namespace camera {
class FakeFrameSource final : public FrameSource {
 public:
  explicit FakeFrameSource(std::size_t frame_count = 0,
                           std::chrono::milliseconds delay = {});
  bool read(cv::Mat& frame) override;
  void set_fail_after(std::size_t successful_reads);
  void set_empty_frame(bool enabled);
  std::size_t reads() const noexcept;
 private:
  std::size_t frame_count_;
  std::chrono::milliseconds delay_;
  std::size_t reads_ = 0;
  std::size_t fail_after_ = static_cast<std::size_t>(-1);
  bool empty_frame_ = false;
};
}  // namespace camera
