#include "camera/fake_frame_source.hpp"
#include <opencv2/core.hpp>
#include <stdexcept>
#include <thread>
namespace camera {
FakeFrameSource::FakeFrameSource(const std::size_t frame_count,
                                 const std::chrono::milliseconds delay)
    : frame_count_(frame_count), delay_(delay) {}
bool FakeFrameSource::read(cv::Mat& frame) {
  if (delay_.count() > 0) std::this_thread::sleep_for(delay_);
  if (reads_ == fail_after_) throw std::runtime_error("fake frame source failure");
  if (frame_count_ != 0 && reads_ >= frame_count_) { frame.release(); return false; }
  ++reads_;
  if (empty_frame_) frame.release();
  else frame = cv::Mat::ones(2, 2, CV_8UC1) * static_cast<int>(reads_ % 255);
  return true;
}
void FakeFrameSource::set_fail_after(const std::size_t successful_reads) { fail_after_ = successful_reads; }
void FakeFrameSource::set_empty_frame(const bool enabled) { empty_frame_ = enabled; }
std::size_t FakeFrameSource::reads() const noexcept { return reads_; }
}  // namespace camera
