#include "camera/camera_session.hpp"
#include <stdexcept>
namespace camera {
CameraSession::CameraSession(std::unique_ptr<FrameSource> source) : source_(std::move(source)) {
  if (!source_) throw std::invalid_argument("CameraSession requires a frame source");
}
CameraSession::~CameraSession() { stop(); join(); }
void CameraSession::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_ || worker_.joinable()) throw std::logic_error("CameraSession already started");
  buffer_.clear(); exception_ = nullptr; stopping_ = false; running_ = true;
  worker_ = std::thread(&CameraSession::capture_loop, this);
}
void CameraSession::stop() noexcept {
  { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
  condition_.notify_all();
}
void CameraSession::join() { if (worker_.joinable()) worker_.join(); }
std::optional<cv::Mat> CameraSession::latest_frame() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (buffer_.empty()) return std::nullopt;
  cv::Mat frame = buffer_.back().clone(); buffer_.clear(); return frame;
}
bool CameraSession::wait_for_frame(const std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  return condition_.wait_for(lock, timeout, [this] { return !buffer_.empty() || exception_ || !running_; });
}
std::exception_ptr CameraSession::exception() const { std::lock_guard<std::mutex> lock(mutex_); return exception_; }
std::size_t CameraSession::buffered_frames() const { std::lock_guard<std::mutex> lock(mutex_); return buffer_.size(); }
bool CameraSession::is_running() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return running_;
}
void CameraSession::capture_loop() {
  try {
    while (true) {
      { std::lock_guard<std::mutex> lock(mutex_); if (stopping_) break; }
      cv::Mat frame;
      if (!source_->read(frame)) break;
      if (frame.empty()) continue;
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) break;
      if (buffer_.size() >= 2) buffer_.erase(buffer_.begin());
      buffer_.push_back(std::move(frame)); condition_.notify_all();
    }
  } catch (...) { std::lock_guard<std::mutex> lock(mutex_); exception_ = std::current_exception(); }
  { std::lock_guard<std::mutex> lock(mutex_); running_ = false; }
  condition_.notify_all();
}
}  // namespace camera
