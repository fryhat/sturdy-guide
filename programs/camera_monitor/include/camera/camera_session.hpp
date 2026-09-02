#pragma once
#include "camera/frame_source.hpp"
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
namespace camera {
class CameraSession {
 public:
  explicit CameraSession(std::unique_ptr<FrameSource> source);
  ~CameraSession();
  CameraSession(const CameraSession&) = delete;
  CameraSession& operator=(const CameraSession&) = delete;
  void start();
  void stop() noexcept;
  void join();
  std::optional<cv::Mat> latest_frame();
  bool wait_for_frame(std::chrono::milliseconds timeout);
  std::exception_ptr exception() const;
  std::size_t buffered_frames() const;
 private:
  void capture_loop();
  std::unique_ptr<FrameSource> source_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::vector<cv::Mat> buffer_;
  std::exception_ptr exception_;
  std::thread worker_;
  bool stopping_ = false;
  bool running_ = false;
};
}  // namespace camera
