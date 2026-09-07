#include "preview_application.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace std::chrono_literals;

namespace camera {

namespace {

// Poll cadence keeps keyboard and window events responsive while the worker
// thread (possibly blocked on a slow device) still owns the camera read.
constexpr std::chrono::milliseconds kPollInterval{33};

}  // namespace

PreviewApplication::PreviewApplication(std::string window_name)
    : window_name_(std::move(window_name)) {}

void PreviewApplication::run(CameraSession& session) {
  cv::namedWindow(window_name_, cv::WINDOW_NORMAL);

  std::size_t frame_number = 0;
  std::size_t frames_in_window = 0;
  double frames_per_second = 0.0;
  auto rate_started_at = std::chrono::steady_clock::now();

  for (;;) {
    if (std::exception_ptr error = session.exception()) {
      std::rethrow_exception(error);
    }
    if (!session.is_running()) {
      throw std::runtime_error("camera stopped returning valid frames");
    }

    // Returns as soon as a frame arrives, an exception is stored, or the
    // worker finishes; otherwise it polls up to the interval above.
    session.wait_for_frame(kPollInterval);

    if (std::exception_ptr error = session.exception()) {
      std::rethrow_exception(error);
    }
    std::optional<cv::Mat> frame = session.latest_frame();
    if (!frame) {
      continue;
    }

    ++frame_number;
    ++frames_in_window;

    const auto now = std::chrono::steady_clock::now();
    const auto rate_window = now - rate_started_at;
    if (rate_window >= std::chrono::seconds{1}) {
      const auto seconds = std::chrono::duration<double>{rate_window}.count();
      frames_per_second = static_cast<double>(frames_in_window) / seconds;
      frames_in_window = 0;
      rate_started_at = now;
    }

    std::ostringstream overlay;
    overlay << "frame " << frame_number << "  " << std::fixed
            << std::setprecision(1) << frames_per_second << " FPS";
    cv::putText(*frame, overlay.str(), cv::Point{20, 36},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar{40, 230, 90}, 2,
                cv::LINE_AA);
    cv::imshow(window_name_, *frame);

    const int key = cv::waitKey(1) & 0xFF;
    const double visibility =
        cv::getWindowProperty(window_name_, cv::WND_PROP_VISIBLE);
    // Some backends report a negative value when visibility is unsupported;
    // only a non-negative value below one means the window was closed.
    if (key == 'q' || key == 'Q' ||
        (visibility >= 0.0 && visibility < 1.0)) {
      break;
    }
    if (key == 's' || key == 'S') {
      save_screenshot(*frame);
    }
  }

  cv::destroyAllWindows();
}

void PreviewApplication::save_screenshot(const cv::Mat& frame) {
  std::filesystem::create_directories("captures");
  std::filesystem::path filename;
  do {
    ++capture_number_;
    filename = std::filesystem::path{"captures"} /
               ("capture-" + std::to_string(capture_number_) + ".png");
  } while (std::filesystem::exists(filename));

  if (!cv::imwrite(filename.string(), frame)) {
    throw std::runtime_error("failed to save " + filename.string());
  }
  std::cout << "Saved " << filename << '\n';
}

}  // namespace camera
