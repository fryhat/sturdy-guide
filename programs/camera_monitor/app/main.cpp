#include "camera/camera_session.hpp"
#include "camera/open_cv_camera.hpp"
#include "preview_application.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Options {
  int device = 0;
  int width = 1280;
  int height = 720;
  bool show_help = false;
};

Options parse_options(const int argc, const char* const argv[]) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--device INDEX] [--width PIXELS] [--height PIXELS]\n"
                << "Keys: S saves a frame; Q exits.\n";
      options.show_help = true;
      return options;
    }

    if (argument != "--device" && argument != "--width" &&
        argument != "--height") {
      throw std::invalid_argument("unknown option: " +
                                  std::string{argument});
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument(std::string{argument} +
                                  " requires an integer value");
    }

    const int value = std::stoi(argv[++index]);
    if (argument == "--device") {
      options.device = value;
    } else if (argument == "--width") {
      options.width = value;
    } else {
      options.height = value;
    }
  }

  if (options.device < 0 || options.width <= 0 || options.height <= 0) {
    throw std::invalid_argument(
        "device must be non-negative and dimensions must be positive");
  }
  return options;
}

}  // namespace

int main(const int argc, const char* const argv[]) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.show_help) {
      return 0;
    }

    // Dependency assembly: a real device is wrapped by the same FrameSource
    // interface that FakeFrameSource implements, so CameraSession never needs
    // to know which kind of source produced each frame.
    auto source = std::make_unique<camera::OpenCvCamera>(
        options.device, options.width, options.height);
    camera::CameraSession session(std::move(source));
    session.start();

    constexpr std::string_view window_name = "Sturdy Guide Camera";
    camera::PreviewApplication preview{std::string{window_name}};
    preview.run(session);

    // Explicit teardown mirrors the order documented in README.md; the
    // CameraSession destructor performs the same stop/join as a safety net.
    session.stop();
    session.join();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "camera: " << error.what() << '\n';
    return 1;
  }
}
