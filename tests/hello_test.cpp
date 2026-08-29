#include "sturdy_guide/hello.hpp"

#include <iostream>
#include <string>

int main() {
  const std::string actual = sturdy_guide::hello();
  const std::string expected = "Hello, World!";
  if (actual == expected) {
    return 0;
  }

  std::cerr << "expected '" << expected << "', got '" << actual << "'\n";
  return 1;
}
