#include "sturdy_guide/add.hpp"

#include <iostream>
#include <string_view>

namespace {

bool expect_equal(const std::string_view case_name, const int actual,
                  const int expected) {
  if (actual == expected) {
    return true;
  }

  std::cerr << case_name << ": expected " << expected << ", got " << actual
            << '\n';
  return false;
}

}  // namespace

int main() {
  bool passed = true;
  passed &= expect_equal("positive numbers", sturdy_guide::add(2, 3), 5);
  passed &= expect_equal("negative numbers", sturdy_guide::add(-2, -3), -5);
  passed &= expect_equal("mixed signs", sturdy_guide::add(-4, 6), 2);
  passed &= expect_equal("zero", sturdy_guide::add(0, 7), 7);
  return passed ? 0 : 1;
}