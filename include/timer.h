#pragma once
#include <chrono>

// Lightweight CPU timer for measuring CPU baseline and overall wall time.
struct CpuTimer {
  using clock = std::chrono::high_resolution_clock;
  clock::time_point t0;

  inline void start() { t0 = clock::now(); }

  inline double ms() const {
    auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
};