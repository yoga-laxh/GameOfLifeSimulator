#pragma once
#include <cstdint>
#include <string>

// Keep configuration in one place so benchmarking is reproducible and consistent.
struct Config {
  int width  = 4096;       // Grid width
  int height = 4096;       // Grid height
  int steps  = 200;        // Simulation steps
  int warmup = 10;         // Warmup steps (not timed)
  int runs   = 5;          // Timed runs
  int block  = 16;         // Block dimension (block x block)
  bool verify = true;      // Verify GPU output against CPU
  bool periodic = true;    // Periodic boundary conditions (wrap around edges)

  // kernel choices:
  // "cpu"    : CPU baseline
  // "naive"  : GPU naive global memory kernel
  // "tiled"  : GPU shared-memory tiled kernel
  std::string kernel = "tiled";
};