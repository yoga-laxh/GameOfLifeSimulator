#pragma once
#include <vector>
#include <cstdint>
#include <cassert>

// Simple grid container for Game of Life.
// We store cells as uint8_t (0 = dead, 1 = alive).
struct Grid {
  int w = 0;
  int h = 0;
  std::vector<uint8_t> cells;

  Grid() = default;

  Grid(int width, int height)
      : w(width), h(height), cells(static_cast<size_t>(w) * static_cast<size_t>(h), 0u) {}

  inline int idx(int x, int y) const {
    // Flatten (x,y) into 1D index: row-major layout.
    return y * w + x;
  }

  inline uint8_t get(int x, int y) const {
    assert(x >= 0 && x < w && y >= 0 && y < h);
    return cells[static_cast<size_t>(idx(x, y))];
  }

  inline void set(int x, int y, uint8_t v) {
    assert(x >= 0 && x < w && y >= 0 && y < h);
    cells[static_cast<size_t>(idx(x, y))] = v;
  }
};