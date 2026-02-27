#include "grid.h"
#include "life_rule.h"
#include <algorithm>

// Helper for periodic wrap.
// If periodic=true, wrap edges around (torus). Otherwise clamp.
static inline int wrap_or_clamp(int v, int lo, int hi, bool periodic) {
  if (periodic) {
    int range = hi - lo + 1;
    int x = (v - lo) % range;
    if (x < 0) x += range;
    return lo + x;
  } else {
    return std::max(lo, std::min(hi, v));
  }
}

// CPU step: compute next grid from current grid.
void cpu_step(const Grid& cur, Grid& nxt, bool periodic) {
  // For each cell: count 8 neighbors and apply rules.
  for (int y = 0; y < cur.h; ++y) {
    for (int x = 0; x < cur.w; ++x) {
      int live = 0;

      // Visit neighbors (3x3 minus center).
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;

          int nx = wrap_or_clamp(x + dx, 0, cur.w - 1, periodic);
          int ny = wrap_or_clamp(y + dy, 0, cur.h - 1, periodic);
          live += (cur.cells[static_cast<size_t>(ny * cur.w + nx)] != 0);
        }
      }

      uint8_t alive = cur.cells[static_cast<size_t>(y * cur.w + x)];
      nxt.cells[static_cast<size_t>(y * cur.w + x)] = next_state(alive, live);
    }
  }
}