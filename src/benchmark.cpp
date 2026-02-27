#include "grid.h"
#include <cstdio>
#include <cstdint>

// Compare two grids; print mismatch summary.
bool verify_equal(const Grid& a, const Grid& b) {
  if (a.w != b.w || a.h != b.h) return false;

  size_t mismatches = 0;
  for (size_t i = 0; i < a.cells.size(); ++i) {
    if (a.cells[i] != b.cells[i]) {
      ++mismatches;
      if (mismatches < 10) {
        std::printf("Mismatch at idx=%zu (a=%u, b=%u)\n",
                    i, (unsigned)a.cells[i], (unsigned)b.cells[i]);
      }
    }
  }

  if (mismatches == 0) {
    std::printf("VERIFY: PASS\n");
    return true;
  } else {
    std::printf("VERIFY: FAIL mismatches=%zu\n", mismatches);
    return false;
  }
}