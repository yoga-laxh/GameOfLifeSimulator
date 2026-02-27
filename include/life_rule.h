#pragma once
#include <cstdint>

// Pure rule logic: given current state and live neighbor count, compute next state.
// This mirrors Conway's Game of Life rules.
__host__ __device__ inline uint8_t next_state(uint8_t alive, int live_neighbors) {
  // If alive: survives with 2 or 3 neighbors.
  // If dead : becomes alive with exactly 3 neighbors.
  if (alive) {
    return (live_neighbors == 2 || live_neighbors == 3) ? 1u : 0u;
  } else {
    return (live_neighbors == 3) ? 1u : 0u;
  }
}