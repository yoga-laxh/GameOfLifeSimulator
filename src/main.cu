#include "config.h"
#include "grid.h"
#include <cstdio>
#include <cstdlib>
#include <string>

// Forward decls.
void init_random(Grid& g, uint32_t seed);
double run_cpu(const Config& cfg, Grid& cur, Grid& nxt);
float run_gpu(const Config& cfg, Grid& cur_host, Grid& out_host);
void print_perf(const Config& cfg, double cpu_ms, float gpu_ms);
bool verify_equal(const Grid& a, const Grid& b);

// Minimal CLI parser.
static void usage() {
  std::printf(
    "Usage: ./gol [--w N] [--h N] [--steps N] [--kernel cpu|naive|tiled]\n"
    "            [--block 8|16|32] [--runs N] [--warmup N]\n"
    "            [--verify 0|1] [--periodic 0|1]\n"
  );
}

static int to_int(const char* s) { return std::atoi(s); }

int main(int argc, char** argv) {
  Config cfg;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) { std::printf("Missing value for %s\n", name); usage(); std::exit(1); }
      return argv[++i];
    };

    if (a == "--w") cfg.width = to_int(need("--w"));
    else if (a == "--h") cfg.height = to_int(need("--h"));
    else if (a == "--steps") cfg.steps = to_int(need("--steps"));
    else if (a == "--kernel") cfg.kernel = need("--kernel");
    else if (a == "--block") cfg.block = to_int(need("--block"));
    else if (a == "--runs") cfg.runs = to_int(need("--runs"));
    else if (a == "--warmup") cfg.warmup = to_int(need("--warmup"));
    else if (a == "--verify") cfg.verify = (to_int(need("--verify")) != 0);
    else if (a == "--periodic") cfg.periodic = (to_int(need("--periodic")) != 0);
    else if (a == "--help") { usage(); return 0; }
    else { std::printf("Unknown arg: %s\n", a.c_str()); usage(); return 1; }
  }

  // Input grid + working buffers.
  Grid init(cfg.width, cfg.height);
  init_random(init, 1234);

  Grid cpu_cur = init;
  Grid cpu_nxt(cfg.width, cfg.height);

  // Always compute CPU baseline once for comparison (and optional verification).
  double cpu_ms = run_cpu(cfg, cpu_cur, cpu_nxt);

  if (cfg.kernel == "cpu") {
    std::printf("CPU-only run complete.\n");
    std::printf("CPU time: %.3f ms\n", cpu_ms);
    return 0;
  }

  // Run GPU from same initial state.
  Grid gpu_out;
  float gpu_ms = run_gpu(cfg, init, gpu_out);

  print_perf(cfg, cpu_ms, gpu_ms);

  if (cfg.verify) {
    // cpu_cur currently holds final CPU state after run_cpu() (because we swapped buffers).
    // So compare CPU final vs GPU final.
    verify_equal(cpu_cur, gpu_out);
  }

  return 0;
}