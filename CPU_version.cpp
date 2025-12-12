#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <chrono>

const int rows = 20;
const int cols = 50;
const int generations = 50;
const int delay_ms = 200;

std::vector<std::vector<int>> create_grid() {
    std::vector<std::vector<int>> grid(rows, std::vector<int>(cols, 0));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            grid[r][c] = rand() % 2;
    return grid;
}

int count_live_neighbors(const std::vector<std::vector<int>>& grid, int r, int c) {
    int count = 0;
    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) continue;
            int nr = (r + dr + rows) % rows;
            int nc = (c + dc + cols) % cols;
            count += grid[nr][nc];
        }
    }
    return count;
}

std::vector<std::vector<int>> next_generation(const std::vector<std::vector<int>>& grid) {
    std::vector<std::vector<int>> new_grid = grid;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int live = count_live_neighbors(grid, r, c);
            if (grid[r][c] == 1) {
                new_grid[r][c] = (live == 2 || live == 3) ? 1 : 0;
            } else {
                new_grid[r][c] = (live == 3) ? 1 : 0;
            }
        }
    }
    return new_grid;
}

void print_grid(const std::vector<std::vector<int>>& grid) {
    system("clear"); // or "cls" on Windows
    for (auto& row : grid) {
        for (auto cell : row)
            std::cout << (cell ? "o" : ".");
        std::cout << "\n";
    }
}

int main() {
    srand(time(0));
    auto grid = create_grid();
    auto start = std::chrono::high_resolution_clock::now();


    for (int gen = 0; gen < generations; gen++) {
        print_grid(grid);
        grid = next_generation(grid);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
    auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = end - start;

std::cout << "CPU Time: " << elapsed.count() << " seconds\n";
    return 0;
}
