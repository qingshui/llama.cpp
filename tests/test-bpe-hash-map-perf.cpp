// Test for flat_hash_map BPE Lookup Performance
// Measures the performance of BPE merge lookup using flat_hash_map

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <functional>
#include <numeric>

#include "../third_party/abseil/absl/container/flat_hash_map.h"

// Test data simulating BPE merges
struct Bigram {
    std::string left;
    std::string right;
    int rank;
};

const int NUM_MERGES = 50000;
const int NUM_LOOKUPS = 10000;
const int NUM_ITERATIONS = 100;

// Hash function for pair<string, string>
struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        size_t h1 = std::hash<std::string>{}(p.first);
        size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// Generate test bigrams
std::vector<Bigram> generate_bigrams(int count) {
    std::vector<Bigram> bigrams;
    bigrams.reserve(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> token_dist(0, 1000);

    for (int i = 0; i < count; i++) {
        Bigram b;
        b.left = "token" + std::to_string(token_dist(gen));
        b.right = "token" + std::to_string(token_dist(gen));
        b.rank = i;
        bigrams.push_back(b);
    }
    return bigrams;
}

// Create flat_hash_map from bigrams
absl::flat_hash_map<std::pair<std::string, std::string>, int>
create_bpe_map(const std::vector<Bigram>& bigrams) {
    absl::flat_hash_map<std::pair<std::string, std::string>, int> map;
    map.reserve(bigrams.size());
    for (const auto& b : bigrams) {
        map[{b.left, b.right}] = b.rank;
    }
    return map;
}

int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     flat_hash_map BPE Lookup Performance Test            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Number of BPE merges: " << NUM_MERGES << "\n";
    std::cout << "  Number of lookups per iteration: " << NUM_LOOKUPS << "\n";
    std::cout << "  Number of iterations: " << NUM_ITERATIONS << "\n\n";

    // Generate test data
    auto bigrams = generate_bigrams(NUM_MERGES);

    // Create map
    std::cout << "Creating flat_hash_map...\n";
    auto flat_map = create_bpe_map(bigrams);
    std::cout << "  flat_hash_map created: " << flat_map.size() << " entries\n\n";

    // Create test queries
    std::vector<std::pair<std::string, std::string>> queries;
    queries.reserve(NUM_LOOKUPS);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> idx_dist(0, NUM_MERGES - 1);

    for (int i = 0; i < NUM_LOOKUPS; i++) {
        int idx = idx_dist(gen);
        queries.push_back({bigrams[idx].left, bigrams[idx].right});
    }

    // Test flat_hash_map - measure total time for all iterations
    std::cout << "Testing flat_hash_map...\n";
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        for (const auto& q : queries) {
            auto it = flat_map.find(q);
            (void)it;  // Suppress unused warning
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double total_ms = total_us / 1000.0;
    double avg_lookup_us = (double)total_us / ((double)NUM_ITERATIONS * (double)NUM_LOOKUPS);
    double avg_lookup_ms = avg_lookup_us / 1000.0;

    // Print results
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Results (absl::flat_hash_map)                        ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "║ Total time (all iterations): " << std::setw(20) << total_ms << " ms ║\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "║ Average lookup time:         " << std::setw(20) << avg_lookup_ms << " ms ║\n";
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "║ Average lookup time (us):    " << std::setw(20) << avg_lookup_us << " us ║\n";
    double throughput_ms = (double)NUM_LOOKUPS / avg_lookup_ms;
    double throughput_s = throughput_ms * 1000.0;
    std::cout << "║ Throughput:                  " << std::setw(20) << throughput_ms << " lookups/ms ║\n";
    std::cout << "║ Total throughput:            " << std::setw(20) << throughput_s << " lookups/s ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Note: flat_hash_map uses contiguous memory layout for better cache performance.\n";
    std::cout << "Expected improvement over unordered_map: 2-5x for lookup operations.\n";

    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Test Complete                                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    return 0;
}
