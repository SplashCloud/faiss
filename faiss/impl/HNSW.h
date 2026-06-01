/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/impl/maybe_owned_vector.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "pq.h"

namespace faiss {

/** Implementation of the Hierarchical Navigable Small World
 * datastructure.
 *
 * Efficient and robust approximate nearest neighbor search using
 * Hierarchical Navigable Small World graphs
 *
 *  Yu. A. Malkov, D. A. Yashunin, arXiv 2017
 *
 * This implementation is heavily influenced by the NMSlib
 * implementation by Yury Malkov and Leonid Boystov
 * (https://github.com/searchivarius/nmslib)
 *
 * The HNSW object stores only the neighbor link structure, see
 * IndexHNSW.h for the full index object.
 */

struct VisitedTable;
struct DistanceComputer; // from AuxIndexStructures
struct HNSWStats;
template <class C>
struct ResultHandler;

struct SearchParametersHNSW : SearchParameters {
    int efSearch = 16;
    bool check_relative_distance = true;
    bool bounded_queue = true;

    // Batch processing and beam size
    int beam_size = 1;  // Beam size for beam search (1 = original)
    int batch_size = 0; // Batch size for neighbor processing (0 = no batching)

    // PQ-instructed pruning
    float pq_pruning_ratio = 0; // Ratio of candidates to select via PQ

    bool local_prune = false;
    float send_neigh_times_ratio = 0;

    int zmq_port = 5557;
    //     bool cache_distances = false;

    ~SearchParametersHNSW() {}
};

class IndexHNSW;
struct HNSW {
    /// internal storage of vectors (32 bits: this is expensive)
    using storage_idx_t = int32_t;

    int zmq_port = 5557;

    struct Level0EdgeLocation {
        storage_idx_t node_id;
        size_t neighbor_array_index; // Index in the flat neighbors array

        Level0EdgeLocation(int n, size_t idx)
                : node_id(n), neighbor_array_index(idx) {}
    };

    // Add a struct to store percentile-to-distance mapping
    struct PercentileThreshold {
        float percentile; // Percentile value (e.g., 0.5, 1.0, etc.)
        float threshold;  // Distance threshold at this percentile

        PercentileThreshold(float p, float t) : percentile(p), threshold(t) {}
    };

    // Store the percentile thresholds
    std::vector<PercentileThreshold> percentile_thresholds;

    // Method to set the percentile thresholds with predefined values
    void set_percentile_thresholds() {
        percentile_thresholds.clear();
        // Add the predefined percentile-threshold pairs
        percentile_thresholds.emplace_back(0.5f, -2.801647f);
        percentile_thresholds.emplace_back(1.0f, -2.708753f);
        percentile_thresholds.emplace_back(2.0f, -2.625265f);
        percentile_thresholds.emplace_back(3.0f, -2.576904f);
        percentile_thresholds.emplace_back(5.0f, -2.515250f);
        percentile_thresholds.emplace_back(8.0f, -2.451823f);
        percentile_thresholds.emplace_back(10.0f, -2.417586f);
        percentile_thresholds.emplace_back(15.0f, -2.345697f);
        percentile_thresholds.emplace_back(20.0f, -2.287264f);
        percentile_thresholds.emplace_back(30.0f, -2.191441f);
        percentile_thresholds.emplace_back(40.0f, -2.104487f);
        percentile_thresholds.emplace_back(50.0f, -2.012791f);
        percentile_thresholds.emplace_back(60.0f, -1.909652f);
        percentile_thresholds.emplace_back(70.0f, -1.803186f);
    }

    // Method to get the threshold for a specific percentile
    float get_threshold_for_percentile(float percentile) const {
        for (const auto& pt : percentile_thresholds) {
            if (std::abs(pt.percentile - percentile) < 0.001f) {
                return pt.threshold;
            }
        }
        return 0.0f; // Default if not found
    }

    void delete_random_level0_edges_minimal(float prune_ratio = 0.5);
    void merge_nodes(float merge_threshold);
    //     void save_edge_stats(const char* filename) ;

    // for now we do only these distances
    using C = CMax<float, int64_t>;

    typedef std::pair<float, storage_idx_t> Node;

    /** Heap structure that allows fast
     */
    struct MinimaxHeap {
        int n;
        int k;
        int nvalid;

        std::vector<storage_idx_t> ids;
        std::vector<float> dis;
        typedef faiss::CMax<float, storage_idx_t> HC;

        explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}

        void push(storage_idx_t i, float v);

        float max() const;

        int size() const;

        void clear();

        int pop_min(float* vmin_out = nullptr);

        int count_below(float thresh);
    };

    /// to sort pairs of (id, distance) from nearest to fathest or the reverse
    struct NodeDistCloser {
        float d;
        int id;
        NodeDistCloser(float d, int id) : d(d), id(id) {}
        bool operator<(const NodeDistCloser& obj1) const {
            return d < obj1.d;
        }
    };

    struct NodeDistFarther {
        float d;
        int id;
        NodeDistFarther(float d, int id) : d(d), id(id) {}
        bool operator<(const NodeDistFarther& obj1) const {
            return d > obj1.d;
        }
    };

    /// assignment probability to each layer (sum=1)
    std::vector<double> assign_probas;

    /// number of neighbors stored per layer (cumulative), should not
    /// be changed after first add
    std::vector<int> cum_nneighbor_per_level;

    /// level of each vector (base level = 1), size = ntotal
    std::vector<int> levels;

    /// offsets[i] is the offset in the neighbors array where vector i is stored
    /// size ntotal + 1
    std::vector<size_t> offsets;

    /// neighbors[offsets[i]:offsets[i+1]] is the list of neighbors of vector i
    /// for all levels. this is where all storage goes.
    MaybeOwnedVector<storage_idx_t> neighbors;

    // --- Compact CSR Storage (New) ---
    bool storage_is_compact = false; // Flag read from file
    MaybeOwnedVector<storage_idx_t> compact_neighbors_data; // CSR data
    MaybeOwnedVector<size_t> compact_level_ptr;             // CSR indptr part 1
    MaybeOwnedVector<size_t> compact_node_offsets;          // CSR indptr part 2

    /// entry point in the search structure (one of the points with maximum
    /// level
    storage_idx_t entry_point = -1;

    faiss::RandomGenerator rng;

    /// maximum level
    int max_level = -1;

    /// expansion factor at construction time
    int efConstruction = 40;

    /// expansion factor at search time
    int efSearch = 16;

    bool neighbors_on_disk = false;
    std::string hnsw_index_filename; // Used for pread
    int graph_fd = -1;               // File descriptor for pread

    // --- New members for disk/mmap access ---
    off_t neighbors_start_offset = -1; // For pread: offset to size field; For
                                       // mmap: relative offset (debug)
    storage_idx_t* neighbors_mmap_ptr =
            nullptr; // For mmap: pointer to neighbor data in memory
    bool neighbors_use_mmap =
            false; // Whether to use mmap pointer instead of pread

    /// during search: do we check whether the next best distance is good
    /// enough?
    bool check_relative_distance = true;

    /// use bounded queue during exploration
    bool search_bounded_queue = true;

    /// optional knobs to speed up incremental add when embeddings are fetched
    /// remotely; disabling RNG reduces distance recomputations at the cost of
    /// graph quality. Exposed mainly for benchmarking.
    bool disable_rng_during_add = false;
    bool disable_reverse_prune = false;

    void set_disable_rng_during_add(bool flag) {
        disable_rng_during_add = flag;
        std::fprintf(
                stderr,
                "[HNSW RNG] set_disable_rng_during_add requested=%d\n",
                flag ? 1 : 0);
    }

    void set_disable_reverse_prune(bool flag) {
        disable_reverse_prune = flag;
        std::fprintf(
                stderr,
                "[HNSW RNG] set_disable_reverse_prune requested=%d\n",
                flag ? 1 : 0);
    }

    // methods that initialize the tree sizes

    /// initialize the assign_probas and cum_nneighbor_per_level to
    /// have 2*M links on level 0 and M links on levels > 0
    void set_default_probas(int M, float levelMult, int M0 = -1);

    /// set nb of neighbors for this level (before adding anything)
    void set_nb_neighbors(int level_no, int n);

    // methods that access the tree sizes

    /// nb of neighbors for this level
    int nb_neighbors(int layer_no) const;

    /// cumumlative nb up to (and excluding) this level
    int cum_nb_neighbors(int layer_no) const;

    /// range of entries in the neighbors table of vertex no at layer_no
    void neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
            const;

    /// only mandatory parameter: nb of neighbors
    explicit HNSW(int M = 32, int M0 = -1);

    /// pick a random level for a new point
    int random_level();

    /// add n random levels to table (for debugging...)
    void fill_with_random_links(size_t n);

    void add_links_starting_from(
            DistanceComputer& ptdis,
            storage_idx_t pt_id,
            storage_idx_t nearest,
            float d_nearest,
            int level,
            omp_lock_t* locks,
            VisitedTable& vt,
            bool keep_max_size_level0 = false);

    /** add point pt_id on all levels <= pt_level and build the link
     * structure for them. */
    void add_with_locks(
            DistanceComputer& ptdis,
            int pt_level,
            int pt_id,
            std::vector<omp_lock_t>& locks,
            VisitedTable& vt,
            bool keep_max_size_level0 = false);

    /// search interface for 1 point, single thread
    HNSWStats search(
            DistanceComputer& qdis,
            ResultHandler<C>& res,
            VisitedTable& vt,
            const SearchParameters* params = nullptr,
            const IndexHNSW* hnsw = nullptr) const;

    /// search only in level 0 from a given vertex
    void search_level_0(
            DistanceComputer& qdis,
            ResultHandler<C>& res,
            idx_t nprobe,
            const storage_idx_t* nearest_i,
            const float* nearest_d,
            int search_type,
            HNSWStats& search_stats,
            VisitedTable& vt,
            const SearchParameters* params = nullptr) const;

    void reset();

    void clear_neighbor_tables(int level);
    void print_neighbor_stats(int level) const;

    int prepare_level_tab(size_t n, bool preset_levels = false);

    static void shrink_neighbor_list(
            DistanceComputer& qdis,
            std::priority_queue<NodeDistFarther>& input,
            std::vector<NodeDistFarther>& output,
            int max_size,
            bool keep_max_size_level0 = false);

    void permute_entries(const idx_t* map);

    void save_degree_distribution(int level, const char* filename) const;

    float pq_pruning_ratio = 0;

    std::shared_ptr<PQPrunerDataLoader> pq_data_loader;
    std::vector<uint8_t> pq_codes; // PQ codes of all vectors (N * code_size)
    size_t code_size = 0;          // number of chunks per vector

    bool load_pq_pruning_data(
            const std::string& pq_pivots_path,
            const std::string& pq_compressed_path);

    bool pq_loaded = false;

    // On-demand neighbor fetch method
    size_t fetch_neighbors(
            idx_t node_id,
            int level,
            std::vector<storage_idx_t>& buffer) const;

    void initialize_graph(const std::string& index_filename);

    ~HNSW(); // Close file descriptor
    std::vector<int> ems;

    void set_zmq_port(int port) {
        zmq_port = port;
    }

    int get_zmq_port() const {
        return zmq_port;
    }
};

struct HNSWStats {
    struct CandidateTrace {
        size_t hop_id = 0;
        idx_t candidate_id = -1;
        idx_t parent_id = -1;
        size_t candidate_degree = 0;
        float pq_distance = -1.0f;
        int pq_rank = -1;
        float exact_distance = 0.0f;
        bool was_selected_for_recompute = false;
        bool was_recomputed = false;
    };

    size_t n1 = 0; /// number of vectors searched
    size_t n2 =
            0; /// number of queries for which the candidate list is exhausted
    size_t ndis = 0;   /// number of distances computed
    size_t nhops = 0;  /// number of hops aka number of edges traversed
    size_t npq = 0;    /// number of PQ candidates
    size_t nfetch = 0; /// number of neighbors fetched
    size_t n_ios = 0;  /// number of neighbors fetched on demand
    size_t n_pq_calcs = 0;

    double total_ms = 0.0;
    double distance_computer_setup_ms = 0.0;
    double upper_greedy_ms = 0.0;
    double level0_total_ms = 0.0;
    double level0_pop_fetch_ms = 0.0;
    double level0_dedupe_ms = 0.0;
    double level0_pq_ms = 0.0;
    double level0_exact_distance_ms = 0.0;
    double level0_heap_update_ms = 0.0;
    double postprocess_ms = 0.0;

    size_t upper_distance_batch_calls = 0;
    size_t upper_requested_nodes_total = 0;
    size_t upper_batch_size_bins[5] = {0, 0, 0, 0, 0};
    std::unordered_set<idx_t> upper_requested_nodes_unique;

    size_t level0_distance_batch_calls = 0;
    size_t level0_requested_nodes_total = 0;
    size_t level0_batch_size_bins[5] = {0, 0, 0, 0, 0};
    std::unordered_set<idx_t> level0_requested_nodes_unique;
    size_t level0_iterations = 0;
    size_t level0_beam_pops = 0;
    size_t level0_neighbors_seen_total = 0;
    size_t level0_unique_neighbors_seen_total = 0;
    size_t level0_recompute_selected_total = 0;

    size_t zmq_distance_requests = 0;
    size_t zmq_distance_nodes_total = 0;
    double zmq_pack_ms = 0.0;
    double zmq_connect_ms = 0.0;
    double zmq_send_ms = 0.0;
    double zmq_recv_ms = 0.0;
    double zmq_unpack_ms = 0.0;

    std::vector<idx_t> final_labels;
    std::vector<CandidateTrace> candidate_trace;
    size_t candidate_trace_limit = 0;

    // Track visited node counts
    std::unordered_map<idx_t, size_t> node_visit_counts;

    void reset() {
        n1 = n2 = 0;
        ndis = 0;
        nhops = 0;
        npq = 0;
        nfetch = 0;
        n_ios = 0;
        n_pq_calcs = 0;
        total_ms = 0.0;
        distance_computer_setup_ms = 0.0;
        upper_greedy_ms = 0.0;
        level0_total_ms = 0.0;
        level0_pop_fetch_ms = 0.0;
        level0_dedupe_ms = 0.0;
        level0_pq_ms = 0.0;
        level0_exact_distance_ms = 0.0;
        level0_heap_update_ms = 0.0;
        postprocess_ms = 0.0;
        upper_distance_batch_calls = 0;
        upper_requested_nodes_total = 0;
        std::fill(std::begin(upper_batch_size_bins), std::end(upper_batch_size_bins), 0);
        upper_requested_nodes_unique.clear();
        level0_distance_batch_calls = 0;
        level0_requested_nodes_total = 0;
        std::fill(std::begin(level0_batch_size_bins), std::end(level0_batch_size_bins), 0);
        level0_requested_nodes_unique.clear();
        level0_iterations = 0;
        level0_beam_pops = 0;
        level0_neighbors_seen_total = 0;
        level0_unique_neighbors_seen_total = 0;
        level0_recompute_selected_total = 0;
        zmq_distance_requests = 0;
        zmq_distance_nodes_total = 0;
        zmq_pack_ms = 0.0;
        zmq_connect_ms = 0.0;
        zmq_send_ms = 0.0;
        zmq_recv_ms = 0.0;
        zmq_unpack_ms = 0.0;
        final_labels.clear();
        candidate_trace.clear();
        candidate_trace_limit = 0;
        // printf("Resetting node visit counts\n");
        // printf("Original size: %zu\n", node_visit_counts.size());
        node_visit_counts.clear();
    }

    void combine(const HNSWStats& other) {
        n1 += other.n1;
        n2 += other.n2;
        ndis += other.ndis;
        nhops += other.nhops;
        npq += other.npq;
        nfetch += other.nfetch;
        n_ios += other.n_ios;
        n_pq_calcs += other.n_pq_calcs;
        total_ms += other.total_ms;
        distance_computer_setup_ms += other.distance_computer_setup_ms;
        upper_greedy_ms += other.upper_greedy_ms;
        level0_total_ms += other.level0_total_ms;
        level0_pop_fetch_ms += other.level0_pop_fetch_ms;
        level0_dedupe_ms += other.level0_dedupe_ms;
        level0_pq_ms += other.level0_pq_ms;
        level0_exact_distance_ms += other.level0_exact_distance_ms;
        level0_heap_update_ms += other.level0_heap_update_ms;
        postprocess_ms += other.postprocess_ms;
        upper_distance_batch_calls += other.upper_distance_batch_calls;
        upper_requested_nodes_total += other.upper_requested_nodes_total;
        for (size_t i = 0; i < 5; i++) {
            upper_batch_size_bins[i] += other.upper_batch_size_bins[i];
        }
        upper_requested_nodes_unique.insert(
                other.upper_requested_nodes_unique.begin(),
                other.upper_requested_nodes_unique.end());
        level0_distance_batch_calls += other.level0_distance_batch_calls;
        level0_requested_nodes_total += other.level0_requested_nodes_total;
        for (size_t i = 0; i < 5; i++) {
            level0_batch_size_bins[i] += other.level0_batch_size_bins[i];
        }
        level0_requested_nodes_unique.insert(
                other.level0_requested_nodes_unique.begin(),
                other.level0_requested_nodes_unique.end());
        level0_iterations += other.level0_iterations;
        level0_beam_pops += other.level0_beam_pops;
        level0_neighbors_seen_total += other.level0_neighbors_seen_total;
        level0_unique_neighbors_seen_total +=
                other.level0_unique_neighbors_seen_total;
        level0_recompute_selected_total += other.level0_recompute_selected_total;
        zmq_distance_requests += other.zmq_distance_requests;
        zmq_distance_nodes_total += other.zmq_distance_nodes_total;
        zmq_pack_ms += other.zmq_pack_ms;
        zmq_connect_ms += other.zmq_connect_ms;
        zmq_send_ms += other.zmq_send_ms;
        zmq_recv_ms += other.zmq_recv_ms;
        zmq_unpack_ms += other.zmq_unpack_ms;
        final_labels.insert(
                final_labels.end(), other.final_labels.begin(), other.final_labels.end());
        if (candidate_trace_limit == 0) {
            candidate_trace_limit = other.candidate_trace_limit;
        }
        if (candidate_trace_limit > 0 &&
            candidate_trace.size() < candidate_trace_limit) {
            size_t remaining = candidate_trace_limit - candidate_trace.size();
            size_t take = std::min(remaining, other.candidate_trace.size());
            candidate_trace.insert(
                    candidate_trace.end(),
                    other.candidate_trace.begin(),
                    other.candidate_trace.begin() + take);
        }

        // Combine node visit counts
        // printf("Two sizes: %zu, %zu\n",
        //        node_visit_counts.size(),
        //        other.node_visit_counts.size());
        for (const auto& [node_id, count] : other.node_visit_counts) {
            node_visit_counts[node_id] += count;
        }
    }

    static size_t batch_bin(size_t batch_size) {
        if (batch_size <= 1) {
            return 0;
        } else if (batch_size <= 4) {
            return 1;
        } else if (batch_size <= 16) {
            return 2;
        } else if (batch_size <= 64) {
            return 3;
        }
        return 4;
    }

    void record_upper_distance_batch(const std::vector<idx_t>& ids) {
        upper_distance_batch_calls++;
        upper_requested_nodes_total += ids.size();
        upper_batch_size_bins[batch_bin(ids.size())]++;
        upper_requested_nodes_unique.insert(ids.begin(), ids.end());
    }

    void record_level0_distance_batch(const std::vector<idx_t>& ids) {
        level0_distance_batch_calls++;
        level0_requested_nodes_total += ids.size();
        level0_batch_size_bins[batch_bin(ids.size())]++;
        level0_requested_nodes_unique.insert(ids.begin(), ids.end());
    }

    void add_candidate_trace(const CandidateTrace& event) {
        if (candidate_trace_limit == 0 ||
            candidate_trace.size() >= candidate_trace_limit) {
            return;
        }
        candidate_trace.push_back(event);
    }

    // Dump node visit frequency distribution to a file
    void dump_node_visit_stats(const char* filename) const {
        FILE* f = fopen(filename, "w");
        if (!f) {
            fprintf(stderr,
                    "Could not open %s for writing: %s\n",
                    filename,
                    strerror(errno));
            return;
        }

        fprintf(f, "node_id,visit_count\n");
        for (const auto& [node_id, count] : node_visit_counts) {
            fprintf(f, "%ld,%ld\n", (long)node_id, (long)count);
        }

        fclose(f);
    }

    // Dump distribution of visit frequencies (how many nodes were visited X
    // times)
    void dump_visit_frequency_distribution(const char* filename) const {
        std::unordered_map<size_t, size_t> frequency_counts;

        // Count how many nodes had each visit count
        for (const auto& [node_id, visits] : node_visit_counts) {
            frequency_counts[visits]++;
        }

        // Write to file
        FILE* f = fopen(filename, "w");
        if (!f) {
            fprintf(stderr,
                    "Could not open %s for writing: %s\n",
                    filename,
                    strerror(errno));
            return;
        }

        fprintf(f, "visit_frequency,node_count,percentage\n");

        // Sort by visit frequency for better readability
        std::vector<std::pair<size_t, size_t>> sorted_counts(
                frequency_counts.begin(), frequency_counts.end());
        std::sort(sorted_counts.begin(), sorted_counts.end());

        size_t total_nodes = node_visit_counts.size();
        for (const auto& [visits, count] : sorted_counts) {
            double percentage = (100.0 * count) / total_nodes;
            fprintf(f,
                    "%ld,%ld,%.4f%%\n",
                    (long)visits,
                    (long)count,
                    percentage);
        }

        fclose(f);
    }
};

// global var that collects them all
FAISS_API extern HNSWStats hnsw_stats;

int search_from_candidates(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler<HNSW::C>& res,
        HNSW::MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in = 0,
        const SearchParameters* params = nullptr,
        const IndexHNSW* hnsw_index = nullptr);

HNSWStats greedy_update_nearest(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        HNSW::storage_idx_t& nearest,
        float& d_nearest);

std::priority_queue<HNSW::Node> search_from_candidate_unbounded(
        const HNSW& hnsw,
        const HNSW::Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        HNSWStats& stats);

void search_neighbors_to_add(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<HNSW::NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        bool reference_version = false);

} // namespace faiss
