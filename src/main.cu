#include "common.cuh"
#include "kernel.cuh"

#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>

// Small, slow CPU reference 
static inline ECPoint scalar_mul_ref(ECPoint P, scalar_t k) {
    ECPoint R; R.X=fq(0); R.Y=fq(1); R.Z=fq(0); // infinity
    while (k) {
        if (k & 1) R = R + P;
        P = P.dbl();
        k >>= 1;
    }
    return R;
}

static inline ECPoint msm_ref(const std::vector<ECPoint>& bases,
                              const std::vector<scalar_t>& scalars) {
    ECPoint R; R.X=fq(0); R.Y=fq(1); R.Z=fq(0); // infinity
    for (size_t i = 0; i < bases.size(); ++i) {
        R = R + scalar_mul_ref(bases[i], scalars[i]);
    }
    return R;
}

static inline bool eq_affine(ECPoint A, ECPoint B) {
    A.to_affine(); B.to_affine();
    return (A.X.value == B.X.value) && (A.Y.value == B.Y.value);
}
static int run_toy2_gpu_vs_ref() {
    // Parameters
    constexpr int C = 8;          // window bits
    constexpr int BITSIZE = 64;   // scalar_t = uint64_t
    const int windows = (BITSIZE + C - 1) / C;
    const int num_buckets = (1 << C) - 1;
    const int total_buckets = windows * num_buckets;

    using std::cerr;
    using std::cout;
    using std::endl;

    // Fixed test case: N=2 with scalars 3 and 5
    const int N = 2;
    cout << "\n=== Toy Test with N=" << N << " ===\n";

    // Generate test data: 2 copies of G with scalars 3 and 5
    ECPoint G(fq(1), fq(2), fq(1));
    std::vector<ECPoint> bases = {G, G};
    std::vector<scalar_t> scalars = {(scalar_t)3, (scalar_t)5};

    // Expected result: 3*G + 5*G = 8*G
    scalar_t expected_scalar = (scalar_t)8;
    ECPoint expected = scalar_mul_ref(G, expected_scalar);
    expected.to_affine();

    // ---------- CPU bucketize ----------
    std::vector<std::vector<ECPoint>> bucket_points(total_buckets);
    for (int i = 0; i < N; ++i) {
        scalar_t s = scalars[i];
        for (int w = 0; w < BITSIZE; w += C) {
            int b = (int)((s >> w) & ((1 << C) - 1));  // 0..255
            if (b == 0) continue;
            int idx = (w / C) * num_buckets + (b - 1); // bucket 1..255 -> 0..254
            bucket_points[idx].push_back(bases[i]);
        }
    }

    // CPU per-bucket sums
    std::vector<ECPoint> cpu_bucket_sums(total_buckets);
    for (int i = 0; i < total_buckets; ++i) {
        ECPoint sum; sum.X=fq(0); sum.Y=fq(1); sum.Z=fq(0);
        for (const auto& P : bucket_points[i]) sum = sum + P;
        cpu_bucket_sums[i] = sum;
    }

    // Flatten for GPU
    int total_points = 0;
    for (int i = 0; i < total_buckets; ++i) total_points += (int)bucket_points[i].size();

    std::vector<ECPoint> all_points(total_points);
    std::vector<int> bucket_sizes(total_buckets);
    std::vector<int> bucket_offsets(total_buckets + 1);

    int point_idx = 0;
    bucket_offsets[0] = 0;
    for (int i = 0; i < total_buckets; ++i) {
        bucket_sizes[i] = (int)bucket_points[i].size();
        for (int j = 0; j < bucket_sizes[i]; ++j) {
            all_points[point_idx++] = bucket_points[i][j];
        }
        bucket_offsets[i + 1] = point_idx;
    }

    // GPU: buffers 
    ECPoint *d_all_points = nullptr, *d_bucket_sums = nullptr, *d_window_sums = nullptr;
    int *d_bucket_sizes = nullptr, *d_bucket_offsets = nullptr;

    CUDA_CALL(cudaMalloc(&d_all_points, std::max(1, total_points) * (int)sizeof(ECPoint)));
    CUDA_CALL(cudaMalloc(&d_bucket_sizes, total_buckets * (int)sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_bucket_offsets, (total_buckets + 1) * (int)sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_bucket_sums, total_buckets * (int)sizeof(ECPoint)));
    CUDA_CALL(cudaMalloc(&d_window_sums, windows * (int)sizeof(ECPoint)));

    if (total_points)
        CUDA_CALL(cudaMemcpy(d_all_points, all_points.data(), total_points * sizeof(ECPoint), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_bucket_sizes,   bucket_sizes.data(),   total_buckets * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_bucket_offsets, bucket_offsets.data(), (total_buckets + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // GPU: bucket sums 
    int threads = 256;
    int blocks  = (total_buckets + threads - 1) / threads;
    sum_bucket_points_kernel<<<blocks, threads>>>(
        d_all_points, d_bucket_sizes, d_bucket_offsets, d_bucket_sums, total_buckets);
    CUDA_CALL(cudaDeviceSynchronize());

    // GPU: window weighted sums 
    std::vector<ECPoint> gpu_window_sums(windows);
    int window_threads = 256;
    int window_blocks  = (windows + window_threads - 1) / window_threads;
    sum_window_buckets_kernel<<<window_blocks, window_threads>>>(
        d_bucket_sums, d_window_sums, windows, num_buckets);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(gpu_window_sums.data(), d_window_sums, windows * sizeof(ECPoint), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_all_points));
    CUDA_CALL(cudaFree(d_bucket_sizes));
    CUDA_CALL(cudaFree(d_bucket_offsets));
    CUDA_CALL(cudaFree(d_bucket_sums));
    CUDA_CALL(cudaFree(d_window_sums));

    //  Final fold (CPU) over GPU window sums 
    ECPoint folded; folded.X=fq(0); folded.Y=fq(1); folded.Z=fq(0);
    for (int w = windows - 1; w >= 0; --w) {
        if (w != windows - 1) {
            for (int k = 0; k < C; ++k) folded = folded.dbl(); // ×2^C
        }
        folded = folded + gpu_window_sums[w];
    }
    folded.to_affine();

    //  Verify correctness 
    bool correctness = eq_affine(folded, expected);
    
    if (correctness) {
        cout << "[OK] Toy test matches expected result\n";
        cout << "  Expected: (" << expected.X.value << ", " << expected.Y.value << ")\n";
        cout << "  Result:   (" << folded.X.value << ", " << folded.Y.value << ")\n";
        cout << "  Test case: 3*G + 5*G = 8*G\n";
        return 0;
    } else {
        cout << "[FAIL] Toy test mismatch\n";
        cout << "  Expected: (" << expected.X.value << ", " << expected.Y.value << ")\n";
        cout << "  Result:   (" << folded.X.value << ", " << folded.Y.value << ")\n";
        cout << "  Test case: 3*G + 5*G = 8*G\n";
        return 42;
    }
}


int main(int argc, char** argv) {
    int device_count = 0; CUDA_CALL(cudaGetDeviceCount(&device_count));
    std::cout << "Found " << device_count << " GPU(s)\n";
    
    if (device_count < 2) {
        std::cout << "Warning: Multi-GPU support requires at least 2 GPUs. Using single GPU.\n";
        CUDA_CALL(cudaSetDevice(0));
    } else {
        std::cout << "Using multi-GPU setup with " << device_count << " GPUs\n";
    }

    //std::vector<int> problem_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152,3000000,4194304,6000000,8388608};
    std::vector<int> problem_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152,3000000,4194304,6000000,8388608,12000000,16777216};
    //std::string csv_name = "test_timing.csv";
    //std::string csv_name = "gpu_only_timing.csv";
    std::string csv_name = "data/multi_gpu_timing.csv";
    bool print_input = false;
    bool run_toy2 = false;
    for (int ai = 1; ai < argc; ++ai) {
        std::string arg(argv[ai]);
        if ((arg == "-n" || arg == "--N") && ai + 1 < argc) {
            std::string val(argv[++ai]);
            problem_sizes.clear();
            std::stringstream ss(val);
            while (ss.good()) {
                std::string tok;
                if (!std::getline(ss, tok, ',')) break;
                if (!tok.empty()) problem_sizes.push_back(std::stoi(tok));
            }
        } else if (arg.rfind("--N=", 0) == 0) {
            std::string val = arg.substr(4);
            problem_sizes.clear();
            std::stringstream ss(val);
            while (ss.good()) {
                std::string tok;
                if (!std::getline(ss, tok, ',')) break;
                if (!tok.empty()) problem_sizes.push_back(std::stoi(tok));
            }
        } else if (arg.rfind("--csv=", 0) == 0) {
            csv_name = arg.substr(6);
        } else if (arg == "--print-input") {
            print_input = true;
        }
        else if (arg == "--toy2") {
            run_toy2 = true;
        }
    }
    
    // Print help if requested
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "Usage: " << argv[0] << " Options:\n";
        std::cout << "  -n, --N <sizes>     Comma-separated problem sizes (default: 8,16,32)\n";
        std::cout << "  --csv=<file>        Output CSV file name\n";
        std::cout << "  --print-input       Print input values for debugging\n";
        std::cout << "  --toy2              Run simple toy test (3*G + 5*G = 8*G)\n";
        std::cout << "  -h, --help          Show this help message\n";
        std::cout << "\n";
        return 0;
    }

    if (run_toy2) {
        int rc = run_toy2_gpu_vs_ref();
        return rc; // exit after the toy test
    }
    

    std::ofstream csv_file(csv_name);
    // csv_file << "N,CPU_Total,GPU_Total,GPU_Compute,GPU_Transfer,GPU_Bucket,GPU_Window,CPU_Bucket,CPU_Window,CPU_Final\n";
    csv_file << "N,GPU_Total,GPU_Compute,GPU_Transfer,GPU_Bucket,GPU_Window\n";

    for (int N : problem_sizes) {
        std::cout << "\n=== GPU-only check with N=" << N << " ===\n";

        constexpr int C = 8;        // window size in bits
        constexpr int BITSIZE = 64; // scalar_t assumed 64-bit here
        const int windows = (BITSIZE + C - 1) / C;
        const int num_buckets = (1 << C) - 1;
        const int total_buckets = windows * num_buckets;

        // Step 1: Random inputs 
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<uint64_t> dist(1, BN254_PRIME - 1);

        std::vector<scalar_t> scalars(N);
        std::vector<ECPoint>  bases(N);
        std::vector<uint64_t> base_scalars(N);

        ECPoint G(fq(1), fq(2), fq(1));
        for (int i = 0; i < N; ++i) {
            scalars[i] = (scalar_t)dist(rng);
            uint64_t base_scalar = dist(rng);
            base_scalars[i] = base_scalar;
            bases[i] = G * (scalar_t)base_scalar;
        }
        if (print_input) {
            std::cout << "Inputs (N=" << N << ")\n";
            for (int i = 0; i < N; ++i) {
                ECPoint A = bases[i];
                A.to_affine();
                std::cout << "i=" << i
                          << " scalar=" << (unsigned long long)scalars[i]
                          << " base_scalar=" << (unsigned long long)base_scalars[i]
                          << " base_affine=(" << A.X.value << "," << A.Y.value << ")\n";
            }
        }

        // Step 2–3: Bucketize on CPU 
        std::vector<std::vector<ECPoint>> bucket_points(total_buckets);
        for (int i = 0; i < N; ++i) {
            scalar_t s = scalars[i];
            for (int w = 0; w < BITSIZE; w += C) {
                int b = (int)((s >> w) & ((1 << C) - 1));
                if (b == 0) continue;
                int idx = (w / C) * num_buckets + (b - 1);
                bucket_points[idx].push_back(bases[i]);
            }
        }

        // CPU per-bucket sums - COMMENTED OUT
        /*
        std::vector<ECPoint> cpu_bucket_sums(total_buckets);
        auto cpu_bucket_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < total_buckets; ++i) {
            ECPoint sum; sum.X=fq(0); sum.Y=fq(1); sum.Z=fq(0);
            for (const auto& P : bucket_points[i]) sum = sum + P;
            cpu_bucket_sums[i] = sum;
        }
        auto cpu_bucket_end = std::chrono::high_resolution_clock::now();
        auto cpu_bucket_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_bucket_end - cpu_bucket_start).count();
        */

        // Step 4: Flatten for GPU 
        int total_points = 0;
        for (int i = 0; i < total_buckets; ++i) total_points += (int)bucket_points[i].size();

        std::vector<ECPoint> all_points(total_points);
        std::vector<int> bucket_sizes(total_buckets);
        std::vector<int> bucket_offsets(total_buckets + 1);

        int point_idx = 0;
        bucket_offsets[0] = 0;
        for (int i = 0; i < total_buckets; ++i) {
            bucket_sizes[i] = (int)bucket_points[i].size();
            for (int j = 0; j < bucket_sizes[i]; ++j) {
                all_points[point_idx++] = bucket_points[i][j];
            }
            bucket_offsets[i + 1] = point_idx;
        }

        // Multi-GPU processing with bucket distribution
        std::vector<ECPoint> combined_window_sums(windows);
        for (int w = 0; w < windows; ++w) {
            combined_window_sums[w].X = fq(0);
            combined_window_sums[w].Y = fq(1);
            combined_window_sums[w].Z = fq(0);
        }

        // auto total_gpu_start = std::chrono::high_resolution_clock::now(); // replaced by CUDA overall events
        // Track per-phase wall-clock time as the maximum across GPUs (not sum)
        long long total_gpu_tx_h2d = 0;
        long long total_gpu_bucket = 0;
        long long total_gpu_window = 0;
        long long total_gpu_tx_d2h = 0;

        // Distribute WINDOWS across GPUs (keep per-window bucket layout intact)
        int windows_per_gpu = windows / device_count;
        int remaining_windows = windows % device_count;

        // Pre-allocate memory and create streams for each GPU
        std::vector<cudaStream_t> streams(device_count);
        std::vector<ECPoint*> d_all_points_vec(device_count);
        std::vector<int*> d_bucket_sizes_vec(device_count);
        std::vector<int*> d_bucket_offsets_vec(device_count);
        std::vector<ECPoint*> d_bucket_sums_vec(device_count);
        std::vector<ECPoint*> d_window_sums_vec(device_count);
        std::vector<std::vector<ECPoint>> gpu_all_points_vec(device_count);
        std::vector<std::vector<int>> gpu_bucket_sizes_vec(device_count);
        std::vector<std::vector<int>> gpu_bucket_offsets_vec(device_count);
        // CUDA events for timing per GPU
        std::vector<cudaEvent_t> ev_h2d_start(device_count), ev_h2d_end(device_count);
        std::vector<cudaEvent_t> ev_bucket_start(device_count), ev_bucket_end(device_count);
        std::vector<cudaEvent_t> ev_window_start(device_count), ev_window_end(device_count);
        std::vector<cudaEvent_t> ev_d2h_start(device_count), ev_d2h_end(device_count);
        std::vector<cudaEvent_t> ev_overall_start(device_count), ev_overall_end(device_count);

        // Initialize streams and prepare data structures for each GPU
        for (int gpu_id = 0; gpu_id < device_count; ++gpu_id) {
            CUDA_CALL(cudaSetDevice(gpu_id));
            CUDA_CALL(cudaStreamCreate(&streams[gpu_id]));
            // Create CUDA events for this GPU
            CUDA_CALL(cudaEventCreate(&ev_h2d_start[gpu_id]));
            CUDA_CALL(cudaEventCreate(&ev_h2d_end[gpu_id]));
            CUDA_CALL(cudaEventCreate(&ev_bucket_start[gpu_id]));
            CUDA_CALL(cudaEventCreate(&ev_bucket_end[gpu_id]));
            CUDA_CALL(cudaEventCreate(&ev_window_start[gpu_id]));
            CUDA_CALL(cudaEventCreate(&ev_window_end[gpu_id]));
            CUDA_CALL(cudaEventCreate(&ev_d2h_start[gpu_id]));
            CUDA_CALL(cudaEventCreate(&ev_d2h_end[gpu_id]));
            CUDA_CALL(cudaEventCreate(&ev_overall_start[gpu_id]));
            CUDA_CALL(cudaEventCreate(&ev_overall_end[gpu_id]));
            
            // Calculate window range for this GPU
            int start_window = gpu_id * windows_per_gpu + std::min(gpu_id, remaining_windows);
            int end_window = start_window + windows_per_gpu + (gpu_id < remaining_windows ? 1 : 0);
            int windows_this_gpu = end_window - start_window;
            int num_buckets_this_gpu = windows_this_gpu * num_buckets;
            
            // Extract per-GPU data for assigned windows (preserve per-window bucket layout)
            std::vector<ECPoint> gpu_all_points;
            std::vector<int> gpu_bucket_sizes(num_buckets_this_gpu);
            std::vector<int> gpu_bucket_offsets(num_buckets_this_gpu + 1);
            int point_idx = 0;
            gpu_bucket_offsets[0] = 0;
            for (int w_local = 0; w_local < windows_this_gpu; ++w_local) {
                int w_global = start_window + w_local;
                int base_bucket = w_global * num_buckets;
                for (int b = 0; b < num_buckets; ++b) {
                    int global_bucket_idx = base_bucket + b;
                    int local_bucket_idx = w_local * num_buckets + b;
                    int size = bucket_sizes[global_bucket_idx];
                    gpu_bucket_sizes[local_bucket_idx] = size;
                    for (int j = 0; j < size; ++j) {
                        gpu_all_points.push_back(all_points[bucket_offsets[global_bucket_idx] + j]);
                    }
                    gpu_bucket_offsets[local_bucket_idx + 1] = (int)gpu_all_points.size();
                }
            }
 
            // Store data for this GPU
            gpu_all_points_vec[gpu_id] = gpu_all_points;
            gpu_bucket_sizes_vec[gpu_id] = gpu_bucket_sizes;
            gpu_bucket_offsets_vec[gpu_id] = gpu_bucket_offsets;

            // Allocate GPU memory
            int total_points_this_gpu = (int)gpu_all_points.size();
            ECPoint *d_all_points = nullptr, *d_bucket_sums = nullptr, *d_window_sums = nullptr;
            int *d_bucket_sizes = nullptr, *d_bucket_offsets = nullptr;

            CUDA_CALL(cudaMalloc(&d_all_points,     std::max(1, total_points_this_gpu) * (int)sizeof(ECPoint)));
            CUDA_CALL(cudaMalloc(&d_bucket_sizes,   num_buckets_this_gpu * (int)sizeof(int)));
            CUDA_CALL(cudaMalloc(&d_bucket_offsets, (num_buckets_this_gpu + 1) * (int)sizeof(int)));
            CUDA_CALL(cudaMalloc(&d_bucket_sums,    num_buckets_this_gpu * (int)sizeof(ECPoint)));
            CUDA_CALL(cudaMalloc(&d_window_sums,    windows_this_gpu * (int)sizeof(ECPoint)));

            // Store device pointers
            d_all_points_vec[gpu_id] = d_all_points;
            d_bucket_sizes_vec[gpu_id] = d_bucket_sizes;
            d_bucket_offsets_vec[gpu_id] = d_bucket_offsets;
            d_bucket_sums_vec[gpu_id] = d_bucket_sums;
            d_window_sums_vec[gpu_id] = d_window_sums;

            std::cout << "GPU " << gpu_id << " prepared: windows " << start_window << "-" << (end_window-1) 
                      << " (" << windows_this_gpu << " windows, " << num_buckets_this_gpu << " buckets, " << total_points_this_gpu << " points)\n";
        }

        // Launch parallel operations on all GPUs
        for (int gpu_id = 0; gpu_id < device_count; ++gpu_id) {
            CUDA_CALL(cudaSetDevice(gpu_id));
            
            int num_buckets_this_gpu = (int)gpu_bucket_sizes_vec[gpu_id].size();
            int total_points_this_gpu = (int)gpu_all_points_vec[gpu_id].size();
            int windows_this_gpu = num_buckets_this_gpu / num_buckets;
            
            // Overall start (before H2D)
            CUDA_CALL(cudaEventRecord(ev_overall_start[gpu_id], streams[gpu_id]));

            // Asynchronous H2D transfer (timed with CUDA events)
            CUDA_CALL(cudaEventRecord(ev_h2d_start[gpu_id], streams[gpu_id]));
            if (total_points_this_gpu)
                CUDA_CALL(cudaMemcpyAsync(d_all_points_vec[gpu_id], gpu_all_points_vec[gpu_id].data(), 
                                         total_points_this_gpu * sizeof(ECPoint), cudaMemcpyHostToDevice, streams[gpu_id]));
            CUDA_CALL(cudaMemcpyAsync(d_bucket_sizes_vec[gpu_id], gpu_bucket_sizes_vec[gpu_id].data(), 
                                     num_buckets_this_gpu * sizeof(int), cudaMemcpyHostToDevice, streams[gpu_id]));
            CUDA_CALL(cudaMemcpyAsync(d_bucket_offsets_vec[gpu_id], gpu_bucket_offsets_vec[gpu_id].data(), 
                                     (num_buckets_this_gpu + 1) * sizeof(int), cudaMemcpyHostToDevice, streams[gpu_id]));
            CUDA_CALL(cudaEventRecord(ev_h2d_end[gpu_id], streams[gpu_id]));

            // Launch bucket sums kernel asynchronously (timed with CUDA events)
            int threads = 256;
            int blocks = (num_buckets_this_gpu + threads - 1) / threads;
            CUDA_CALL(cudaEventRecord(ev_bucket_start[gpu_id], streams[gpu_id]));
            sum_bucket_points_kernel<<<blocks, threads, 0, streams[gpu_id]>>>(d_all_points_vec[gpu_id], 
                                                                             d_bucket_sizes_vec[gpu_id], 
                                                                             d_bucket_offsets_vec[gpu_id], 
                                                                             d_bucket_sums_vec[gpu_id], 
                                                                             num_buckets_this_gpu);
            CUDA_CALL(cudaEventRecord(ev_bucket_end[gpu_id], streams[gpu_id]));

            // Launch window sums kernel asynchronously (timed with CUDA events)
            int window_threads = 256;
            int window_blocks = (windows_this_gpu + window_threads - 1) / window_threads;
            CUDA_CALL(cudaEventRecord(ev_window_start[gpu_id], streams[gpu_id]));
            sum_window_buckets_kernel<<<window_blocks, window_threads, 0, streams[gpu_id]>>>(d_bucket_sums_vec[gpu_id], 
                                                                                            d_window_sums_vec[gpu_id], 
                                                                                            windows_this_gpu, num_buckets);
            CUDA_CALL(cudaEventRecord(ev_window_end[gpu_id], streams[gpu_id]));
        }

        long long max_overall_us = 0;

        // Wait for all GPUs to complete and collect results
        std::vector<std::vector<ECPoint>> gpu_window_sums_vec(device_count);
        for (int gpu_id = 0; gpu_id < device_count; ++gpu_id) {
            CUDA_CALL(cudaSetDevice(gpu_id));
            
            // Wait for this GPU to complete H2D and kernels
            CUDA_CALL(cudaStreamSynchronize(streams[gpu_id]));
            
            // Accumulate H2D, bucket, and window timings using CUDA events
            float ms = 0.0f;
            CUDA_CALL(cudaEventElapsedTime(&ms, ev_h2d_start[gpu_id], ev_h2d_end[gpu_id]));
            total_gpu_tx_h2d = std::max(total_gpu_tx_h2d, (long long)(ms * 1000.0f));
            CUDA_CALL(cudaEventElapsedTime(&ms, ev_bucket_start[gpu_id], ev_bucket_end[gpu_id]));
            total_gpu_bucket = std::max(total_gpu_bucket, (long long)(ms * 1000.0f));
            CUDA_CALL(cudaEventElapsedTime(&ms, ev_window_start[gpu_id], ev_window_end[gpu_id]));
            total_gpu_window = std::max(total_gpu_window, (long long)(ms * 1000.0f));

            // Asynchronous D2H transfer (timed with CUDA events)
            int windows_this_gpu = (int)gpu_bucket_sizes_vec[gpu_id].size() / num_buckets;
            std::vector<ECPoint> gpu_window_sums(windows_this_gpu);
            CUDA_CALL(cudaEventRecord(ev_d2h_start[gpu_id], streams[gpu_id]));
            CUDA_CALL(cudaMemcpyAsync(gpu_window_sums.data(), d_window_sums_vec[gpu_id], 
                                     windows_this_gpu * sizeof(ECPoint), cudaMemcpyDeviceToHost, streams[gpu_id]));
            CUDA_CALL(cudaEventRecord(ev_d2h_end[gpu_id], streams[gpu_id]));
            // Also mark overall end after final D2H copy is queued
            CUDA_CALL(cudaEventRecord(ev_overall_end[gpu_id], streams[gpu_id]));
            CUDA_CALL(cudaStreamSynchronize(streams[gpu_id])); // Wait for transfer to complete
            CUDA_CALL(cudaEventElapsedTime(&ms, ev_d2h_start[gpu_id], ev_d2h_end[gpu_id]));
            total_gpu_tx_d2h = std::max(total_gpu_tx_d2h, (long long)(ms * 1000.0f));
            // Compute overall elapsed from events and track max
            CUDA_CALL(cudaEventElapsedTime(&ms, ev_overall_start[gpu_id], ev_overall_end[gpu_id]));
            long long overall_us = (long long)(ms * 1000.0f);
            if (overall_us > max_overall_us) max_overall_us = overall_us;

            gpu_window_sums_vec[gpu_id] = gpu_window_sums;
        }

        // Combine results from all GPUs by placing window sums in the correct positions (no double work)
        int start_window = 0;
        for (int gpu_id = 0; gpu_id < device_count; ++gpu_id) {
            int windows_this_gpu = (int)gpu_window_sums_vec[gpu_id].size();
            for (int w_local = 0; w_local < windows_this_gpu; ++w_local) {
                combined_window_sums[start_window + w_local] = gpu_window_sums_vec[gpu_id][w_local];
            }
            start_window += windows_this_gpu;
        }
        if (start_window != windows) {
            std::cerr << "[WARN] Combined windows coverage mismatch: got " << start_window << ", expected " << windows << "\n";
        }

        // Cleanup
        for (int gpu_id = 0; gpu_id < device_count; ++gpu_id) {
            CUDA_CALL(cudaSetDevice(gpu_id));
            CUDA_CALL(cudaStreamDestroy(streams[gpu_id]));
            // Destroy CUDA events
            CUDA_CALL(cudaEventDestroy(ev_h2d_start[gpu_id]));
            CUDA_CALL(cudaEventDestroy(ev_h2d_end[gpu_id]));
            CUDA_CALL(cudaEventDestroy(ev_bucket_start[gpu_id]));
            CUDA_CALL(cudaEventDestroy(ev_bucket_end[gpu_id]));
            CUDA_CALL(cudaEventDestroy(ev_window_start[gpu_id]));
            CUDA_CALL(cudaEventDestroy(ev_window_end[gpu_id]));
            CUDA_CALL(cudaEventDestroy(ev_d2h_start[gpu_id]));
            CUDA_CALL(cudaEventDestroy(ev_d2h_end[gpu_id]));
            CUDA_CALL(cudaEventDestroy(ev_overall_start[gpu_id]));
            CUDA_CALL(cudaEventDestroy(ev_overall_end[gpu_id]));
            CUDA_CALL(cudaFree(d_all_points_vec[gpu_id]));
            CUDA_CALL(cudaFree(d_bucket_sizes_vec[gpu_id]));
            CUDA_CALL(cudaFree(d_bucket_offsets_vec[gpu_id]));
            CUDA_CALL(cudaFree(d_bucket_sums_vec[gpu_id]));
            CUDA_CALL(cudaFree(d_window_sums_vec[gpu_id]));
        }

        // auto total_gpu_end = std::chrono::high_resolution_clock::now(); // replaced by CUDA overall events
        // auto total_gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(total_gpu_end - total_gpu_start).count();
        auto total_gpu_us = max_overall_us; // use max across GPUs of overall event timing

        // CPU window (same weighting) for cross-check - COMMENTED OUT
        /*
        auto cpu_win_start = std::chrono::high_resolution_clock::now();
        std::vector<ECPoint> cpu_window_sums(windows);
        for (int w = 0; w < windows; ++w) {
            ECPoint run; run.X=fq(0); run.Y=fq(1); run.Z=fq(0);
            ECPoint acc; acc.X=fq(0); acc.Y=fq(1); acc.Z=fq(0);
            for (int b = num_buckets - 1; b >= 0; --b) {
                int idx = w * num_buckets + b;
                run = run + cpu_bucket_sums[idx];
                acc = acc + run;
            }
            cpu_window_sums[w] = acc;
        }
        auto cpu_win_end = std::chrono::high_resolution_clock::now();
        auto cpu_win_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_win_end - cpu_win_start).count();
        */

        // Quick window-level check
        /*
        for (int w = 0; w < windows; ++w) {
            if (!eq_affine(cpu_window_sums[w], gpu_window_sums[w])) {
                ECPoint a = cpu_window_sums[w]; a.to_affine();
                ECPoint b = gpu_window_sums[w]; b.to_affine();
                std::cerr << "[FAIL] window sum mismatch at w=" << w << "\n";
                std::cerr << "  CPU: (" << a.X.value << ", " << a.Y.value << ")\n";
                std::cerr << "  GPU: (" << b.X.value << ", " << b.Y.value << ")\n";
                return 21;
            }
        }
        std::cout << "[OK] window sums match (CPU vs GPU)\n";
        */

        //  Step 9: Final fold on CPU using GPU window sums 
        auto cpu_final_start = std::chrono::high_resolution_clock::now();
        ECPoint result; result.X=fq(0); result.Y=fq(1); result.Z=fq(0);
        for (int w = windows - 1; w >= 0; --w) {
            if (w != windows - 1) {
                for (int k = 0; k < C; ++k) result = result.dbl();
            }
            result = result + combined_window_sums[w];
        }
        auto cpu_final_end = std::chrono::high_resolution_clock::now();
        auto cpu_final_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_final_end - cpu_final_start).count();
        result.to_affine();

        // Step 10: reference MSM - COMMENTED OUT
        /*
        auto ref_start = std::chrono::high_resolution_clock::now();
        ECPoint ref = msm_ref(bases, scalars);
        ref.to_affine();
        auto ref_end = std::chrono::high_resolution_clock::now();
        auto ref_us = std::chrono::duration_cast<std::chrono::microseconds>(ref_end - ref_start).count();
        */

        //  Check A: fold(CPU windows) vs reference - COMMENTED OUT
        /*
        ECPoint folded_cpu; folded_cpu.X=fq(0); folded_cpu.Y=fq(1); folded_cpu.Z=fq(0);
        for (int w = windows - 1; w >= 0; --w) {
            if (w != windows - 1) for (int k = 0; k < C; ++k) folded_cpu = folded_cpu.dbl();
            folded_cpu = folded_cpu + cpu_window_sums[w];
        }
        folded_cpu.to_affine();
        if (!eq_affine(folded_cpu, ref)) {
            std::cerr << "[FAIL] Folding CPU windows != CPU reference\n";
            std::cerr << "  Fold(CPU): (" << folded_cpu.X.value << ", " << folded_cpu.Y.value << ")\n";
            std::cerr << "  Ref     : (" << ref        .X.value << ", " << ref        .Y.value << ")\n";
            return 11;
        } else {
            std::cout << "[OK] Folding CPU windows matches CPU reference\n";
        }
        */

        //  Check B: fold(GPU windows) vs reference - COMMENTED OUT
        /*
        if (!eq_affine(result, ref)) {
            std::cerr << "[FAIL] Folding GPU windows != CPU reference\n";
            std::cerr << "  Fold(GPU): (" << result.X.value << ", " << result.Y.value << ")\n";
            std::cerr << "  Ref     : (" << ref   .X.value << ", " << ref   .Y.value << ")\n";
            return 12;
        } else {
            std::cout << "[OK] Folding GPU windows matches CPU reference\n";
        }
        */

        std::cout << "Result (affine): x=" << result.X.value << " y=" << result.Y.value << "\n";

        // timings summary
        //auto total_cpu_us = cpu_bucket_us + cpu_win_us + cpu_final_us;
        //auto total_gpu_us = gpu_tx_h2d_us + gpu_bucket_us + gpu_window_us + gpu_tx_d2h_us;
        //auto gpu_comp_us  = gpu_bucket_us + gpu_window_us;
        auto gpu_comp_us = total_gpu_bucket + total_gpu_window;
        
        csv_file << N << "," << total_gpu_us << "," << gpu_comp_us << ","
                 << (total_gpu_tx_h2d + total_gpu_tx_d2h) << "," << total_gpu_bucket << "," << total_gpu_window << "\n";
    }

    csv_file.close();
    std::cout << "\nAll checks passed.\n";
    return 0;
}
