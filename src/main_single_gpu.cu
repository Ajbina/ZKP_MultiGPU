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
    std::vector<int> problem_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152,3000000,4194304,6000000,8388608,12000000,16777216};
    std::string csv_name = "data/single_gpu_timing.csv";
    bool print_input = false;
    bool run_toy2 = false;
    int batch_size = 0; // 0 means no batching
    
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
        } else if (arg.rfind("--batch-size=", 0) == 0) {
            batch_size = std::stoi(arg.substr(13));
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
        std::cout << "  --batch-size=<k>    Process N in batches of this size (single GPU)\n";
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
    csv_file << "N,GPU_Total,GPU_Compute,GPU_Transfer,GPU_Bucket,GPU_Window\n";

    for (int N : problem_sizes) {
        std::cout << "\n=== Single GPU check with N=" << N << " ===\n";

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
        // Determine batching
        int batches = (batch_size > 0) ? ( (N + batch_size - 1) / batch_size ) : 1;
        if (batches > 1) {
            std::cout << "Batching enabled: batch_size=" << batch_size << ", batches=" << batches << "\n";
        }

        // Accumulators for totals across batches
        long long total_gpu_tx_h2d_all = 0;
        long long total_gpu_bucket_all = 0;
        long long total_gpu_window_all = 0;
        long long total_gpu_tx_d2h_all = 0;
        long long total_gpu_all = 0;

        // Combined window sums across batches
        std::vector<ECPoint> combined_window_sums(windows);
        for (int w = 0; w < windows; ++w) {
            combined_window_sums[w].X = fq(0);
            combined_window_sums[w].Y = fq(1);
            combined_window_sums[w].Z = fq(0);
        }

        for (int batch_idx = 0; batch_idx < batches; ++batch_idx) {
            int start_i = (batch_size > 0) ? batch_idx * batch_size : 0;
            int end_i = (batch_size > 0) ? std::min(N, start_i + batch_size) : N;
            int Nb = end_i - start_i;
            if (Nb <= 0) continue;

            // Step 2–3: Bucketize on CPU for this batch
            std::vector<std::vector<ECPoint>> bucket_points(total_buckets);
            for (int i = start_i; i < end_i; ++i) {
                scalar_t s = scalars[i];
                for (int w = 0; w < BITSIZE; w += C) {
                    int b = (int)((s >> w) & ((1 << C) - 1));
                    if (b == 0) continue;
                    int idx = (w / C) * num_buckets + (b - 1);
                    bucket_points[idx].push_back(bases[i]);
                }
            }

            // Step 4: Flatten for GPU (batch)
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

            // Step 5: Copy to GPU 
        ECPoint *d_all_points = nullptr, *d_bucket_sums = nullptr, *d_window_sums = nullptr;
        int *d_bucket_sizes = nullptr, *d_bucket_offsets = nullptr;

        CUDA_CALL(cudaMalloc(&d_all_points,     std::max(1, total_points) * (int)sizeof(ECPoint)));
        CUDA_CALL(cudaMalloc(&d_bucket_sizes,   total_buckets * (int)sizeof(int)));
        CUDA_CALL(cudaMalloc(&d_bucket_offsets, (total_buckets + 1) * (int)sizeof(int)));
        CUDA_CALL(cudaMalloc(&d_bucket_sums,    total_buckets * (int)sizeof(ECPoint)));
        CUDA_CALL(cudaMalloc(&d_window_sums,    windows * (int)sizeof(ECPoint)));

            // CUDA events for GPU timings
        cudaEvent_t ev_h2d_start, ev_h2d_end;
        cudaEvent_t ev_bucket_start, ev_bucket_end;
        cudaEvent_t ev_window_start, ev_window_end;
        cudaEvent_t ev_d2h_start, ev_d2h_end;
        cudaEvent_t ev_overall_start, ev_overall_end;
        CUDA_CALL(cudaEventCreate(&ev_h2d_start));
        CUDA_CALL(cudaEventCreate(&ev_h2d_end));
        CUDA_CALL(cudaEventCreate(&ev_bucket_start));
        CUDA_CALL(cudaEventCreate(&ev_bucket_end));
        CUDA_CALL(cudaEventCreate(&ev_window_start));
        CUDA_CALL(cudaEventCreate(&ev_window_end));
        CUDA_CALL(cudaEventCreate(&ev_d2h_start));
        CUDA_CALL(cudaEventCreate(&ev_d2h_end));
        CUDA_CALL(cudaEventCreate(&ev_overall_start));
        CUDA_CALL(cudaEventCreate(&ev_overall_end));

            // H2D transfer timing using CUDA events
        CUDA_CALL(cudaEventRecord(ev_overall_start, 0));
        CUDA_CALL(cudaEventRecord(ev_h2d_start, 0));
        if (total_points)
            CUDA_CALL(cudaMemcpy(d_all_points, all_points.data(), total_points * sizeof(ECPoint), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_bucket_sizes,   bucket_sizes.data(),   total_buckets * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_bucket_offsets, bucket_offsets.data(), (total_buckets + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaEventRecord(ev_h2d_end, 0));
        CUDA_CALL(cudaEventSynchronize(ev_h2d_end));
        float ms_tmp = 0.0f;
        CUDA_CALL(cudaEventElapsedTime(&ms_tmp, ev_h2d_start, ev_h2d_end));
        long long gpu_tx_h2d_us = (long long)(ms_tmp * 1000.0f);

            //  Step 6: GPU bucket sums (timed with CUDA events)
        int threads = 256;
        int blocks  = (total_buckets + threads - 1) / threads;

        CUDA_CALL(cudaEventRecord(ev_bucket_start, 0));
        sum_bucket_points_kernel<<<blocks, threads>>>(d_all_points, d_bucket_sizes, d_bucket_offsets, d_bucket_sums, total_buckets);
        CUDA_CALL(cudaEventRecord(ev_bucket_end, 0));
        CUDA_CALL(cudaEventSynchronize(ev_bucket_end));
        CUDA_CALL(cudaEventElapsedTime(&ms_tmp, ev_bucket_start, ev_bucket_end));
        long long gpu_bucket_us = (long long)(ms_tmp * 1000.0f);

            //  Step 7: GPU window sums (timed with CUDA events)
        int window_threads = 256;
        int window_blocks  = (windows + window_threads - 1) / window_threads;

        CUDA_CALL(cudaEventRecord(ev_window_start, 0));
        sum_window_buckets_kernel<<<window_blocks, window_threads>>>(d_bucket_sums, d_window_sums, windows, num_buckets);
        CUDA_CALL(cudaEventRecord(ev_window_end, 0));
        CUDA_CALL(cudaEventSynchronize(ev_window_end));
        CUDA_CALL(cudaEventElapsedTime(&ms_tmp, ev_window_start, ev_window_end));
        long long gpu_window_us = (long long)(ms_tmp * 1000.0f);

            //  Step 8: Copy back window sums (timed with CUDA events)
            std::vector<ECPoint> gpu_window_sums(windows);
        CUDA_CALL(cudaEventRecord(ev_d2h_start, 0));
        CUDA_CALL(cudaMemcpy(gpu_window_sums.data(), d_window_sums, windows * sizeof(ECPoint), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaEventRecord(ev_d2h_end, 0));
        CUDA_CALL(cudaEventSynchronize(ev_d2h_end));
        CUDA_CALL(cudaEventRecord(ev_overall_end, 0));
        CUDA_CALL(cudaEventSynchronize(ev_overall_end));
        CUDA_CALL(cudaEventElapsedTime(&ms_tmp, ev_d2h_start, ev_d2h_end));
        long long gpu_tx_d2h_us = (long long)(ms_tmp * 1000.0f);

            // Compute overall elapsed using events
            CUDA_CALL(cudaEventElapsedTime(&ms_tmp, ev_overall_start, ev_overall_end));
            long long total_gpu_us = (long long)(ms_tmp * 1000.0f);

            // Destroy CUDA events
        CUDA_CALL(cudaEventDestroy(ev_h2d_start));
        CUDA_CALL(cudaEventDestroy(ev_h2d_end));
        CUDA_CALL(cudaEventDestroy(ev_bucket_start));
        CUDA_CALL(cudaEventDestroy(ev_bucket_end));
        CUDA_CALL(cudaEventDestroy(ev_window_start));
        CUDA_CALL(cudaEventDestroy(ev_window_end));
        CUDA_CALL(cudaEventDestroy(ev_d2h_start));
        CUDA_CALL(cudaEventDestroy(ev_d2h_end));
        CUDA_CALL(cudaEventDestroy(ev_overall_start));
        CUDA_CALL(cudaEventDestroy(ev_overall_end));

            // Accumulate totals across batches
            total_gpu_tx_h2d_all += gpu_tx_h2d_us;
            total_gpu_bucket_all += gpu_bucket_us;
            total_gpu_window_all += gpu_window_us;
            total_gpu_tx_d2h_all += gpu_tx_d2h_us;
            total_gpu_all        += total_gpu_us;

            // Accumulate window sums across batches
            for (int w = 0; w < windows; ++w) {
                combined_window_sums[w] = combined_window_sums[w] + gpu_window_sums[w];
            }

        CUDA_CALL(cudaFree(d_all_points));
        CUDA_CALL(cudaFree(d_bucket_sizes));
        CUDA_CALL(cudaFree(d_bucket_offsets));
        CUDA_CALL(cudaFree(d_bucket_sums));
        CUDA_CALL(cudaFree(d_window_sums));
        }

        //  Step 9: Final fold on CPU using aggregated window sums 
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

        std::cout << "Result (affine): x=" << result.X.value << " y=" << result.Y.value << "\n";

        // timings summary (across batches if any)
        auto total_gpu_us = total_gpu_all;
        auto gpu_comp_us = total_gpu_bucket_all + total_gpu_window_all;
        
        // Component-wise breakdown with percentages and bandwidth estimates
        double total_us_d = static_cast<double>(total_gpu_us);
        auto pct = [&](long long part_us) -> double {
            return total_us_d > 0.0 ? (static_cast<double>(part_us) / total_us_d) * 100.0 : 0.0;
        };

        // Estimate transferred bytes
        // Note: approximate per-batch H2D bytes times number of batches
        const size_t bytes_per_batch_h2d = (size_t)sizeof(ECPoint) * 0 /* unknown here */
                               + static_cast<size_t>(total_buckets) * sizeof(int)
                               + static_cast<size_t>(total_buckets + 1) * sizeof(int);
        const size_t bytes_h2d = bytes_per_batch_h2d * (size_t)batches; // rough estimate
        const size_t bytes_d2h = static_cast<size_t>(windows) * sizeof(ECPoint) * (size_t)batches;

        auto mbps = [](size_t bytes, long long us) -> double {
            if (us <= 0) return 0.0;
            // Using 1 MB = 1e6 bytes. MB/s = (bytes/us)
            return static_cast<double>(bytes) / static_cast<double>(us);
        };

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "GPU total (events): " << total_gpu_us << " us\n";
        std::cout << "  - H2D:    " << total_gpu_tx_h2d_all << " us (" << pct(total_gpu_tx_h2d_all) << "%)\n";
        std::cout << "  - Bucket: " << total_gpu_bucket_all << " us (" << pct(total_gpu_bucket_all) << "%)\n";
        std::cout << "  - Window: " << total_gpu_window_all << " us (" << pct(total_gpu_window_all) << "%)\n";
        std::cout << "  - D2H:    " << total_gpu_tx_d2h_all << " us (" << pct(total_gpu_tx_d2h_all) << "%)\n";
        std::cout << "GPU compute only: " << gpu_comp_us << " us ("
                  << pct(gpu_comp_us) << "%)\n";

        csv_file << N << "," << total_gpu_us << "," << gpu_comp_us << ","
                 << (total_gpu_tx_h2d_all + total_gpu_tx_d2h_all) << "," << total_gpu_bucket_all << "," << total_gpu_window_all << "\n";
    }

    csv_file.close();
    std::cout << "\nAll single GPU checks completed.\n";
    return 0;
} 