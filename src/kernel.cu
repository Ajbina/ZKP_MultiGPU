// kernel.cu
#include "common.cuh"
#include "kernel.cuh"

// Small helper macro for strong inlining on both host & device
#if defined(__CUDA_ARCH__)
#define HD_INLINE __host__ __device__ __forceinline__
#else
#define HD_INLINE __host__ __device__ inline
#endif

// ---------------------- Sum Points Within Each Bucket ----------------------
__global__ void sum_bucket_points_kernel(ECPoint* all_points, int* bucket_sizes, int* bucket_offsets, ECPoint* bucket_sums, int total_buckets) {
    int bucket_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_id >= total_buckets) return;

    // Silence unused-parameter warning for bucket_sizes (range is from offsets)
    (void)bucket_sizes;

    int start_idx = bucket_offsets[bucket_id];
    int end_idx   = bucket_offsets[bucket_id + 1];

    ECPoint acc; // infinity
    acc.X = fq(0); 
    acc.Y = fq(1); 
    acc.Z = fq(0);

    for (int i = start_idx; i < end_idx; ++i) {
        acc = acc + all_points[i];
    }
    bucket_sums[bucket_id] = acc;
}

// ---------------------- Sum Buckets Within Each Window ----------------------
__global__ void sum_window_buckets_kernel(ECPoint* bucket_sums, ECPoint* window_sums, int windows, int num_buckets) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= windows) return;

    ECPoint running; 
    running.X = fq(0); 
    running.Y = fq(1); 
    running.Z = fq(0);
    ECPoint acc; 
    acc.X = fq(0);
    acc.Y = fq(1); 
    acc.Z = fq(0);

    // Weighted sum trick: iterate buckets from high->low
    for (int b = num_buckets - 1; b >= 0; --b) {
        int idx = w * num_buckets + b;
        running = running + bucket_sums[idx]; // running += S[w,b]
        acc     = acc + running;              // acc     += running
    }
    window_sums[w] = acc;
}

// ---------------------- Test: Field Multiplication ----------------------
__global__ void test_fq_mul_kernel(const uint64_t* a, const uint64_t* b, uint64_t* c, int  n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fq fa(a[i]);
    fq fb(b[i]);
    fq fc = fa * fb;       // reduction is embedded in fq::operator*
    c[i] = fc.value;
}
