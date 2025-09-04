#pragma once
#include "common.cuh"

// Kernel for GPU bucket point summation
__global__ void sum_bucket_points_kernel(ECPoint* all_points, int* bucket_sizes, int* bucket_offsets, ECPoint* bucket_sums, int total_buckets);

// Kernel for GPU window-wise bucket summation
__global__ void sum_window_buckets_kernel(ECPoint* bucket_sums, ECPoint* window_sums, int windows, int num_buckets);

// Test kernel: field multiplication of arrays
__global__ void test_fq_mul_kernel(const uint64_t* a, const uint64_t* b, uint64_t* c, int n);
