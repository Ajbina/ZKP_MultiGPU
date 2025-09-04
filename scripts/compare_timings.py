#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the timing data
single_gpu_df = pd.read_csv('data/single_gpu_timing.csv')
multi_gpu_df = pd.read_csv('multi_gpu_timing_comparison.csv')

# Merge the datasets on N
comparison_df = pd.merge(single_gpu_df, multi_gpu_df, on='N', suffixes=('_single', '_multi'))

# Calculate speedup metrics
comparison_df['Total_Speedup'] = comparison_df['GPU_Total_single'] / comparison_df['GPU_Total_multi']
comparison_df['Compute_Speedup'] = comparison_df['GPU_Compute_single'] / comparison_df['GPU_Compute_multi']
comparison_df['Transfer_Ratio'] = comparison_df['GPU_Transfer_multi'] / comparison_df['GPU_Transfer_single']

# Create comparison CSV
comparison_df.to_csv('timing_comparison.csv', index=False)

# Create plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Total GPU Time Comparison
ax1.loglog(comparison_df['N'], comparison_df['GPU_Total_single'], 'b-o', label='Single GPU', linewidth=2, markersize=6)
ax1.loglog(comparison_df['N'], comparison_df['GPU_Total_multi'], 'r-s', label='Multi GPU', linewidth=2, markersize=6)
ax1.set_xlabel('Problem Size (N)')
ax1.set_ylabel('Total GPU Time (μs)')
ax1.set_title('Total GPU Time: Single vs Multi-GPU')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: GPU Compute Time Comparison
ax2.loglog(comparison_df['N'], comparison_df['GPU_Compute_single'], 'b-o', label='Single GPU', linewidth=2, markersize=6)
ax2.loglog(comparison_df['N'], comparison_df['GPU_Compute_multi'], 'r-s', label='Multi GPU', linewidth=2, markersize=6)
ax2.set_xlabel('Problem Size (N)')
ax2.set_ylabel('GPU Compute Time (μs)')
ax2.set_title('GPU Compute Time: Single vs Multi-GPU')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Speedup Analysis
ax3.semilogx(comparison_df['N'], comparison_df['Total_Speedup'], 'g-o', label='Total Speedup', linewidth=2, markersize=6)
ax3.semilogx(comparison_df['N'], comparison_df['Compute_Speedup'], 'm-s', label='Compute Speedup', linewidth=2, markersize=6)
ax3.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No Speedup')
ax3.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='Ideal 2x Speedup')
ax3.set_xlabel('Problem Size (N)')
ax3.set_ylabel('Speedup (Single/Multi)')
ax3.set_title('Speedup Analysis: Multi-GPU vs Single-GPU')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.1, 10000)
ax3.set_yscale('log')

# Plot 4: Transfer Time Comparison
ax4.loglog(comparison_df['N'], comparison_df['GPU_Transfer_single'], 'b-o', label='Single GPU', linewidth=2, markersize=6)
ax4.loglog(comparison_df['N'], comparison_df['GPU_Transfer_multi'], 'r-s', label='Multi GPU', linewidth=2, markersize=6)
ax4.set_xlabel('Problem Size (N)')
ax4.set_ylabel('GPU Transfer Time (μs)')
ax4.set_title('GPU Transfer Time: Single vs Multi-GPU')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('timing_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("=== TIMING COMPARISON SUMMARY ===\n")

print("Single-GPU vs Multi-GPU Performance Analysis:")
print("-" * 50)

# Find crossover points
total_speedup_above_1 = comparison_df[comparison_df['Total_Speedup'] > 1]
total_speedup_below_1 = comparison_df[comparison_df['Total_Speedup'] < 1]

if len(total_speedup_above_1) > 0:
    print(f"Multi-GPU faster for N ≤ {total_speedup_above_1['N'].max():,}")
    print(f"  - Best total speedup: {total_speedup_above_1['Total_Speedup'].max():.2f}x at N={total_speedup_above_1.loc[total_speedup_above_1['Total_Speedup'].idxmax(), 'N']:,}")

if len(total_speedup_below_1) > 0:
    print(f"Single-GPU faster for N ≥ {total_speedup_below_1['N'].min():,}")
    print(f"  - Worst multi-GPU performance: {total_speedup_below_1['Total_Speedup'].min():.2f}x at N={total_speedup_below_1.loc[total_speedup_below_1['Total_Speedup'].idxmin(), 'N']:,}")

print("\nCompute Performance Analysis:")
print("-" * 30)
print(f"Average compute speedup: {comparison_df['Compute_Speedup'].mean():.1f}x")
print(f"Best compute speedup: {comparison_df['Compute_Speedup'].max():.1f}x at N={comparison_df.loc[comparison_df['Compute_Speedup'].idxmax(), 'N']:,}")
print(f"Worst compute speedup: {comparison_df['Compute_Speedup'].min():.1f}x at N={comparison_df.loc[comparison_df['Compute_Speedup'].idxmin(), 'N']:,}")

print("\nTransfer Overhead Analysis:")
print("-" * 25)
avg_transfer_ratio = comparison_df['Transfer_Ratio'].mean()
print(f"Average transfer overhead: {avg_transfer_ratio:.2f}x (multi-GPU vs single-GPU)")
max_transfer_ratio = comparison_df['Transfer_Ratio'].max()
max_transfer_n = comparison_df.loc[comparison_df['Transfer_Ratio'].idxmax(), 'N']
print(f"Worst transfer overhead: {max_transfer_ratio:.2f}x at N={max_transfer_n:,}")

# Create detailed comparison table for key problem sizes
key_sizes = [64, 128, 1024, 16384, 131072, 1048576, 8388608]
key_data = comparison_df[comparison_df['N'].isin(key_sizes)]

print(f"\nDetailed Comparison for Key Problem Sizes:")
print("=" * 80)
print(f"{'N':>10} | {'Single Total':>12} | {'Multi Total':>11} | {'Total Speedup':>13} | {'Compute Speedup':>15}")
print("-" * 80)

for _, row in key_data.iterrows():
    print(f"{row['N']:>10,} | {row['GPU_Total_single']:>10,.0f} μs | {row['GPU_Total_multi']:>9,.0f} μs | {row['Total_Speedup']:>11.2f}x | {row['Compute_Speedup']:>13.1f}x")

print("\n" + "=" * 80)
print("Key Insights:")
print("- Multi-GPU shows excellent compute speedup (100-8000x) due to work distribution")
print("- Total performance is limited by communication overhead for large problems")
print("- Optimal strategy: Multi-GPU for small-medium problems, single-GPU for large problems") 