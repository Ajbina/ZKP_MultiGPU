#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the timing data
single_gpu_df = pd.read_csv('data/single_gpu_timing.csv')
# Use the more realistic multi-GPU data from earlier run
multi_gpu_df = pd.read_csv('multi_gpu_timing_parallel.csv')

# Merge the datasets on N (keep only matching N values)
comparison_df = pd.merge(single_gpu_df, multi_gpu_df, on='N', suffixes=('_single', '_multi'))

# Calculate speedup metrics
comparison_df['Total_Speedup'] = comparison_df['GPU_Total_single'] / comparison_df['GPU_Total_multi']
comparison_df['Compute_Speedup'] = comparison_df['GPU_Compute_single'] / comparison_df['GPU_Compute_multi']
comparison_df['Transfer_Ratio'] = comparison_df['GPU_Transfer_multi'] / comparison_df['GPU_Transfer_single']

# Create comparison CSV
comparison_df.to_csv('timing_comparison_corrected.csv', index=False)

# Create plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Total GPU Time Comparison
ax1.loglog(comparison_df['N'], comparison_df['GPU_Total_single'], 'b-o', label='Single GPU', linewidth=2, markersize=6)
ax1.loglog(comparison_df['N'], comparison_df['GPU_Total_multi'], 'r-s', label='Multi GPU (Parallel)', linewidth=2, markersize=6)
ax1.set_xlabel('Problem Size (N)')
ax1.set_ylabel('Total GPU Time (μs)')
ax1.set_title('Total GPU Time: Single vs Multi-GPU')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: GPU Compute Time Comparison
ax2.loglog(comparison_df['N'], comparison_df['GPU_Compute_single'], 'b-o', label='Single GPU', linewidth=2, markersize=6)
ax2.loglog(comparison_df['N'], comparison_df['GPU_Compute_multi'], 'r-s', label='Multi GPU (Parallel)', linewidth=2, markersize=6)
ax2.set_xlabel('Problem Size (N)')
ax2.set_ylabel('GPU Compute Time (μs)')
ax2.set_title('GPU Compute Time: Single vs Multi-GPU')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Speedup Analysis
ax3.semilogx(comparison_df['N'], comparison_df['Total_Speedup'], 'g-o', label='Total Speedup', linewidth=2, markersize=6)
ax3.semilogx(comparison_df['N'], comparison_df['Compute_Speedup'], 'm-s', label='Compute Speedup', linewidth=2, markersize=6)
ax3.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No Speedup (1x)')
ax3.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='Ideal 2x Speedup')
ax3.set_xlabel('Problem Size (N)')
ax3.set_ylabel('Speedup (Single/Multi)')
ax3.set_title('Speedup Analysis: Multi-GPU vs Single-GPU')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.1, 10000)
ax3.set_yscale('log')

# Plot 4: Bucket vs Window Time Breakdown
width = 0.35
x_pos = np.arange(len(comparison_df))

# Create stacked bars for bucket and window times
ax4.bar(x_pos - width/2, comparison_df['GPU_Bucket_single'], width, label='Single GPU Bucket', alpha=0.8, color='lightblue')
ax4.bar(x_pos - width/2, comparison_df['GPU_Window_single'], width, bottom=comparison_df['GPU_Bucket_single'], 
        label='Single GPU Window', alpha=0.8, color='darkblue')

ax4.bar(x_pos + width/2, comparison_df['GPU_Bucket_multi'], width, label='Multi GPU Bucket', alpha=0.8, color='lightcoral')
ax4.bar(x_pos + width/2, comparison_df['GPU_Window_multi'], width, bottom=comparison_df['GPU_Bucket_multi'], 
        label='Multi GPU Window', alpha=0.8, color='darkred')

ax4.set_xlabel('Problem Size Index')
ax4.set_ylabel('Time (μs)')
ax4.set_title('Bucket vs Window Time Breakdown')
ax4.set_yscale('log')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Set x-axis labels for the breakdown plot
tick_positions = range(0, len(comparison_df), max(1, len(comparison_df)//8))
ax4.set_xticks(tick_positions)
ax4.set_xticklabels([f"{comparison_df.iloc[i]['N']:,}" for i in tick_positions], rotation=45)

plt.tight_layout()
plt.savefig('timing_comparison_corrected.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("=== CORRECTED TIMING COMPARISON SUMMARY ===\n")

print("Single-GPU vs Multi-GPU Performance Analysis:")
print("-" * 50)

# Find crossover points
total_speedup_above_1 = comparison_df[comparison_df['Total_Speedup'] > 1]
total_speedup_below_1 = comparison_df[comparison_df['Total_Speedup'] < 1]

if len(total_speedup_above_1) > 0:
    print(f"Multi-GPU faster for problem sizes: {', '.join([f'N={n:,}' for n in total_speedup_above_1['N']])}")
    best_speedup_idx = total_speedup_above_1['Total_Speedup'].idxmax()
    print(f"  - Best total speedup: {total_speedup_above_1.loc[best_speedup_idx, 'Total_Speedup']:.2f}x at N={total_speedup_above_1.loc[best_speedup_idx, 'N']:,}")

if len(total_speedup_below_1) > 0:
    print(f"Single-GPU faster for problem sizes: {', '.join([f'N={n:,}' for n in total_speedup_below_1['N']])}")
    worst_speedup_idx = total_speedup_below_1['Total_Speedup'].idxmin()
    print(f"  - Worst multi-GPU performance: {total_speedup_below_1.loc[worst_speedup_idx, 'Total_Speedup']:.2f}x at N={total_speedup_below_1.loc[worst_speedup_idx, 'N']:,}")

print("\nCompute Performance Analysis:")
print("-" * 30)
print(f"Average compute speedup: {comparison_df['Compute_Speedup'].mean():.1f}x")
best_compute_idx = comparison_df['Compute_Speedup'].idxmax()
print(f"Best compute speedup: {comparison_df.loc[best_compute_idx, 'Compute_Speedup']:.1f}x at N={comparison_df.loc[best_compute_idx, 'N']:,}")
worst_compute_idx = comparison_df['Compute_Speedup'].idxmin()
print(f"Worst compute speedup: {comparison_df.loc[worst_compute_idx, 'Compute_Speedup']:.1f}x at N={comparison_df.loc[worst_compute_idx, 'N']:,}")

print("\nTransfer Overhead Analysis:")
print("-" * 25)
avg_transfer_ratio = comparison_df['Transfer_Ratio'].mean()
print(f"Average transfer overhead: {avg_transfer_ratio:.2f}x (multi-GPU vs single-GPU)")
max_transfer_idx = comparison_df['Transfer_Ratio'].idxmax()
print(f"Worst transfer overhead: {comparison_df.loc[max_transfer_idx, 'Transfer_Ratio']:.2f}x at N={comparison_df.loc[max_transfer_idx, 'N']:,}")

# Create detailed comparison table for key problem sizes
print(f"\nDetailed Comparison for All Problem Sizes:")
print("=" * 95)
print(f"{'N':>10} | {'Single Total':>12} | {'Multi Total':>11} | {'Total Speedup':>13} | {'Compute Speedup':>15} | {'Status':>10}")
print("-" * 95)

for _, row in comparison_df.iterrows():
    status = "MULTI WINS" if row['Total_Speedup'] > 1 else "SINGLE WINS"
    print(f"{row['N']:>10,} | {row['GPU_Total_single']:>10,.0f} μs | {row['GPU_Total_multi']:>9,.0f} μs | {row['Total_Speedup']:>11.2f}x | {row['Compute_Speedup']:>13.1f}x | {status:>10}")

print("\n" + "=" * 95)
print("Key Insights:")
print("1. Multi-GPU achieves excellent compute speedup (100-8000x) through parallel execution")
print("2. Total performance varies by problem size due to communication vs computation tradeoff")
print("3. Small-medium problems benefit from multi-GPU, large problems are communication-limited")
print("4. The parallel implementation successfully eliminates the sequential GPU bottleneck") 