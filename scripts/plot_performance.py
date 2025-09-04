#!/usr/bin/env python3
"""
Performance plotting script for MSM CPU vs GPU comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogLocator

def plot_performance_comparison():
    """Plot CPU vs GPU performance comparison"""
    
    # Read the CSV data
    try:
        df = pd.read_csv('../data/test_timing.csv')
        print("Data loaded successfully:")
        print(df.head())
    except FileNotFoundError:
        print("Error: ../data/test_timing.csv not found. Please run the MSM program first.")
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Get the range of N values for x-axis limits
    n_min = df['N'].min()
    n_max = df['N'].max()
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MSM Performance: CPU vs GPU Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Time Comparison
    ax1.plot(df['N'], df['CPU_Total'], 'o-', label='CPU Total', linewidth=2, markersize=6)
    ax1.plot(df['N'], df['GPU_Total'], 's-', label='GPU Total (with transfers)', linewidth=2, markersize=6)
    ax1.plot(df['N'], df['GPU_Compute'], '^-', label='GPU Compute Only', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Points (N)')
    ax1.set_ylabel('Time (microseconds)')
    ax1.set_title('Total Execution Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.xaxis.set_major_locator(LogLocator(base=2))
    ax1.set_xlim(n_min * 0.8, n_max * 1.2)  # Ensure all points are visible
    
    # Plot 2: Speedup Analysis
    speedup_total = df['CPU_Total'] / df['GPU_Total']
    speedup_compute = df['CPU_Total'] / df['GPU_Compute']
    
    ax2.plot(df['N'], speedup_total, 'o-', label='Total Speedup (with transfers)', linewidth=2, markersize=6)
    ax2.plot(df['N'], speedup_compute, 's-', label='Compute Speedup (no transfers)', linewidth=2, markersize=6)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Break-even (1x)')
    ax2.set_xlabel('Number of Points (N)')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('GPU Speedup vs CPU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.xaxis.set_major_locator(LogLocator(base=2))
    ax2.set_xlim(n_min * 0.8, n_max * 1.2)  # Ensure all points are visible
    
    # Plot 3: Breakdown of GPU Operations
    ax3.plot(df['N'], df['GPU_Bucket'], 'o-', label='Bucket Summation', linewidth=2, markersize=6)
    ax3.plot(df['N'], df['GPU_Window'], 's-', label='Window Summation', linewidth=2, markersize=6)
    ax3.plot(df['N'], df['GPU_Transfer'], '^-', label='Memory Transfers', linewidth=2, markersize=6)
    ax3.set_xlabel('Number of Points (N)')
    ax3.set_ylabel('Time (microseconds)')
    ax3.set_title('GPU Operation Breakdown')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log')
    ax3.xaxis.set_major_locator(LogLocator(base=2))
    ax3.set_xlim(n_min * 0.8, n_max * 1.2)  # Ensure all points are visible
    
    # Plot 4: CPU vs GPU Component Comparison
    ax4.plot(df['N'], df['CPU_Bucket'], 'o-', label='CPU Bucket', linewidth=2, markersize=6)
    ax4.plot(df['N'], df['GPU_Bucket'], 's-', label='GPU Bucket', linewidth=2, markersize=6)
    ax4.plot(df['N'], df['CPU_Window'], '^-', label='CPU Window', linewidth=2, markersize=6)
    ax4.plot(df['N'], df['GPU_Window'], 'v-', label='GPU Window', linewidth=2, markersize=6)
    ax4.set_xlabel('Number of Points (N)')
    ax4.set_ylabel('Time (microseconds)')
    ax4.set_title('Component-wise CPU vs GPU')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log', base=2)
    ax4.set_yscale('log')
    ax4.xaxis.set_major_locator(LogLocator(base=2))
    ax4.set_xlim(n_min * 0.8, n_max * 1.2)  # Ensure all points are visible
    
    # Adjust layout and save
    plt.tight_layout()
            plt.savefig('../plots/msm_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('../plots/msm_performance_comparison.pdf', bbox_inches='tight')
    
    print("Graphs saved as:")
    print("- ../plots/msm_performance_comparison.png")
    print("- ../plots/msm_performance_comparison.pdf")
    
    # Print performance analysis
    print("\n=== PERFORMANCE ANALYSIS ===")
    
    # Find break-even point
    break_even_idx = np.where(speedup_total >= 1)[0]
    if len(break_even_idx) > 0:
        break_even_n = df.iloc[break_even_idx[0]]['N']
        print(f"GPU becomes faster than CPU at N = {break_even_n}")
    else:
        print("GPU does not achieve speedup over CPU in the tested range")
    
    # Best speedup
    max_speedup = speedup_total.max()
    max_speedup_n = df.loc[speedup_total.idxmax(), 'N']
    print(f"Best GPU speedup: {max_speedup:.2f}x at N = {max_speedup_n}")
    
    # Memory transfer overhead analysis
    transfer_overhead = (df['GPU_Transfer'] / df['GPU_Total'] * 100).mean()
    print(f"Average memory transfer overhead: {transfer_overhead:.1f}%")
    
    # Efficiency analysis
    print(f"\nEfficiency at different problem sizes:")
    for _, row in df.iterrows():
        n = row['N']
        cpu_time = row['CPU_Total']
        gpu_time = row['GPU_Total']
        speedup = cpu_time / gpu_time
        efficiency = "✓" if speedup > 1 else "✗"
        print(f"  N={n:5d}: CPU={cpu_time:6.0f}μs, GPU={gpu_time:6.0f}μs, Speedup={speedup:5.2f}x {efficiency}")

if __name__ == "__main__":
    plot_performance_comparison() 