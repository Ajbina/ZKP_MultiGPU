#!/usr/bin/env python3
"""
Single GPU vs Multi-GPU MSM Performance Comparison
Compares performance data from single_gpu_timing.csv and multi_gpu_timing.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogLocator
import sys

def plot_performance_comparison(single_gpu_csv='../data/single_gpu_timing.csv', 
                               multi_gpu_csv='../data/multi_gpu_timing.csv'):
    """Compare single GPU vs multi-GPU MSM performance"""
    
    try:
        # Read both CSV files
        print("Loading performance data...")
        single_df = pd.read_csv(single_gpu_csv)
        multi_df = pd.read_csv(multi_gpu_csv)
        
        print(f"Single GPU data: {single_df.shape[0]} samples")
        print(f"Multi-GPU data: {multi_df.shape[0]} samples")
        
        # Merge dataframes on N for comparison
        comparison_df = pd.merge(single_df, multi_df, on='N', suffixes=('_single', '_multi'))
        print(f"Common problem sizes: {len(comparison_df)}")
        
        if len(comparison_df) == 0:
            print("Error: No common problem sizes found between the two datasets")
            return False
        
        print("\nFirst few rows of comparison data:")
        print(comparison_df.head())
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create single main plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('Single GPU vs Multi-GPU MSM Performance Comparison', fontsize=16, fontweight='bold')
        
        # Get data ranges for x-axis limits
        n_min, n_max = comparison_df['N'].min(), comparison_df['N'].max()
        
        # Main plot: Total Execution Time Comparison with Speedup
        ax.loglog(comparison_df['N'], comparison_df['GPU_Total_single'], 'o-', 
                 linewidth=3, markersize=10, label='Single GPU', color='blue')
        ax.loglog(comparison_df['N'], comparison_df['GPU_Total_multi'], 's-', 
                 linewidth=3, markersize=10, label='Multi-GPU', color='red')
        
        # Add speedup annotations
        speedup = comparison_df['GPU_Total_single'] / comparison_df['GPU_Total_multi']
        for i, (n, sp) in enumerate(zip(comparison_df['N'], speedup)):
            if sp > 1.5:  # Only annotate significant speedups
                ax.annotate(f'{sp:.1f}x', 
                           xy=(n, comparison_df['GPU_Total_multi'].iloc[i]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Problem Size (N)', fontsize=14)
        ax.set_ylabel('Total Execution Time (μs)', fontsize=14)
        ax.set_title('Performance Comparison', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_locator(LogLocator(base=2))
        ax.set_xlim(n_min * 0.8, n_max * 1.2)
        
        plt.tight_layout()
        plt.savefig('../plots/gpu_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Console analysis
        print("\n" + "="*60)
        print("SINGLE GPU vs MULTI-GPU PERFORMANCE COMPARISON")
        print("="*60)
        
        # Performance statistics
        print(f"\nPerformance Summary:")
        print(f"  Problem sizes compared: {len(comparison_df)}")
        print(f"  Problem size range: {n_min:,} to {n_max:,}")
        
        # Timing analysis
        single_total = comparison_df['GPU_Total_single']
        multi_total = comparison_df['GPU_Total_multi']
        
        print(f"\nTiming Analysis (μs):")
        print(f"  Single GPU Total - Min: {single_total.min():.0f}, Max: {single_total.max():.0f}, Avg: {single_total.mean():.0f}")
        print(f"  Multi-GPU Total  - Min: {multi_total.min():.0f}, Max: {multi_total.max():.0f}, Avg: {multi_total.mean():.0f}")
        
        # Speedup analysis
        avg_speedup = speedup.mean()
        max_speedup = speedup.max()
        min_speedup = speedup.min()
        
        print(f"\nSpeedup Analysis:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Maximum speedup: {max_speedup:.2f}x")
        print(f"  Minimum speedup: {min_speedup:.2f}x")
        
        # Efficiency analysis
        efficiency = speedup / 2  # Assuming 2 GPUs
        avg_efficiency = efficiency.mean()
        print(f"  Average efficiency: {avg_efficiency:.1%}")
        
        # Problem size specific analysis
        print(f"\nProblem Size Specific Analysis:")
        for idx, row in comparison_df.iterrows():
            n = row['N']
            single_time = row['GPU_Total_single']
            multi_time = row['GPU_Total_multi']
            speedup_val = row['GPU_Total_single'] / row['GPU_Total_multi']
            print(f"  N={n:,}: Single={single_time:.0f}μs, Multi={multi_time:.0f}μs, Speedup={speedup_val:.2f}x")
        
        print("\n" + "="*60)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the CSV files: {e}")
        print("Please run both single GPU and multi-GPU programs first to generate the timing data.")
        return False
    except Exception as e:
        print(f"Error plotting comparison data: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Allow command line arguments for CSV files
    single_csv = sys.argv[1] if len(sys.argv) > 1 else 'single_gpu_timing.csv'
    multi_csv = sys.argv[2] if len(sys.argv) > 2 else 'multi_gpu_timing.csv'
    plot_performance_comparison(single_csv, multi_csv) 