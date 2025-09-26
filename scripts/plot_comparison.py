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
import os

def get_plots_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def plot_performance_comparison(single_gpu_csv='../data/single_gpu_timing.csv', 
                               multi_gpu_csv='../data/multi_gpu_timing.csv'):
    """Compare single GPU vs multi-GPU MSM performance"""
    
    try:
        plots_dir = get_plots_dir()
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
        
        # Create multi-panel plots for each component
        metrics = [
            ('GPU_Total', 'Total Execution Time (μs)'),
            ('GPU_Compute', 'Compute Time (μs)'),
            ('GPU_Transfer', 'Transfer Time (μs) [H2D + D2H]'),
            ('GPU_Bucket', 'Bucket Kernel Time (μs)'),
            ('GPU_Window', 'Window Kernel Time (μs)')
        ]

        n_min, n_max = comparison_df['N'].min(), comparison_df['N'].max()

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        axes = axes.flatten()
        fig.suptitle('Single GPU vs Multi-GPU MSM: Component-wise Comparison', fontsize=18, fontweight='bold')

        for idx, (col, ylabel) in enumerate(metrics):
            ax = axes[idx]
            ax.loglog(comparison_df['N'], comparison_df[f'{col}_single'], 'o-',
                      linewidth=2.5, markersize=6, label='Single GPU', color='tab:blue')
            ax.loglog(comparison_df['N'], comparison_df[f'{col}_multi'], 's-',
                      linewidth=2.5, markersize=6, label='Multi-GPU', color='tab:red')
            ax.set_title(col.replace('_', ' '), fontsize=14, fontweight='bold')
            ax.set_xlabel('N (log2)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, which='both', alpha=0.3)
            ax.set_xscale('log', base=2)
            ax.xaxis.set_major_locator(LogLocator(base=2))
            ax.set_xlim(n_min * 0.8, n_max * 1.2)
            ax.legend(fontsize=10)

        # Speedup subplot
        speedup = comparison_df['GPU_Total_single'] / comparison_df['GPU_Total_multi']
        ax = axes[len(metrics)]
        ax.semilogx(comparison_df['N'], speedup, 'd-', linewidth=2.5, markersize=6, color='tab:green')
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        ax.set_title('Total Speedup (Single / Multi)', fontsize=14, fontweight='bold')
        ax.set_xlabel('N (log2)', fontsize=12)
        ax.set_ylabel('Speedup (×)', fontsize=12)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_locator(LogLocator(base=2))
        ax.set_xlim(n_min * 0.8, n_max * 1.2)

        # Hide any unused subplot
        if len(axes) > len(metrics) + 1:
            for j in range(len(metrics) + 1, len(axes)):
                fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        out_path = os.path.join(plots_dir, 'gpu_comparison_components(after batching single gpu).png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {out_path}")
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    default_single = os.path.join(project_root, 'data', 'single_gpu_timing.csv')
    default_multi = os.path.join(project_root, 'data', 'multi_gpu_timing.csv')
    single_csv = sys.argv[1] if len(sys.argv) > 1 else default_single
    multi_csv = sys.argv[2] if len(sys.argv) > 2 else default_multi
    plot_performance_comparison(single_csv, multi_csv) 