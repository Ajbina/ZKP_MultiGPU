#!/usr/bin/env python3
"""
Multi-GPU MSM Performance Analysis and Plotting
Plots performance data from multi_gpu_timing.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogLocator
import sys

def plot_multi_gpu_performance(csv_file='../data/multi_gpu_timing.csv'):
    """Plot multi-GPU MSM performance data"""
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        print(f"Loaded data from {csv_file}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-GPU MSM Performance Analysis', fontsize=16, fontweight='bold')
        
        # Get data ranges for x-axis limits
        n_min, n_max = df['N'].min(), df['N'].max()
        
        # Plot 1: Total GPU Time vs Problem Size
        ax1.loglog(df['N'], df['GPU_Total'], 'o-', linewidth=2, markersize=8, label='Total GPU Time')
        ax1.set_xlabel('Problem Size (N)', fontsize=12)
        ax1.set_ylabel('Time (μs)', fontsize=12)
        ax1.set_title('Total Multi-GPU Execution Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xscale('log', base=2)
        ax1.xaxis.set_major_locator(LogLocator(base=2))
        ax1.set_xlim(n_min * 0.8, n_max * 1.2)
        
        # Plot 2: GPU Operation Breakdown
        ax2.loglog(df['N'], df['GPU_Bucket'], 's-', linewidth=2, markersize=8, label='Bucket Sums')
        ax2.loglog(df['N'], df['GPU_Window'], '^-', linewidth=2, markersize=8, label='Window Sums')
        ax2.loglog(df['N'], df['GPU_Transfer'], 'd-', linewidth=2, markersize=8, label='Memory Transfer')
        ax2.set_xlabel('Problem Size (N)', fontsize=12)
        ax2.set_ylabel('Time (μs)', fontsize=12)
        ax2.set_title('GPU Operation Breakdown', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xscale('log', base=2)
        ax2.xaxis.set_major_locator(LogLocator(base=2))
        ax2.set_xlim(n_min * 0.8, n_max * 1.2)
        
        # Plot 3: Memory Transfer Overhead
        transfer_ratio = df['GPU_Transfer'] / df['GPU_Total'] * 100
        ax3.semilogx(df['N'], transfer_ratio, 'o-', linewidth=2, markersize=8, color='red')
        ax3.set_xlabel('Problem Size (N)', fontsize=12)
        ax3.set_ylabel('Transfer Overhead (%)', fontsize=12)
        ax3.set_title('Memory Transfer Overhead', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)
        ax3.xaxis.set_major_locator(LogLocator(base=2))
        ax3.set_xlim(n_min * 0.8, n_max * 1.2)
        
        # Plot 4: Throughput Analysis (MSM operations per second)
        throughput = df['N'] / (df['GPU_Total'] / 1e6)  # ops per second
        ax4.loglog(df['N'], throughput, 'o-', linewidth=2, markersize=8, color='green')
        ax4.set_xlabel('Problem Size (N)', fontsize=12)
        ax4.set_ylabel('Throughput (MSM ops/sec)', fontsize=12)
        ax4.set_title('Multi-GPU Throughput', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)
        ax4.xaxis.set_major_locator(LogLocator(base=2))
        ax4.set_xlim(n_min * 0.8, n_max * 1.2)
        
        plt.tight_layout()
        plt.savefig('../plots/multi_gpu_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Console analysis
        print("\n" + "="*60)
        print("MULTI-GPU PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Performance statistics
        print(f"\nPerformance Summary:")
        print(f"  Total problem sizes tested: {len(df)}")
        print(f"  Problem size range: {n_min:,} to {n_max:,}")
        
        # Timing analysis
        total_times = df['GPU_Total']
        bucket_times = df['GPU_Bucket']
        window_times = df['GPU_Window']
        transfer_times = df['GPU_Transfer']
        
        print(f"\nTiming Analysis (μs):")
        print(f"  Total GPU time - Min: {total_times.min():.0f}, Max: {total_times.max():.0f}, Avg: {total_times.mean():.0f}")
        print(f"  Bucket sums    - Min: {bucket_times.min():.0f}, Max: {bucket_times.max():.0f}, Avg: {bucket_times.mean():.0f}")
        print(f"  Window sums    - Min: {window_times.min():.0f}, Max: {window_times.max():.0f}, Avg: {window_times.mean():.0f}")
        print(f"  Memory transfer- Min: {transfer_times.min():.0f}, Max: {transfer_times.max():.0f}, Avg: {transfer_times.mean():.0f}")
        
        # Transfer overhead analysis
        avg_transfer_overhead = transfer_ratio.mean()
        print(f"\nTransfer Overhead Analysis:")
        print(f"  Average transfer overhead: {avg_transfer_overhead:.1f}%")
        print(f"  Min transfer overhead: {transfer_ratio.min():.1f}%")
        print(f"  Max transfer overhead: {transfer_ratio.max():.1f}%")
        
        # Throughput analysis
        print(f"\nThroughput Analysis:")
        print(f"  Peak throughput: {throughput.max():,.0f} MSM ops/sec")
        print(f"  Average throughput: {throughput.mean():,.0f} MSM ops/sec")
        
        # Scaling analysis
        if len(df) > 1:
            # Calculate scaling efficiency (how well it scales with problem size)
            scaling_factor = df['N'].iloc[-1] / df['N'].iloc[0]
            time_factor = df['GPU_Total'].iloc[-1] / df['GPU_Total'].iloc[0]
            ideal_scaling = scaling_factor / time_factor
            print(f"\nScaling Analysis:")
            print(f"  Problem size increase: {scaling_factor:.1f}x")
            print(f"  Time increase: {time_factor:.1f}x")
            print(f"  Scaling efficiency: {ideal_scaling:.2f}")
        
        print("\n" + "="*60)
        
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please run the multi-GPU MSM program first to generate the timing data.")
        return False
    except Exception as e:
        print(f"Error plotting data: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Allow command line argument for CSV file
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'multi_gpu_timing.csv'
    plot_multi_gpu_performance(csv_file) 