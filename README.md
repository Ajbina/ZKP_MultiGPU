# MSM (Multi-Scalar Multiplication) GPU Implementation

This repository contains optimized GPU implementations of Multi-Scalar Multiplication (MSM) for elliptic curve cryptography, featuring both single GPU and multi-GPU distributed approaches.

## 📁 Project Structure

```
├── src/                    # Source code files
│   ├── main.cu            # Multi-GPU distributed implementation
│   ├── main_single_gpu.cu # Single GPU implementation
│   ├── kernel.cu          # CUDA kernel implementations
│   ├── kernel.cuh         # CUDA kernel headers
│   ├── common.cu          # Common utility functions
│   └── common.cuh         # Common headers and definitions
├── bin/                   # Compiled binaries
│   ├── msm_main           # Multi-GPU executable
│   ├── msm_single         # Single GPU executable
│   └── msm_smallcheck     # Legacy executable
├── scripts/               # Analysis and plotting scripts
│   ├── plot_comparison.py # Single vs Multi-GPU comparison
│   ├── plot_multi_gpu.py  # Multi-GPU performance analysis
│   ├── plot_performance.py # General performance plotting
│   └── run_performance_test.sh # Performance testing script
├── data/                  # Performance data files
│   ├── multi_gpu_timing.csv
│   ├── single_gpu_timing.csv
│   └── 
├── plots/                 # Generated plots and visualizations
│   ├── gpu_comparison_components.png #single and multi gpu comparison
│   ├── 
│   └── 
├── build/                 # Build artifacts
├── CMakeLists.txt         # CMake configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU(s) with CUDA support
- CUDA Toolkit (version 11.0 or higher)
- Python 3.7+ with required packages (see requirements.txt)

### Compilation

#### Multi-GPU Implementation
```bash
cd src
nvcc -O2 main.cu kernel.cu common.cu -o ../bin/msm_main
```

#### Single GPU Implementation
```bash
cd src
nvcc -O2 main_single_gpu.cu kernel.cu common.cu -o ../bin/msm_single_gpu
```

### Running Tests

#### Toy Test (Correctness Check)
```bash
# Multi-GPU toy test
./bin/msm_main --toy2

# Single GPU toy test
./bin/msm_single --toy2
```

#### Performance Benchmarking
```bash
# Multi-GPU performance test
./bin/msm_main

# Single GPU performance test
./bin/msm_single

# Custom problem sizes
./bin/msm_main --N=64,128,256,512,1024
```

## 📊 Performance Analysis

### Generate Performance Plots
```bash
# Compare single vs multi-GPU performance
python3 scripts/compare_timings.py

# Analyze multi-GPU performance only
python3 scripts/plot_multi_gpu.py

# General performance analysis
python3 scripts/plot_performance.py
```

## 🔧 Implementation Details

### Multi-GPU Distributed Approach
- **Window Distribution**: Evenly distributes windows across available GPUs
- **Result Combination**: Combines partial results from each GPU on CPU
- **Memory Efficiency**: Each GPU processes only its assigned subset

### Single GPU Approach
- **Simplified Architecture**: All buckets processed on one GPU
- **Streamlined Memory Management**: Single device allocation
- **Direct Timing Measurements**: No aggregation needed

### Key Features
- **Elliptic Curve Arithmetic**: Point addition, doubling, scalar multiplication
- **Windowed MSM**: 8-bit window size for optimal performance
- **Bucket-based Algorithm**: Efficient grouping of similar operations
- **CUDA Optimization**: Parallel processing with custom kernels

## 📈 Performance Metrics

The implementations measure and report:
- **Total GPU Time**: Complete execution time
- **GPU Compute Time**: Pure computation (bucket + window sums)
- **Memory Transfer Time**: Host-to-device and device-to-host transfers
- **Bucket Sum Time**: Time for bucket-level point additions
- **Window Sum Time**: Time for window-level weighted sums

## 🎯 Command Line Options

```
Usage: msm_multigpu [Options]
  -n, --N <sizes>     Comma-separated problem sizes (default: 64,128,256,...)
  --csv=<file>        Output CSV file name
  --print-input       Print input values for debugging
  --toy2              Run simple toy test (3*G + 5*G = 8*G)
  -h, --help          Show this help message
```

## 📋 Output Files

### CSV Data Format
```
N,GPU_Total,GPU_Compute,GPU_Transfer,GPU_Bucket,GPU_Window
64,1234,1000,234,800,200
128,2345,2000,345,1600,400
...
```

### Generated Plots
- **gpu_comparison_components.png**: Single vs Multi-GPU performance comparison components wise

## 🔍 Analysis Scripts

### plot_comparison.py
-


## 🛠️ Development



### Debugging


## 📚 Technical Background

### MSM Algorithm
Multi-Scalar Multiplication computes: `∑(i=0 to N-1) scalar[i] * base[i]`

### Windowed Approach
- **Window Size**: 8 bits (configurable)
- **Buckets**: 255 buckets per window (1-255)
- **Windows**: 8 windows for 64-bit scalars

### GPU Optimization
- **Parallel Bucket Processing**: Each thread handles one bucket


