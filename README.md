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
├── executables/           # Compiled binaries
│   ├── msm_multigpu       # Multi-GPU executable
│   ├── msm_single_gpu     # Single GPU executable
│   └── msm_smallcheck     # Legacy executable
├── scripts/               # Analysis and plotting scripts
│   ├── plot_comparison.py # Single vs Multi-GPU comparison
│   ├── plot_multi_gpu.py  # Multi-GPU performance analysis
│   ├── plot_performance.py # General performance plotting
│   └── run_performance_test.sh # Performance testing script
├── data/                  # Performance data files
│   ├── multi_gpu_timing.csv
│   ├── single_gpu_timing.csv
│   └── test_timing.csv
├── plots/                 # Generated plots and visualizations
│   ├── gpu_comparison.png
│   ├── multi_gpu_performance.png
│   └── msm_performance_comparison.png
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
nvcc -O2 main.cu kernel.cu common.cu -o ../executables/msm_multigpu
```

#### Single GPU Implementation
```bash
cd src
nvcc -O2 main_single_gpu.cu kernel.cu common.cu -o ../executables/msm_single_gpu
```

### Running Tests

#### Toy Test (Correctness Check)
```bash
# Multi-GPU toy test
./executables/msm_multigpu --toy2

# Single GPU toy test
./executables/msm_single_gpu --toy2
```

#### Performance Benchmarking
```bash
# Multi-GPU performance test
./executables/msm_multigpu

# Single GPU performance test
./executables/msm_single_gpu

# Custom problem sizes
./executables/msm_multigpu --N=64,128,256,512,1024
```

## 📊 Performance Analysis

### Generate Performance Plots
```bash
# Compare single vs multi-GPU performance
python3 scripts/plot_comparison.py

# Analyze multi-GPU performance only
python3 scripts/plot_multi_gpu.py

# General performance analysis
python3 scripts/plot_performance.py
```

### Automated Performance Testing
```bash
# Run comprehensive performance tests
bash scripts/run_performance_test.sh
```

## 🔧 Implementation Details

### Multi-GPU Distributed Approach
- **Bucket Distribution**: Evenly distributes buckets across available GPUs
- **Load Balancing**: Handles cases where total buckets don't divide evenly
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
- **gpu_comparison.png**: Single vs Multi-GPU performance comparison
- **multi_gpu_performance.png**: Detailed multi-GPU analysis
- **msm_performance_comparison.png**: General performance visualization

## 🔍 Analysis Scripts

### plot_comparison.py
- **Main Plot**: Total execution time comparison
- **Speedup Analysis**: Shows multi-GPU speedup over single GPU
- **Console Analysis**: Detailed performance statistics

### plot_multi_gpu.py
- **4-Panel Analysis**: Total time, operation breakdown, transfer overhead, throughput
- **Comprehensive Metrics**: Detailed multi-GPU performance analysis

### plot_performance.py
- **General Analysis**: Works with any timing CSV file
- **Base-2 Log Scale**: Optimized for power-of-2 problem sizes

## 🛠️ Development

### Adding New Features
1. Modify source files in `src/`
2. Update compilation commands
3. Test with toy examples first
4. Run performance benchmarks
5. Update documentation

### Debugging
- Use `--print-input` flag for detailed input inspection
- Run toy tests for correctness verification
- Check CSV output for timing breakdowns

## 📚 Technical Background

### MSM Algorithm
Multi-Scalar Multiplication computes: `∑(i=0 to N-1) scalar[i] * base[i]`

### Windowed Approach
- **Window Size**: 8 bits (configurable)
- **Buckets**: 255 buckets per window (1-255)
- **Windows**: 8 windows for 64-bit scalars

### GPU Optimization
- **Parallel Bucket Processing**: Each thread handles one bucket
- **Memory Coalescing**: Optimized memory access patterns
- **Kernel Fusion**: Combined bucket and window operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CUDA programming best practices
- Elliptic curve cryptography fundamentals
- Performance optimization techniques 