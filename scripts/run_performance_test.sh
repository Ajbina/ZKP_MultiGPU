#!/bin/bash

echo "=== MSM Performance Testing and Plotting ==="
echo

# Check if we're in the build directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please run this script from the project root directory (not from build/)"
    exit 1
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

echo "1. Building MSM program..."
cmake .. && make msm

if [ $? -ne 0 ]; then
    echo "Error: Build failed!"
    exit 1
fi

echo "2. Running performance tests..."
./msm

if [ $? -ne 0 ]; then
    echo "Error: MSM program failed!"
    exit 1
fi

echo "3. Checking if timing data was generated..."
if [ ! -f "timing_data.csv" ]; then
    echo "Error: timing_data.csv not found!"
    exit 1
fi

echo "4. Installing Python dependencies..."
cd ..
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Warning: Could not install Python dependencies. You may need to install them manually:"
    echo "  pip3 install pandas matplotlib numpy seaborn"
fi

echo "5. Generating performance graphs..."
python3 plot_performance.py

if [ $? -ne 0 ]; then
    echo "Error: Plotting failed!"
    exit 1
fi

echo
echo "=== Performance testing completed successfully! ==="
echo "Results:"
echo "- timing_data.csv: Raw timing data"
echo "- msm_performance_comparison.png: Performance graph (PNG)"
echo "- msm_performance_comparison.pdf: Performance graph (PDF)"
echo
echo "You can now analyze the performance characteristics!" 