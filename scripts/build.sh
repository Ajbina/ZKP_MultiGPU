#!/bin/bash

# MSM GPU Implementation Build Script
# Compiles both single GPU and multi-GPU versions

echo "Building MSM GPU implementations..."

# Create executables directory if it doesn't exist
mkdir -p ../executables

# Build Multi-GPU implementation
echo "Building multi-GPU implementation..."
cd ../src
nvcc -O2 main.cu kernel.cu common.cu -o ../executables/msm_multigpu
if [ $? -eq 0 ]; then
    echo "✅ Multi-GPU build successful: ../executables/msm_multigpu"
else
    echo "❌ Multi-GPU build failed"
    exit 1
fi

# Build Single GPU implementation
echo "Building single GPU implementation..."
nvcc -O2 main_single_gpu.cu kernel.cu common.cu -o ../executables/msm_single_gpu
if [ $? -eq 0 ]; then
    echo "✅ Single GPU build successful: ../executables/msm_single_gpu"
else
    echo "❌ Single GPU build failed"
    exit 1
fi

echo ""
echo "🎉 All builds completed successfully!"
echo ""
echo "Available executables:"
echo "  ../executables/msm_multigpu     - Multi-GPU distributed implementation"
echo "  ../executables/msm_single_gpu   - Single GPU implementation"
echo ""
echo "Usage examples:"
echo "  ../executables/msm_multigpu --toy2                    # Run toy test"
echo "  ../executables/msm_single_gpu --toy2                  # Run toy test"
echo "  ../executables/msm_multigpu                           # Run performance test"
echo "  ../executables/msm_single_gpu                         # Run performance test" 