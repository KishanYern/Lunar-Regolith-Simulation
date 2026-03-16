#!/bin/bash
# Unified compilation and execution script for AMD GPU OpenMP offloading
# Targets gfx1032 (RX 6600/XT, RDNA2) directly.
#
# Usage:
#   ./run_simulation.sh [N]
#
# Examples:
#   ./run_simulation.sh           # default 5 000 particles
#   ./run_simulation.sh 1000000   # 1 million particles

ROCM_PATH=/opt/rocm-6.3.1
CLANGXX=$ROCM_PATH/lib/llvm/bin/clang++
ARCH="gfx1030"  # gfx1032 uses the same ISA as gfx1030; runtime override handles the rest

N=${1:-5000}

echo "Attempting to compile with ROCm clang++ for AMD GPU..."
echo "Targeting architecture: $ARCH (runtime override will map this to your gfx1032)"

$CLANGXX -O3 -std=c++17 lunar_sim.cpp -o simulator_gpu \
    -fopenmp \
    --offload-arch=$ARCH

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Successfully compiled! Starting GPU-Accelerated Simulation..."
    echo "Particle count: $N"
    echo "========================================"
    echo "Running: HSA_OVERRIDE_GFX_VERSION=10.3.0 ./simulator_gpu $N"
    echo ""

    HSA_OVERRIDE_GFX_VERSION=10.3.0 ./simulator_gpu "$N"

    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "Simulation completed successfully!"
        echo "Check results in results.csv"
    else
        echo ""
        echo "========================================"
        echo "Error: Simulation failed during execution."
        exit 1
    fi
else
    echo "Compilation failed!"
    echo "CPU fallback: g++ -O3 -std=c++17 -fopenmp lunar_sim.cpp -o simulator_cpu"
    exit 1
fi
