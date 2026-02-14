#!/bin/bash
# Fowl Benchmark Runner Script
# Run this to establish baseline measurements

set -e

echo "=========================================="
echo "Fowl Benchmark Suite - Baseline Runner"
echo "=========================================="
echo ""

# Check if running in Release mode
if [ "$1" != "--release" ]; then
    echo "WARNING: Run with --release for accurate measurements!"
    echo "Usage: ./run-benchmarks.sh --release"
    echo ""
fi

# Navigate to benchmark directory
cd "$(dirname "$0")"

echo "Building benchmark project..."
dotnet build -c Release

echo ""
echo "Running benchmarks..."
echo "This may take 10-30 minutes depending on your hardware"
echo ""

# Run all benchmarks
dotnet run -c Release -- --artifacts ./results

echo ""
echo "=========================================="
echo "Benchmarks Complete!"
echo "=========================================="
echo ""
echo "Results saved to: ./results"
echo ""
echo "Next steps:"
echo "1. Review results in ./results"
echo "2. Copy baseline results to docs/BASELINE_RESULTS.md"
echo "3. Commit: git add benchmarks/ && git commit -m 'bench(baseline): ...'"
echo ""
