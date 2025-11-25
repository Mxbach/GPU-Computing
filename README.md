# GPU-Computing
Course exercises on GPU computing using C++

## Exercise 1: CUDA Vector and Matrix Operations

GPU-accelerated vector addition and matrix-vector multiplication using CUDA. Includes CPU baseline comparison, benchmarking, and kernel configuration experiments showing significant performance improvements over sequential implementation.

## Exercise 2: Parallel Inclusive Scan (Kogge-Stone Algorithm)

GPU-accelerated inclusive scan (prefix sum) for complex number arrays using the Kogge-Stone algorithm. Complex numbers are represented as adjacent float pairs (real, imaginary), with complex multiplication as the scan operation.

### Optimizations
Progressive implementation of performance optimizations:
1. **Divergence Reduction**: Minimizing warp divergence
2. **Shared Memory Utilization**: Reducing global memory accesses (DONE)
3. **Thread Coarsening**: Processing multiple elements per thread
4. **Memory Coalescing** (optional): Optimizing memory access patterns

Benchmarked against CPU baseline for correctness and performance validation.

## Exercise 3: Tiled Matrix Multiplication and CUDA Streams

GPU-optimized matrix multiplication using tiling for improved memory locality and CUDA streams for overlapping computation and communication. Implements collaborative data loading into shared memory to minimize expensive global memory accesses.

### Tasks
1. **Stream-based Computation**: Overlap communication and computation using CUDA streams for multiple matrix multiplications
2. **Tiled Matrix Multiplication**: Implement optimized kernel using shared memory tiling to reduce memory bandwidth bottlenecks and maximize data reuse

Validated against CPU baseline with performance profiling using NVIDIA Nsight tools.