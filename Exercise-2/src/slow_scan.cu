#include <chrono>
#include <curand.h>
#include <iostream>
#include <stdlib.h>

#include "helper.cu"

#define BLOCK_SIZE 1024

void sequential_scan(size_t size, int *in_h, int *out_h) {
  out_h[0] = in_h[0];
  for (auto i = 1; i < size; i++) {
    out_h[i] = in_h[i] + out_h[i-1];
  }
}

__global__ void parallel_scan_phase_1(size_t size, int *in_d, int *out_d, int *block_sums_d, int num_blocks) {
  __shared__ int temp[BLOCK_SIZE];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    temp[tid] = in_d[idx];
  } else {
    temp[tid] = 0;
  }
  __syncthreads();

  for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    int val = 0;
    if (tid >= stride) {
      val += temp[tid] + temp[tid - stride];
    }
    __syncthreads();

    if (tid >= stride) {
      temp[tid] = val;
    }
    __syncthreads();
  }

  if (idx < size) {
    out_d[idx] = temp[tid];
  }

  if (tid == BLOCK_SIZE - 1 && blockIdx.x < num_blocks) {
    block_sums_d[blockIdx.x] = temp[tid];
  }
}

__global__ void parallel_scan_phase_2(int* block_sums_d, int num_blocks) {
  __shared__ int temp[BLOCK_SIZE];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if (idx < num_blocks) {
    temp[tid] = block_sums_d[idx];
  } else {
    temp[tid] = 0;
  }
  __syncthreads();

  for (int stride = 1; stride < num_blocks; stride *= 2) {
    int val = 0;
    if (tid >= stride && tid < num_blocks) {
      val = temp[tid] + temp[tid - stride];
    }
    __syncthreads();

    if (tid >= stride && tid < num_blocks) {
      temp[tid] = val;
    }

    __syncthreads();
  }

  if (idx < num_blocks) {
    block_sums_d[idx] = temp[tid];
  }
}

__global__ void parallel_scan_phase_3(size_t size, int *out_d, int *block_sums_d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.x;

  if (idx < size && bid > 0) {
    out_d[idx] += block_sums_d[bid - 1];
  }
}

int parallel_scan_multi_block(size_t size, int *in_d, int *out_d) {
  std::cout << "using cuda" << std::endl;
  size_t num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  std::cout << "num_blocks: " << num_blocks << std::endl;

  int *block_sums_d;
  CUDA_CALL(cudaMalloc((void **)&block_sums_d, num_blocks * sizeof(int)));

  parallel_scan_phase_1<<<num_blocks, BLOCK_SIZE>>>(size, in_d, out_d, block_sums_d, num_blocks);
  cudaDeviceSynchronize();

  if (num_blocks > 1) {
    if (num_blocks <= BLOCK_SIZE) {
      parallel_scan_phase_2<<<1, BLOCK_SIZE>>>(block_sums_d, num_blocks);
      cudaDeviceSynchronize();
    } else {
      int *temp_d;
      CUDA_CALL(cudaMalloc((void **)&temp_d, num_blocks * sizeof(int)));
      
      parallel_scan_multi_block(num_blocks , block_sums_d, temp_d);
      
      CUDA_CALL(cudaMemcpy(block_sums_d, temp_d, num_blocks * sizeof(int), 
                           cudaMemcpyDeviceToDevice));
      CUDA_CALL(cudaFree(temp_d));
    }
  }

  if (num_blocks > 1) {
    parallel_scan_phase_3<<<num_blocks, BLOCK_SIZE>>>(size, out_d, block_sums_d);
    cudaDeviceSynchronize();
  }

  CUDA_CALL(cudaFree(block_sums_d));

  return EXIT_SUCCESS;
}

int main() {
  size_t size = 1048576;
  int *in_d, *in_h, *out_d, *out_h;

  // Allocate on host
  in_h = (int *)calloc(size, sizeof(int));
  CHECK_ALLOC(in_h);
  out_h = (int *)calloc(size, sizeof(int));
  CHECK_ALLOC(out_h);
  // Allocate on device
  CUDA_CALL(cudaMalloc((void **)&in_d, size * sizeof(int)));
  CUDA_CALL(cudaMalloc((void **)&out_d, size * sizeof(int)));

  // Initialize
  // int e = random_init(size, in_d, in_h);
  // if (e == EXIT_FAILURE)
  //   return EXIT_FAILURE;

  for (int i = 0; i < size; i++) {
    in_h[i] = 1;
  }

  bool use_cuda = true;
  if(use_cuda) {
    CUDA_CALL(cudaMemcpy(in_d, in_h, size * sizeof(int), cudaMemcpyHostToDevice));
  }
  auto start = std::chrono::system_clock::now();
  if(use_cuda)
    parallel_scan_multi_block(size, in_d, out_d);
  else
    sequential_scan(size, in_h, out_h);
  auto end = std::chrono::system_clock::now();

  if(use_cuda)
    CUDA_CALL(cudaMemcpy(out_h, out_d, size * sizeof(int), cudaMemcpyDeviceToHost));

  // int number_of_prints = 20;
  std::cout << "First 3 entries of In Vec:" << std::endl;
  for (int32_t i = 0; i < 3; i++)
    std::cout << in_h[i] << std::endl;
  std::cout << "First 3 entries of Out Vec:" << std::endl;
  for (int32_t i = 0; i < 3; i++)
    std::cout << out_h[i] << std::endl;

  std::cout << "Last 3 entries of Out Vec:" << std::endl;
  for (size_t i = size - 3; i < size; i++) {
      std::cout << out_h[i] << std::endl;
  }

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

  CUDA_CALL(cudaFree(in_d));
  CUDA_CALL(cudaFree(out_d));
  free(in_h);
  free(out_h);
  return EXIT_SUCCESS;
}
