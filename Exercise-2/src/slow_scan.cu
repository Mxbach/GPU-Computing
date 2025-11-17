#include <chrono>
#include <curand.h>
#include <iostream>
#include <stdlib.h>

#include "helper.cu"

#define BLOCK_SIZE 1024

void sequential_scan(size_t size, float *in_h, float *out_h) {
  out_h[0] = in_h[0];
  out_h[1] = in_h[1];
  for (auto i = 2; i < size; i += 2) {
    float real_prev = out_h[i - 2];
    float real_cur = in_h[i];
    float im_prev = out_h[i - 1];
    float im_cur = in_h[i + 1];

    out_h[i] = real_prev * real_cur - im_prev * im_cur;
    out_h[i + 1] = real_prev * im_cur + real_cur * im_prev;
  }
}

__global__ void parallel_scan_phase_1(size_t size, float *in_d, float *out_d, float *block_sums_d, int num_blocks) {
  __shared__ float temp[BLOCK_SIZE*2];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t num_pairs = size / 2;

  if (idx < num_pairs) {
    temp[tid*2] = in_d[idx*2];
    temp[tid*2 + 1] = in_d[idx*2 + 1];
  }
  __syncthreads();

  for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    float val_real = 0;
    float val_im = 0;
    if (tid >= stride) {
      int idx_prev = tid - stride;
      int idx_cur = tid;

      float real_prev = temp[idx_prev*2];
      float real_cur = temp[idx_cur*2];
      float im_prev = temp[idx_prev*2 + 1];
      float im_cur = temp[idx_cur*2 + 1];

      val_real = real_prev * real_cur - im_prev * im_cur;
      val_im = real_prev * im_cur + real_cur * im_prev;
    }
    __syncthreads();

    if (tid >= stride) {
      temp[tid*2] = val_real;
      temp[tid*2 + 1] = val_im;
    }

    __syncthreads();
  }
  if (idx < num_pairs) {
    out_d[idx*2] = temp[tid*2];
    out_d[idx*2 + 1] = temp[tid*2 + 1];
  }

  if (tid == BLOCK_SIZE - 1 && blockIdx.x < num_blocks || (blockIdx.x == num_blocks - 1) && tid == ((num_pairs - 1) % BLOCK_SIZE)) {
    block_sums_d[blockIdx.x*2] = temp[tid*2];
    block_sums_d[blockIdx.x*2 + 1] = temp[tid*2 + 1];
  }
}

__global__ void parallel_scan_phase_2(float* block_sums_d, int num_blocks) {
  __shared__ float temp[BLOCK_SIZE*2];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if (idx < num_blocks) {
    temp[tid*2] = block_sums_d[idx*2];
    temp[tid*2 + 1] = block_sums_d[idx*2 + 1];
  }
  __syncthreads();

  for (int stride = 1; stride < num_blocks; stride *= 2) {
    float val_real = 0;
    float val_im = 0;
    if (tid >= stride && tid < num_blocks) {
      int idx_prev = tid - stride;
      int idx_cur = tid;

      float real_prev = temp[idx_prev*2];
      float real_cur = temp[idx_cur*2];
      float im_prev = temp[idx_prev*2 + 1];
      float im_cur = temp[idx_cur*2 + 1];

      val_real = real_prev * real_cur - im_prev * im_cur;
      val_im = real_prev * im_cur + real_cur * im_prev;
    }
    __syncthreads();

    if (tid >= stride && tid < num_blocks) {
      temp[2*tid] = val_real;
      temp[2*tid + 1] = val_im;
    }

    __syncthreads();
  }

  if (idx < num_blocks) {
    block_sums_d[idx*2] = temp[idx*2];
    block_sums_d[idx*2 + 1] = temp[idx*2 + 1];
  }
}

__global__ void parallel_scan_phase_3(size_t size, float *out_d, float *block_sums_d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.x;
  size_t num_pairs = size / 2;

  __shared__ float temp[2];

  if(threadIdx.x == 0 && bid > 0) {
    temp[0] = block_sums_d[(bid-1) * 2];
    temp[1] = block_sums_d[(bid-1) * 2 + 1];
  }
  __syncthreads();

  if (idx < num_pairs && bid > 0) {
    float real_cur = out_d[idx*2];
    float im_cur = out_d[idx*2 + 1];
    
    out_d[idx*2] = temp[0] * real_cur - temp[1] * im_cur;
    out_d[idx*2 + 1] = temp[0] * im_cur + real_cur * temp[1];
  }
}

int parallel_scan_multi_block(size_t size, float *in_d, float *out_d) {
  size_t num_pairs = size / 2;
  size_t num_blocks = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
  std::cout << "num_blocks: " << num_blocks << std::endl;

  float *block_sums_d;
  CUDA_CALL(cudaMalloc((void **)&block_sums_d, num_blocks * 2 * sizeof(float)));

  parallel_scan_phase_1<<<num_blocks, BLOCK_SIZE>>>(size, in_d, out_d, block_sums_d, num_blocks);
  cudaDeviceSynchronize();

  if (num_blocks > 1) {
    if (num_blocks <= BLOCK_SIZE) {
      parallel_scan_phase_2<<<1, BLOCK_SIZE>>>(block_sums_d, num_blocks);
      cudaDeviceSynchronize();
    } else {
      float *temp_d;
      CUDA_CALL(cudaMalloc((void **)&temp_d, num_blocks * 2 * sizeof(float)));
      
      parallel_scan_multi_block(num_blocks * 2, block_sums_d, temp_d);
      
      CUDA_CALL(cudaMemcpy(block_sums_d, temp_d, num_blocks * 2 * sizeof(float), 
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
  size_t size = 33554432 * 2;
  float *in_d, *in_h, *out_d, *out_h;

  // Allocate on host
  in_h = (float *)calloc(size, sizeof(float));
  CHECK_ALLOC(in_h);
  out_h = (float *)calloc(size, sizeof(float));
  CHECK_ALLOC(out_h);
  // Allocate on device
  CUDA_CALL(cudaMalloc((void **)&in_d, size * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&out_d, size * sizeof(float)));

  // Initialize
  int e = random_init(size, in_d, in_h);
  if (e == EXIT_FAILURE)
    return EXIT_FAILURE;

  bool use_cuda = true;
  if(use_cuda) {
    CUDA_CALL(cudaMemcpy(in_d, in_h, size * sizeof(float), cudaMemcpyHostToDevice));
  }
  auto start = std::chrono::system_clock::now();
  if(use_cuda)
    parallel_scan_multi_block(size, in_d, out_d);
  else
    sequential_scan(size, in_h, out_h);
  auto end = std::chrono::system_clock::now();

  if(use_cuda)
    CUDA_CALL(cudaMemcpy(out_h, out_d, size * sizeof(float), cudaMemcpyDeviceToHost));

  std::cout << "First 3 entries of In Vec:" << std::endl;
  for (int32_t i = 0; i < 5 * 2; i += 2)
    std::cout << in_h[i] << "," << in_h[i + 1] << std::endl;
  std::cout << "First 3 entries of Out Vec:" << std::endl;
  for (int32_t i = 0; i < 5 * 2; i += 2)
    std::cout << out_h[i] << " + " << out_h[i + 1] << std::endl;

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

  CUDA_CALL(cudaFree(in_d));
  CUDA_CALL(cudaFree(out_d));
  free(in_h);
  free(out_h);
  return EXIT_SUCCESS;
}
