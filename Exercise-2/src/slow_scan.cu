#include <chrono>
#include <curand.h>
#include <iostream>
#include <stdlib.h>

#include "helper.cu"

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

// works for one block
__global__ void parallel_scan(size_t size, float *in_d, float *out_d) {
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  __shared__ float temp[1024];
  int numPairs = size / 2;

  if (tid < numPairs) {
    temp[tid*2] = in_d[tid*2];
    temp[tid*2 + 1] = in_d[tid*2 + 1];
  }

  __syncthreads();

  for (int stride = 1; stride < size; stride*=2) {
    float val_real = 0;
    float val_im = 0;
    if (tid >= stride && tid < numPairs) {
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

    if (tid >= stride && tid < size) {
      temp[2*tid] = val_real;
      temp[2*tid + 1] = val_im;
    }

    __syncthreads();
  }
  if (tid < size) {
    out_d[2*tid] = temp[2*tid];
    out_d[2*tid + 1] = temp[2*tid + 1];
  }

}

int main() {
  // size_t size = 33554432 * 2;
  size_t size = 512 * 2;
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

  // CUDA config
  bool use_cuda = true;
  int threadsPerBlock = 512;
  // int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  int blocksPerGrid = 1;
  if(use_cuda) {
    out_h[0] = in_h[0];
    out_h[1] = in_h[1];
    CUDA_CALL(cudaMemcpy(in_d, in_h, size * sizeof(float), cudaMemcpyHostToDevice));
  }
  auto start = std::chrono::system_clock::now();
  if(use_cuda) {
    parallel_scan<<<blocksPerGrid, threadsPerBlock>>>(size, in_d, out_d);
  }
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
