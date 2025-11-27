#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#define CHECK_CUDA(call)                                        \
    if ((call) != cudaSuccess)                                  \
    {                                                           \
        std::cerr << "CUDA error at " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                     \
    }

const int NUM_MATRICES = 10; // Number of matrix multiplications
const int MATRIX_SIZE = 4096;
const int TILE_SIZE = 32;

void random_init(float* M, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> distrib(1, 15);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[i * n + j] = distrib(gen);
        }
    }
}

void cpu_matmul(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
        if (i % 32 == 0)
            std::cout << "cpu_matmul i: " << i << std::endl; 
    }
}

int compare_matrices(float *A1, float *A2, int n) {
    const float eps = 1.e-6;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float diff = fabs(A1[i * n + j] - A2[i * n + j]);
            float abs_val = fmax(fabs(A1[i * n + j]), fabs(A2[i * n + j]));
            float rel_err = (abs_val > eps) ? diff / abs_val : diff;

            if (rel_err > eps) {
                std::cerr << "Mismatch at i:" << i << ", j:" << j << std::endl;
                return -1;
            }
        }
    }
    return 0;
}

// Simple kernel for matrix multiplication
__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int n)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tiled kernel for matrix multiplication
__global__ void matrixMultiplyKernelTiled(const float *A, const float *B, float *C, int n) {
    // Allocate shared memory for two tiles (one for A and one for B)
    __shared__ float Asub[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float Cvalue = 0.0;

    // Iterate over tiles
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Copy tiles from global memory into shared memory
        int ai = t * TILE_SIZE + threadIdx.x;
        int bi = t * TILE_SIZE + threadIdx.y;

        if (row < n && ai < n) {
            Asub[threadIdx.y][threadIdx.x] = A[row * n + ai];
        } else {
            Asub[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (bi < n && col < n) {
            Bsub[threadIdx.y][threadIdx.x] = B[bi * n + col];
        } else {
            Bsub[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        // Compute the matrix multiplication of the two tiles
        for (int j = 0; j < TILE_SIZE; j++) {
            Cvalue += Asub[threadIdx.y][j] * Bsub[j][threadIdx.x];
        }
        __syncthreads();
    }

    // Write back the results into the matrix C
    if (row < n && col < n) {
        C[row * n + col] = Cvalue;
    }
}

void matrixMultiplyNoStreams() {
    // Host and device pointers
    float *h_A[NUM_MATRICES], *h_B[NUM_MATRICES], *h_C[NUM_MATRICES];
    float *d_A[NUM_MATRICES], *d_B[NUM_MATRICES], *d_C[NUM_MATRICES];

    // custom validation
    // float *val_C[NUM_MATRICES];
    //

    for (int i = 0; i < NUM_MATRICES; i++)
    {
        h_A[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        h_B[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        h_C[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

        // Initialize example matrices with random numbers
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++) {
            // pick testing values, that allow us to compute the expected result on the CPU cheaply
            h_A[i][j] = 1.0f;
            h_B[i][j] = 0.01f;
            h_C[i][j] = 0.0f;
        }

        // custom validation
        // random_init(h_A[i], MATRIX_SIZE);
        // random_init(h_B[i], MATRIX_SIZE);

        // val_C[i] = (float *)calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(float));
        // cpu_matmul(h_A[i], h_B[i], val_C[i], MATRIX_SIZE);
        //

        CHECK_CUDA(cudaMalloc(&d_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));

        // Copy matrices A and B to the device
        CHECK_CUDA(cudaMemcpy(d_A[i], h_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B[i], h_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // Launch matrix multiplication kernel
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 blocksPerGrid(MATRIX_SIZE/TILE_SIZE, MATRIX_SIZE/TILE_SIZE);

        std::cout << "Launch kernel with " << blocksPerGrid.x * blocksPerGrid.y << " blocks each with " << threadsPerBlock.x * threadsPerBlock.y << " threads\n";
        // matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        matrixMultiplyKernelTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        CHECK_CUDA(cudaGetLastError());

        // Copy results back to the host
        CHECK_CUDA(cudaMemcpy(h_C[i], d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        // Verify results
        double eps = 1.e-6;  // machine zero
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++) {
            double abs_err = fabs(h_C[i][j] - (MATRIX_SIZE * 0.01f));
            double dot_length = MATRIX_SIZE;
            double abs_val = fabs(h_C[i][j]);
            double rel_err = abs_err / abs_val / dot_length;

            if (rel_err > eps) {
                printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    j, h_C[i][j], MATRIX_SIZE * 0.01f, eps);
            }
        }

        // custom validation
        // if (compare_matrices(h_C[i], val_C[i], MATRIX_SIZE) != 0) {
        //     std::cout << "Validation failed!" << std::endl;
        // } else {
        //     std::cout << "Validation passed!" << std::endl;
        // }
        // free(val_C[i]);
        //

        // Cleanup
        free(h_A[i]);
        free(h_B[i]);
        free(h_C[i]);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
}

void matrixMultiplyWithStreams()
{
    // Host and device pointers
    float *h_A[NUM_MATRICES], *h_B[NUM_MATRICES], *h_C[NUM_MATRICES];
    float *d_A[NUM_MATRICES], *d_B[NUM_MATRICES], *d_C[NUM_MATRICES];
    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    int NUM_STREAMS = NUM_MATRICES; 
    cudaStream_t streams[NUM_STREAMS];

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE, (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE);

    for (int i = 0; i < NUM_STREAMS; i++) {
        // Allocate memory, initialize data, create streams and copy data asynchronously
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        CHECK_CUDA(cudaMallocHost(&h_A[i], size));
        CHECK_CUDA(cudaMallocHost(&h_B[i], size));
        CHECK_CUDA(cudaMallocHost(&h_C[i], size));

        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++) {
            h_A[i][j] = 1.0f;
            h_B[i][j] = 0.01f;
            h_C[i][j] = 0.0f;
        }

        CHECK_CUDA(cudaMalloc(&d_A[i], size));
        CHECK_CUDA(cudaMalloc(&d_B[i], size));
        CHECK_CUDA(cudaMalloc(&d_C[i], size));
        
        CHECK_CUDA(cudaMemcpyAsync(d_A[i], h_A[i], size, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_B[i], h_B[i], size, cudaMemcpyHostToDevice, streams[i]));

        // Launch matrix multiplication kernel for each stream


        std::cout << "Launch kernel with " << blocksPerGrid.x * blocksPerGrid.y << " blocks each with " << threadsPerBlock.x * threadsPerBlock.y << " threads\n";
        matrixMultiplyKernelTiled<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        CHECK_CUDA(cudaGetLastError());

        // Copy results back to the host asynchronously
        CHECK_CUDA(cudaMemcpyAsync(h_C[i], d_C[i], size, cudaMemcpyDeviceToHost, streams[i]));
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // Verify results (slow! use only for debugging)
    // for (int i = 0; i < 1; i++) {
    //     std::cout << "Matrix C[" << i << "]:" << std::endl;
    //     for (int row = 0; row < MATRIX_SIZE; row++)
    //     {
    //         for (int col = 0; col < MATRIX_SIZE; col++)
    //         {
    //             std::cout << h_C[i][row * MATRIX_SIZE + col] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFreeHost(h_A[i]);
        cudaFreeHost(h_B[i]);
        cudaFreeHost(h_C[i]);

        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);

        cudaStreamDestroy(streams[i]);
    }
}

int main()
{
    matrixMultiplyWithStreams();
    // matrixMultiplyNoStreams();
    return EXIT_SUCCESS;
}
