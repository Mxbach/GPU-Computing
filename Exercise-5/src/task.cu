#include <iostream>
#include <iomanip>

#define N 128                   // Grid size X
#define M 128                   // Grid size Y
#define ITERATIONS 100000       // Number of iterations
#define DIFFUSION_FACTOR 0.5    // Diffusion factor
#define CELL_SIZE 0.01          // Cell size for the simulation

#define CUDA_CALL(x)                                                                                          \
    do                                                                                                        \
    {                                                                                                         \
        cudaError_t error = x;                                                                                \
        if (error != cudaSuccess)                                                                             \
        {                                                                                                     \
            const char *cuda_err_str = cudaGetErrorString(error);                                             \
            std::cerr << "Cuda Error at" << __FILE__ << ":" << __LINE__ << ": " << cuda_err_str << std::endl; \
            return EXIT_FAILURE;                                                                              \
        }                                                                                                     \
    } while (0)

void initializeGrid(float *grid, int n, int m)
{
    for (int y = 0; y < m; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            // Initialize one quadrant to a high temp
            // and the rest to 0.
            if (y > m / 2 && x > n / 2)
            {
                grid[y * m + x] = 100.0f; // Temp in corner
            }
            else
            {
                grid[y * m + x] = 0.0f; // Temp in the rest
            }
        }
    }
}

__global__ void heatSimKernel(float *a, float *b, int n, int m, float dx2, float dy2, float dt) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n  && (row == 0 || row == m-1 || col == 0 || col == n-1)) {
        b[row * m + col] = a[row * m + col];
    }

    if (0 < row && row < m-1 && 0 < col && col < n-1) {
        float left = a[row * m + col - 1];
        float right = a[row * m + col + 1];
        float below = a[(row-1) * m + col];
        float above = a[(row+1) * m + col];
        float center = a[row * m + col];
        b[row * m + col] = center + DIFFUSION_FACTOR * dt *
                                    ((left - 2.0 * center + right) / dy2 +
                                    (above - 2.0 * center + below) / dx2);
    }
}

int main() {
    std::cout << "cuda" << std::endl;

    size_t size = N * M * sizeof(float);
    // Allocate CPU memory
    float *h_matrix = (float*) malloc(N * M * sizeof(float));

    if (h_matrix == NULL) {
        std::cerr << "malloc fail" << std::endl;
    }

    initializeGrid(h_matrix, N, M);

    float dx2 = CELL_SIZE * CELL_SIZE;
    float dy2 = CELL_SIZE * CELL_SIZE;
    float dt = dx2 * dy2 / (2.0 * DIFFUSION_FACTOR * (dx2 + dy2));

    // Allocate GPU memory
    float *d_curr;
    float *d_next;
    CUDA_CALL(cudaMalloc(&d_curr, size));
    CUDA_CALL(cudaMalloc(&d_next, size)); 
    CUDA_CALL(cudaMemcpy(d_curr, h_matrix, size, cudaMemcpyHostToDevice));

    int threads_per_block = 32;
    int blocks_x = (N + threads_per_block - 1) / threads_per_block;
    int blocks_y = (M + threads_per_block - 1) / threads_per_block;
    dim3 block_dim(threads_per_block, threads_per_block);
    dim3 grid_dim(blocks_x, blocks_y);

    float *temp;
    for (int i = 0; i < ITERATIONS; i++) {
        heatSimKernel<<<grid_dim, block_dim>>>(d_curr, d_next, N, M, dx2, dy2, dt);
        CUDA_CALL(cudaDeviceSynchronize());
        temp = d_curr;
        d_curr = d_next;
        d_next = temp;
    }

    CUDA_CALL(cudaMemcpy(h_matrix, d_curr, size, cudaMemcpyDeviceToHost));

    // output 
    std::cout << "Final grid values (top-left corner):" << std::endl;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << h_matrix[i * M + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_curr);
    cudaFree(d_next);
    free(h_matrix);

    return 0;
}