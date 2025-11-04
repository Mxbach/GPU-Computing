#include <random>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

void init(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    // std::random_device dev;
    std::mt19937 prng(2024);
    std::uniform_int_distribution<int32_t> distrib(-16, 16);

    for (auto i = 0; i < size; i++)
    {
        vec_a[i] = distrib(prng);
        vec_b[i] = distrib(prng);
    }

    for (auto i = 0; i < size * size; i++)
        mat[i] = distrib(prng);
}

__global__ void add_vectors(int32_t *a, int32_t *b, int32_t size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < size)
    {
        b[idx] = a[idx] + b[idx];
    }
}

__global__ void matmul(int32_t *mat, int32_t *b, int32_t *out, int32_t size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < size)
    {
        out[idx] = 0;
        for(int i = 0; i < size; i++)
            out[idx] += mat[idx * size + i] * b[i];
    }
}

void pretty_print(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    if (vec_a != NULL)
    {
        std::cout << "Vec A:" << std::endl;
        for (auto i = 0; i < size; i++)
            std::cout << vec_a[i] << std::endl;
        std::cout << "\n" << std::endl;
    }
    if (vec_b != NULL)
    {
        std::cout << "Vec B:" << std::endl;
        for (auto i = 0; i < size; i++)
            std::cout << vec_b[i] << std::endl;
        std::cout << "\n" << std::endl;

    }
    if (mat != NULL)
    {
        std::cout << "Matrix:" << std::endl;
        for (auto i = 0; i < size; i++)
        {
            for (auto j = 0; j < size; j++)
                std::cout << mat[i * size + j] << " ";

            std::cout << std::endl;
        }
        std::cout << "\n";
    }
}

int main()
{
    // int32_t size = 3;
    int32_t size = 32768;

    auto vec_a = (int32_t *)malloc(sizeof(int32_t) * size);
    auto vec_b = (int32_t *)malloc(sizeof(int32_t) * size);
    // Flat Buffer for matrix
    auto mat = (int32_t *)malloc(sizeof(int32_t *) * size * size);
    auto out = (int32_t *)malloc(sizeof(int32_t) * size);

    init(size, vec_a, vec_b, mat);

    // pretty_print(size, vec_a, vec_b, mat);

    int32_t *d_a, *d_b, *d_mat, *d_out;
    cudaMalloc(&d_a, size * sizeof(int32_t));
    cudaMalloc(&d_b, size * sizeof(int32_t));
    cudaMalloc(&d_mat, size * size * sizeof(int32_t));
    cudaMalloc(&d_out, size * sizeof(int32_t));

    cudaMemcpy(d_a, vec_a, size * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, vec_b, size * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, mat, size * size * sizeof(int32_t), cudaMemcpyHostToDevice);

    /* 
    // initial config
    auto start = std::chrono::system_clock::now();
    add_vectors<<<size, 1>>>(d_a, d_b, size);
    matmul<<<size, 1>>>(d_mat, d_b, d_out, size);
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    //*/


    ///*
    // best performing config
    int threads_per_block = 32;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    auto start = std::chrono::system_clock::now();
    add_vectors<<<blocks, threads_per_block>>>(d_a, d_b, size);
    matmul<<<blocks, threads_per_block>>>(d_mat, d_b, d_out, size);
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    //*/

    cudaMemcpy(out, d_out, size * sizeof(int32_t), cudaMemcpyDeviceToHost);

    std::cout << "First 3 entries of Out Vec:" << std::endl;
    for (int32_t i = 0; i < 3; i++)
        std::cout << out[i] << std::endl;

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    free(vec_a);
    free(vec_b);
    free(mat);
    free(out);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_mat);
    cudaFree(d_out);

    return 0;
}
