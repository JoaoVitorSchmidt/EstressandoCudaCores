#include <iostream>
#include <cuda_runtime.h>
#include <chrono>  // Para medir o tempo de execução

// Definição do tamanho da matriz
#define N 2048  // Matriz grande para estressar a GPU

// Kernel CUDA para multiplicação de matrizes
__global__ void matrixMultiply(float* a, float* b, float* c, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < width) {
        float sum = 0.0f;
        for (int i = 0; i < width; i++) {
            sum += a[row * width + i] * b[i * width + col];
        }
        c[row * width + col] = sum;
    }
}

void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = rand() % 100;  // Inicializa com valores aleatórios
    }
}

int main() {
    // Alocação de memória para matrizes no host (CPU)
    float* h_a, * h_b, * h_c;
    int size = N * N * sizeof(float);
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Inicializando as matrizes com valores arbitrários
    initializeMatrix(h_a, N * N);
    initializeMatrix(h_b, N * N);

    // Alocação de memória no device (GPU)
    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copiando as matrizes do host (CPU) para o device (GPU)
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Configurando a grade e os blocos para o kernel CUDA
    dim3 threadsPerBlock(16, 16);  // Blocos de 16x16 threads
    dim3 blocksPerGrid(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Medir o tempo de execução
    auto start = std::chrono::high_resolution_clock::now();

    // Repetir a multiplicação várias vezes para estressar a GPU
    int num_iterations = 100;
    for (int i = 0; i < num_iterations; i++) {
        matrixMultiply << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, N);
    }

    // Sincronizar a GPU para garantir que todos os kernels terminaram
    cudaDeviceSynchronize();

    // Medir o tempo após a execução
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Tempo total para " << num_iterations << " multiplicações: "
        << duration.count() << " segundos" << std::endl;

    // Copiando o resultado de volta para o host (CPU)
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Liberando memória no device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Liberando memória no host
    free(h_a);
    free(h_b);
    free(h_c);

    // Finalizando
    std::cout << "Estresse no CUDA Core concluído!" << std::endl;

    return 0;
}
