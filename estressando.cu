#include <stdio.h>
#include <cuda.h>

// Definição do tamanho da matriz (pode ser ajustado conforme necessário)
#define N 1024  // Tamanho das matrizes NxN

// Kernel CUDA para multiplicação de matrizes
__global__ void matrixMultiply(float *a, float *b, float *c, int width) {
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

int main() {
    // Alocação de memória para matrizes no host
    float *h_a, *h_b, *h_c;
    int size = N * N * sizeof(float);
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c = (float *)malloc(size);

    // Inicializando matrizes com valores arbitrários
    for (int i = 0; i < N * N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }

    // Alocação de memória para matrizes no device (GPU)
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copiando matrizes do host para o device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Definindo a grade e a organização dos blocos
    dim3 threadsPerBlock(16, 16);  // Bloco 16x16 threads
    dim3 blocksPerGrid(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Iniciando a multiplicação de matrizes na GPU
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copiando o resultado de volta para o host
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
    printf("Multiplicação de matrizes concluída!\n");

    return 0;
}
