#include <stdio.h>

#define N 2

// Kernel function for matrix multiplication
__global__ void matrixMul(int *a, int *b, int *c) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += a[row * N + i] * b[i * N + col];
    }

    c[row * N + col] = sum;
}

int main() {
    int a[N][N] = { {1, 2}, {3, 4} };
    int b[N][N] = { {5, 6}, {7, 8} };
    int c[N][N] = { 0 };

    int *d_a, *d_b, *d_c;
    size_t size = N * N * sizeof(int);

    // Allocate device memory
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    // Copy matrices from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel with a 2x2 block
    dim3 threadsPerBlock(N, N);
    matrixMul<<<1, threadsPerBlock>>>(d_a, d_b, d_c);

    // Copy result matrix from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result
    printf("Matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
