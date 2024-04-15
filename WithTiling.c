%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_SIZE 16

__global__ void MatrixMulSquareKernel(float* M, float* N, float* P, int Width) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  
  int Row = by * TILE_SIZE + ty;
  int Col = bx * TILE_SIZE + tx;

  __shared__ float sharedM[TILE_SIZE][TILE_SIZE];
  __shared__ float sharedN[TILE_SIZE][TILE_SIZE];

  float Pvalue = 0;
  for (int i = 0; i < Width / TILE_SIZE; ++i) {
    sharedM[ty][tx] = M[Row * Width + i * TILE_SIZE + tx];
    sharedN[ty][tx] = N[(i * TILE_SIZE + ty) * Width + Col];
    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      Pvalue += sharedM[ty][k] * sharedN[k][tx];
    }
    __syncthreads();
  }
  P[Row * Width + Col] = Pvalue;
}

__global__ void MatrixMulRectangularKernel(float* M, float* N, float* P, int WidthM, int HeightM, int WidthN, int HeightN) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int Row = by * TILE_SIZE + ty;
  int Col = bx * TILE_SIZE + tx;

  __shared__ float sharedM[TILE_SIZE][TILE_SIZE];
  __shared__ float sharedN[TILE_SIZE][TILE_SIZE];

  float Pvalue = 0;
  for (int i = 0; i < WidthM / TILE_SIZE; ++i) {
    sharedM[ty][tx] = M[Row * WidthM + i * TILE_SIZE + tx];
    sharedN[ty][tx] = N[(i * TILE_SIZE + ty) * WidthN + Col];
    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      Pvalue += sharedM[ty][k] * sharedN[k][tx];
    }
    __syncthreads();
  }
  P[Row * WidthN + Col] = Pvalue;
}

int main() {
  int WidthM = 1024;
  int HeightM = 768;
  int WidthN = 1024;
  int HeightN = 768;

  size_t matrixSizeM = WidthM * HeightM * sizeof(float);
  size_t matrixSizeN = WidthN * HeightN * sizeof(float);
  size_t matrixSizeP;

  float *h_M, *h_N, *h_P;
  h_M = (float*)malloc(matrixSizeM);
  h_N = (float*)malloc(matrixSizeN);
  h_P = (float*)malloc(matrixSizeP);

  // Initialize matrices M and N with your data

  float *d_M, *d_N, *d_P;
  cudaMalloc((void**)&d_M, matrixSizeM);
  cudaMalloc((void**)&d_N, matrixSizeN);
  cudaMalloc((void**)&d_P, matrixSizeP);

  cudaMemcpy(d_M, h_M, matrixSizeM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N, matrixSizeN, cudaMemcpyHostToDevice);

  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim;

  if (WidthM == HeightM && WidthN == HeightN) {
    matrixSizeP = WidthM * HeightN * sizeof(float);
    h_P = (float*)malloc(matrixSizeP);

    cudaMemcpy(d_P, h_P, matrixSizeP, cudaMemcpyHostToDevice);

    gridDim = dim3((WidthM + TILE_SIZE - 1) / TILE_SIZE, (HeightN + TILE_SIZE - 1) / TILE_SIZE);

    clock_t start, stop;
    start = clock();

    MatrixMulSquareKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, WidthM);

    cudaDeviceSynchronize();
    stop = clock();

    cudaMemcpy(h_P, d_P, matrixSizeP, cudaMemcpyDeviceToHost);

    float milliseconds = ((float)(stop - start) / CLOCKS_PER_SEC) * 1000.0;
    printf("Execution Time (Square Matrix): %.4f seconds\n", milliseconds / 1000.0);

    free(h_P);
  } else {
    matrixSizeP = HeightM * WidthN * sizeof(float);
    h_P = (float*)malloc(matrixSizeP);

    cudaMemcpy(d_P, h_P, matrixSizeP, cudaMemcpyHostToDevice);

    gridDim = dim3((WidthN + TILE_SIZE - 1) / TILE_SIZE, (HeightM + TILE_SIZE - 1) / TILE_SIZE);

    clock_t start, stop;
    start = clock();

    MatrixMulRectangularKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, WidthM, HeightM, WidthN, HeightN);

    cudaDeviceSynchronize();
    stop = clock();

    cudaMemcpy(h_P, d_P, matrixSizeP, cudaMemcpyDeviceToHost);

    float milliseconds = ((float)(stop - start) / CLOCKS_PER_SEC) * 1000.0;
    printf("Execution Time (Rectangular Matrix): %.4f seconds\n", milliseconds / 1000.0);

    free(h_P);
  }

  free(h_M);
  free(h_N);

  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);

  return 0;
}
