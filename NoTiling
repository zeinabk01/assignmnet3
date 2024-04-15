%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void MatrixMulSquareKernel(float* M, float* N, float* P, int Width) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((Row < Width) && (Col < Width)) {
    float Pvalue = 0;
    for (int k = 0; k < Width; ++k) {
      Pvalue += M[Row * Width + k] * N[k * Width + Col];
    }
    P[Row * Width + Col] = Pvalue;
  }
}

// Rectangular matrix multiplication kernel
__global__ void MatrixMulRectangularKernel(float* M, float* N, float* P, int WidthM, int HeightM, int WidthN, int HeightN) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((Row < HeightM) && (Col < WidthN)) {
    float Pvalue = 0;
    for (int k = 0; k < WidthM; ++k) {
      Pvalue += M[Row * WidthM + k] * N[k * WidthN + Col];
    }
    P[Row * WidthN + Col] = Pvalue;
  }
}
int main() {
  int WidthM = 800;
  int HeightM = 800;
  int WidthN = 800;
  int HeightN = 800;

  size_t matrixSizeM = WidthM * HeightM * sizeof(float);
  size_t matrixSizeN = WidthN * HeightN * sizeof(float);
  size_t matrixSizeP;

  float *h_M, *h_N, *h_P;
  h_M = (float*)malloc(matrixSizeM);
  h_N = (float*)malloc(matrixSizeN);
  h_P = nullptr;

  float *d_M, *d_N, *d_P;
  cudaMalloc((void**)&d_M, matrixSizeM);
  cudaMalloc((void**)&d_N, matrixSizeN);

  cudaMemcpy(d_M, h_M, matrixSizeM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N, matrixSizeN, cudaMemcpyHostToDevice);

  dim3 blockDim(16, 16);
  dim3 gridDim;

  if (WidthM == HeightM && WidthN == HeightN) {
    matrixSizeP = WidthM * HeightN * sizeof(float);
    h_P = (float*)malloc(matrixSizeP);
    cudaMalloc((void**)&d_P, matrixSizeP);
    gridDim = dim3((WidthM + blockDim.x - 1) / blockDim.x, (HeightN + blockDim.y - 1) / blockDim.y);
    MatrixMulSquareKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, WidthM);
    cudaMemcpy(h_P, d_P, matrixSizeP, cudaMemcpyDeviceToHost);
    cudaFree(d_P);
  } else {
    matrixSizeP = HeightM * WidthN * sizeof(float);
    h_P = (float*)malloc(matrixSizeP);
    cudaMalloc((void**)&d_P, matrixSizeP);
    gridDim = dim3((WidthN + blockDim.x - 1) / blockDim.x, (HeightM + blockDim.y - 1) / blockDim.y);
    MatrixMulRectangularKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, WidthM, HeightM, WidthN, HeightN);
    cudaMemcpy(h_P, d_P, matrixSizeP, cudaMemcpyDeviceToHost);
    cudaFree(d_P);
  }

  clock_t start, stop;
  start = clock();

  if (WidthM == HeightM && WidthN == HeightN) {
    MatrixMulSquareKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, WidthM);
  } else {
    MatrixMulRectangularKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, WidthM, HeightM, WidthN, HeightN);
  }

  cudaDeviceSynchronize();
  stop = clock();
  float milliseconds = ((float)(stop - start) / CLOCKS_PER_SEC) * 1000.0;
  printf("Execution Time: %.4f seconds\n", milliseconds / 1000.0);

  free(h_M);
  free(h_N);
  free(h_P);

  cudaFree(d_M);
  cudaFree(d_N);

  return 0;
}

