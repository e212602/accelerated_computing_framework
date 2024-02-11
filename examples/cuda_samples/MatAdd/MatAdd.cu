#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>


__global__ void MatAdd(float *A, float *B, float *C) {
    
    int i = gridDim.y*blockDim.y*blockDim.x*blockIdx.x + gridDim.y*blockDim.y*threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    for(int it = 0; it < 1; it++)
      C[i + j] = A[i + j] + B[i + j];
}


int main(int argc, char **argv){
    // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;
  float *h_A, *h_B, *h_C, *d_A = NULL, *d_B = NULL, *d_C = NULL;

  // Print the vector length to be used, and compute its size
  int N = 1024;
  int numThreads = 16;
  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("      -nN (Vector Size)\n");
    printf("      -nT (Number of Threads per Block)\n");
    exit(EXIT_SUCCESS);
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "nN")){
    N = getCmdLineArgumentInt(argc, (const char **)argv, "nN");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "nT")){
    numThreads = getCmdLineArgumentInt(argc, (const char **)argv, "nT");
  }

  size_t size = N * N * sizeof(float);
  int numBlocks = N / numThreads;

  //printf("[Matrix addition of size %dx%d]\n", N,N);

  // Allocate the host input Matrix A
  h_A = (float *) malloc(size);

  // Allocate the host input Matrix B
  h_B = (float *) malloc(size);

  // Allocate the host output Matrix C
  h_C = (float *) malloc(size);

  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host Matrices!\n");
    exit(EXIT_FAILURE);
  }
  

  // Initialize the host input Matrices
  for (int i = 0; i < N; ++i) {
    for(int j = 0; j < N; ++j){
      h_A[i*N + j] = rand() / (float)RAND_MAX;
      h_B[i*N + j] = rand() / (float)RAND_MAX;
    }
  }
  // Allocate the device input Matrix A
  err = cudaMalloc((void **)&d_A, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input Matrix B
  err = cudaMalloc((void **)&d_B, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output Matrix C
  err = cudaMalloc((void **)&d_C, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
  // printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // int numThreads = 16;
  // int numBlocks = N / numThreads;
  dim3 DimGrid(numBlocks,numBlocks,1);
  dim3 threadsPerBlock(numThreads, numThreads, 1);
  MatAdd<<<DimGrid, threadsPerBlock>>>(d_A, d_B, d_C);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  //printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Verify that the result vector is correct
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j)
    if (fabs(h_A[i*N + j] + h_B[i*N + j] - h_C[i*N + j]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d, %d!\n", i, j);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = cudaFree(d_A);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
  

  //printf("Done\n");
  return 0;


}