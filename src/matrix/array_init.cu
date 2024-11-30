/*
 * CUDA Array Initialization Example
 *
 * This program demonstrates basic CUDA concepts:
 * - Memory allocation on host and device
 * - Kernel execution with multiple blocks and threads
 * - Thread index calculation
 * - Memory transfer between host and device
 *
 * Grid Structure:
 * - Total elements: 16
 * - Threads per block: 4
 * - Number of blocks: 4
 *
 * Created: 2024-12-01
 */
#include <stdio.h>

__global__ void initArray(int *arr) {
   // Calculate thread ID (using only x dimension as it's 1D)
   printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d\n", blockIdx.x, blockDim.x, threadIdx.x);
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   printf("Calculated idx = %d\n", idx);

   // Store thread ID in array element
   arr[idx] = idx;
}

int main() {
   const int N = 16;  // Array size
   int *h_arr;        // Host array
   int *d_arr;        // Device array

   // Allocate host memory
   h_arr = (int*)malloc(N * sizeof(int));

   // Allocate device memory
   cudaMalloc(&d_arr, N * sizeof(int));

   // Set kernel execution parameters
   int threadsPerBlock = 4;              // 4 threads per block
   int blocksPerGrid = N / threadsPerBlock;  // Number of blocks = 16/4 = 4

   // Execute kernel
   initArray<<<blocksPerGrid, threadsPerBlock>>>(d_arr);

   // Copy results back to host
   cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

   // Display results
   for(int i = 0; i < N; i++) {
       printf("%d ", h_arr[i]);
   }
   printf("\n");

   // Free memory
   free(h_arr);
   cudaFree(d_arr);

   return 0;
}

// memo:
// GPU
// ├── Grid
// │   ├── Block
// │   │   ├── Thread
