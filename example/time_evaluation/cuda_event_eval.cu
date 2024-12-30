#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

int main() {
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaStream_t stream;
    
    cudaStreamCreate(&stream);
    
    cudaEventRecord(start, stream);
    
    sleep(2);
    cudaEventRecord(stop, stream);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time elapsed: %f ms\n", milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    
    return 0;
}
