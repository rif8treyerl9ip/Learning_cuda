// scalar_add.cu
#include <stdio.h>

__global__ void ScalarAdd(float* a, float* b, float* c)
{
    *c = *a + *b;
}
__global__ void ScalarMultiply(float* a, float* b, float* c)
{
    *c = *a * *b;
}

int main()
{
    // Host variables
    float h_a = 1.0f;
    float h_b = 2.7f;
    float h_c = 0.0f;

    // Device variables
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sizeof(float));
    cudaMalloc(&d_b, sizeof(float));
    cudaMalloc(&d_c, sizeof(float));

    cudaMemcpy(d_a, &h_a, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice);

    ScalarAdd<<<1, 1>>>(d_a, d_b, d_c);
    cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%.1f + %.1f = %.1f\n", h_a, h_b, h_c);

    ScalarMultiply<<<1, 1>>>(d_a, d_b, d_c);
    cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%.1f * %.1f = %.1f\n", h_a, h_b, h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
