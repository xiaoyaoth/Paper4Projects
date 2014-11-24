
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_functions.h"

#include <stdio.h>
#include <stdlib.h>


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = sqrtf(log(double(a[i] * b[i])));
}

int main()
{
	int N = 102400;
	int *a = (int*)malloc(sizeof(int) * N * 2);
	int *b = (int*)malloc(sizeof(int) * N * 2);
	int *c = (int*)malloc(sizeof(int) * N * 2);

	for (int i = 0; i < N; i++) {
		a[i] = 1;
		b[i] = 2;
	}

	int *a_dev, *b_dev, *c_dev;
	cudaMalloc((void**)&a_dev, N * 2 * sizeof(int));
	cudaMalloc((void**)&b_dev, N * 2 * sizeof(int));
	cudaMalloc((void**)&c_dev, N * 2 * sizeof(int));

	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	for(int i = 0; i < 100; i++) {
	addKernel<<<32, 32, 0, stream1>>>(c, a, b);
	addKernel<<<32, 32, 0, stream2>>>(&c[N], &a[N], &b[N]);
	}

	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

    return 0;
}
