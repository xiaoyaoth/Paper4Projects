
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>

__device__ double myAtomicAnd(int* address, int val)
{
	//int old = *address, assumed;
	//do {
	//	assumed = old;
	//	old = atomicCAS(address, assumed, val & assumed);
	//} while (assumed != old);
	//return old;
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    //c[i] = 
		sqrtf(log(double(a[i] * b[i])));
}

__global__ void testAtomicAndKernel(int *result, int *a) 
{
	//atomicOr(result, a[threadIdx.x]);
	myAtomicAnd(result, a[threadIdx.x]);
}

__global__ void heavyLoadKernel(int *a, int *b, int num) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//if (idx == 0) {
	for (int i = 0; i < num; i++ ) {
		b[idx] += a[i];//sqrtf((float)a[i]);
	}
	//}
}

#define NUM_CLONES 8
#define NUM_GATES 3
__host__ void decode(int cloneidCode, int *cloneid) 
{
	int factor = 1;
	for (int i = 0; i < NUM_GATES; i++)
		factor *= 2;//numChoicesPerGate[i];
	for (int i = 0; i < NUM_GATES; i++) {
		factor /= 2;//numChoicesPerGate[NUM_GATES-i-1];
		int r = cloneidCode / factor;
		cloneidCode = cloneidCode - r * factor;
		cloneid[NUM_GATES-i-1] = r;
	}
}
int encode(int *cloneidArrayVal)
{
	int ret = 0;
	int factor = 1;
	for (int i = 0; i < NUM_GATES; i++) {
		ret += factor * cloneidArrayVal[i];
		factor *= 2;//numChoicesPerGate[i];
	}
	return ret;
}
void fatherCloneidArray(const int *childVal, int *fatherVal) {
	memcpy(fatherVal, childVal, NUM_GATES * sizeof(int));
	for (int i = NUM_GATES-1; i >= 0; i--) {
		if (fatherVal[i] != 0) {
			fatherVal[i] = 0;
			return;
		}
	}
}

int main3() {
	int cloneid[NUM_CLONES][NUM_GATES];
	cudaStream_t *streams = new cudaStream_t[NUM_CLONES];
	bool finished[NUM_CLONES];
	bool launched[NUM_CLONES];

	for (int i = 0; i < NUM_CLONES; i++) {
		decode(i, cloneid[i]);
		cudaStreamCreate(&streams[i]);
		finished[i] = false;
		launched[i] = false;
	}

	for (int i = 0; i < NUM_CLONES; i++) {
		printf("%d ", cudaStreamQuery(streams[i]));
	}

	int NUM_DATA = 1024 * 32;
	int *a = new int[NUM_DATA];
	int *b = new int[NUM_DATA];
	for (int i = 0; i < NUM_DATA; i++) {
		a[i] = i;
		b[i] = i;
	}
	int *a_dev, *b_dev;
	cudaMalloc((void**)&a_dev, sizeof(int) * NUM_DATA);
	cudaMalloc((void**)&b_dev, sizeof(int) * NUM_DATA);
	cudaMemcpy(a_dev, a, sizeof(int) * NUM_DATA, cudaMemcpyHostToDevice);

	int bSize = 32;
	int gSize = NUM_DATA/bSize;
	int fatherVal[NUM_GATES];

	cudaDeviceSynchronize();
	bool allDone = true;
	
	do {
		for (int i = 0; i < NUM_CLONES; i++) {
			if (finished[i] == false) {
				fatherCloneidArray(cloneid[i], fatherVal);
				int fatherId = encode(fatherVal);
				bool execute;
				if (i == 0)	execute = launched[i] == false;
					else	execute = finished[fatherId] == true && launched[i] == false;

				if (execute) {
					printf("%d ", i);
					addKernel<<<gSize, bSize, 0, streams[i]>>>
						(NULL, a_dev, b_dev);
					launched[i] = true;
					finished[i] = false;
				}
				if (launched[i] == true && cudaStreamQuery(streams[i]) == cudaSuccess)
					finished[i] = true;
			}
			
		}
		allDone = true;
		for (int i = 0; i < NUM_CLONES; i++) {
			printf("%d", launched[i]);
		}
		printf(" ");

		for (int i = 0; i < NUM_CLONES; i++) {
			printf("%d", finished[i]);
			allDone = allDone && finished[i];
		}
		printf("\n");
	} while(!allDone);

	return 0;
}

int main() 
{
	int N = 32;
	int res_host = -1;
	int *a_host = new int[N];
	int val = 1;
	for (int i = 0; i < N; i++) {
		a_host[i] = ~val;
		val = val << 3;
	}
	int *res_dev, *a_dev;
	cudaMalloc((void**)&res_dev, sizeof(int));
	cudaMalloc((void**)&a_dev, N * sizeof(int));
	cudaMemcpy(res_dev, &res_host, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(a_dev, a_host, sizeof(int) * N, cudaMemcpyHostToDevice);
	testAtomicAndKernel<<<1, N>>>(res_dev, a_dev);
	cudaMemcpy(&res_host, res_dev, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%llX", res_host);
	return 0;
}

int main1()
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
