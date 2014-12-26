#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>

#define NUM_CLONES 1
#define NUM_GATES 3
__global__ void heavyLoadKernel(int *a, int *b, int num) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//if (idx == 0) {
	for (int i = 0; i < num; i++ ) {
		b[idx] += a[i];//sqrtf((float)a[i]);
	}
	if (idx == 0)
		printf("haha");
	//}
}

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

int main() {
	int cloneid[NUM_CLONES][NUM_GATES];
	cudaStream_t *streams = new cudaStream_t[NUM_CLONES];
	bool finished[NUM_CLONES];

	for (int i = 0; i < NUM_CLONES; i++) {
		decode(i, cloneid[i]);
		//cudaStreamCreate(&streams[i]);
		finished[i] = false;
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

	int bSize = 256;
	int gSize = NUM_DATA/bSize;
	int fatherVal[NUM_GATES];
	int allDone = true;

	cudaDeviceSynchronize();
	
	for (int i = 0; i < NUM_CLONES; i++) {
		printf("%d ", i);
		fatherCloneidArray(cloneid[i], fatherVal);
		int fatherId = encode(fatherVal);
		//if (cudaStreamQuery(streams[fatherId]) == cudaSuccess) 
			heavyLoadKernel<<<gSize, bSize>>>(a_dev, b_dev, NUM_DATA);
		//else
			//i--;
	}
	
	return 0;
}