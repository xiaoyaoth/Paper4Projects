#ifndef LIB_HEADER_H
#define LIB_HEADER_H

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "float.h"
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include "inc/helper_math.h"

#define FLOATn float2
#define INTn int2
//#define GWORLD_3D

struct agentColor
{
	uchar4 blue;
	uchar4 green;
	uchar4 red;
	uchar4 yellow;
	uchar4 white;
	uchar4 black;
};

struct modelConstants
{
	unsigned int AGENT_NO;		//copied from host
	int CELL_NO;		//copied from host
	float WIDTH;		//copied from host
	float HEIGHT;	//copied from host
	float DEPTH;		//copied from host
	int CNO_PER_DIM;	//(int)pow((float)2, DISCRETI)
	float CLEN_X;		//WIDTH/(float)CNO_PER_DIM;
	float CLEN_Y;		//HEIGHT/(float)CNO_PER_DIM;
	float CLEN_Z;		//DEPTH/(float)CNO_PER_DIM;
	int RANDOM_SEED;
	int MAX_AGENT_NO;
};

__constant__ agentColor colorConfigs;
__constant__ modelConstants modelDevParams;
modelConstants modelHostParams;

size_t HEAP_SIZE;	//read from config
size_t STACK_SIZE;	//read from config
int STEPS;			//read from config
bool VISUALIZE;		//read from config
int BLOCK_SIZE;		//read from config
#define GRID_SIZE(n) (n%BLOCK_SIZE==0 ? n/BLOCK_SIZE : n/BLOCK_SIZE + 1)

namespace SCHEDULE_CONSTANT{
	static const float EPOCH = 0.0;
	static const float BEFORE_SIMULATION = EPOCH - 1.0;
	static const float AFTER_SIMULTION = FLT_MAX;
	static const float EPSILON = 1.0;
};

#define checkCudaErrors(err)	__checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors( cudaError err, const char *file, const int line )
{
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString( err ) );
		exit(-1);
	}
}
// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)	__getLastCudaError (msg, __FILE__, __LINE__)
inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
		system("PAUSE");
		exit(-1);
	}
}

#endif