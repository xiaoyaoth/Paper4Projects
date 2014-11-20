#ifndef GSIMCORE_H
#define GSIMCORE_H
#include "gsimlib_header.cuh"

#define GSIM_DEBUG

//class delaration
class GAgent;
class GWorld;
class GScheduler;
class GModel;
class GRandom;
template<class Agent, class AgentData> class AgentPool;

typedef struct iter_info_per_thread
{
	INTn cellCur;
	INTn cellHead;
	INTn cellTail;

	int boarder;
	int ptrInWorld;
	int ptrInSmem;

	bool infected; // if my neighbor has different clone id then I am infected. -> lead to different infect scheme (conservative and optimistic)
					// conservative : as long as different clone is detected, new copy is cloned
					// optimistic	: clone until different clone takes effects.
} iterInfo;
extern __shared__ int gsimSmem[];
int *hash;
FLOATn *pos;
int stepCountHost = 0;
__constant__ int stepCount;


namespace util{
	void genNeighbor(GWorld *world, GWorld *world_h, int numAgent, cudaStream_t poolStream);

	template<class Type> void hostAllocCopyToDevice(Type *hostPtr, Type **devicePtr);
	__device__ int zcode(int x, int y);
	__device__ int zcode(int x, int y, int z);
};
namespace agentPoolUtil{
	//poolUtil implementation
	/*kernel func: set pool->idxArray[idx] = idx*/
	__global__ void initPoolIdxArray(
		int *idxArray, 
		int *delMark,
		int numElemMax) 
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < numElemMax) {
			idxArray[idx] = idx;
			delMark[idx] = true;
		}
	}
	/*kernel func: numElem = numElem + incCount - decCount; no need to be called*/
	template<class Agent, class AgentData> 
	__global__ void genHash(AgentPool<Agent, AgentData> *pDev, int *hash) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < pDev->numElemMax) {
			bool del = pDev->delMark[idx];
			Agent *o = pDev->agentPtrArray[idx];
			if (del == false) {
				hash[idx] = o->locHash();
			}
		}
	}

	template<class Agent>
	__global__ void stepKernel(int numElem, Agent **agentPtrArray, GModel *model)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x; 
		if (idx < numElem) {
			Agent *ag = agentPtrArray[idx];
			ag->ptrInPool = idx;
			ag->step(model);
		}
	}
	template<class Agent>
	__global__ void swapKernel(int numElem, Agent **agentPtrArray)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < numElem) {
			Agent *ag = agentPtrArray[idx];
			ag->swapDataAndCopy(); // swapDataAndCopy should be inside a different kernel?
		}
	}
};

__global__ void generateHash(int *hash, GAgent **agentPtrArray, FLOATn *pos, int numAgent);
__global__ void generateCellIdx(int *hash, GWorld *c2d, int numAgent);
void sortHash(int *hash, GAgent **ptrArray, int numAgent);

typedef struct GAgentData{
	FLOATn loc;
	GAgent *agentPtr;
} GAgentData_t;

class GAgent 
{
public:
	GAgentData_t *data;
	GAgentData_t *dataCopy;
	
	int ptrInPool;
	uchar4 color;

public:
	__device__ void swapDataAndCopy(){
		GAgentData_t *temp = this->data;
		this->data = this->dataCopy;
		this->dataCopy = temp;
	}
	__device__ int locHash() {
		FLOATn myLoc = this->data->loc;
		int xhash = (int)(myLoc.x/modelDevParams.CLEN_X);
		int yhash = (int)(myLoc.y/modelDevParams.CLEN_Y);
		return util::zcode(xhash, yhash);
	}
};

class GWorld
{
public:
	float width;
	float height;
#ifdef GWORLD_3D
	float depth;
#endif
public:
	GAgent **allAgents;
	int *cellIdxStart;
	int *cellIdxEnd;
public:
	__host__ GWorld(){
		this->width = modelHostParams.WIDTH;
		this->height = modelHostParams.HEIGHT;
		size_t sizeCellArray = modelHostParams.CELL_NO*sizeof(int);

		cudaMalloc((void**)&this->allAgents, modelHostParams.MAX_AGENT_NO*sizeof(GAgent*));
		cudaMalloc((void**)&cellIdxStart, sizeCellArray);
		cudaMalloc((void**)&cellIdxEnd, sizeCellArray);
	}
	//agent list manipulation
	__device__ GAgent* obtainAgent(int idx) const;
	//distance utility
	__device__ float stx(float x, float width) const;
	__device__ float sty(float y, float height) const;
	__device__ float tdx(float ax, float bx) const;
	__device__ float tdy(float ay, float by) const;
	//Neighbors related
	__device__ void neighborQueryInit(const FLOATn &loc, float range, iterInfo &info) const;
	__device__ void neighborQueryReset(iterInfo &info) const;
	template<class dataUnion> __device__ dataUnion* nextAgentDataFromSharedMem(iterInfo &info) const;
	__device__ GAgentData_t *nextAgentData(iterInfo &info) const;
	template<class dataUnion> __device__ GAgent *nextAgent(iterInfo &info) const;
#ifdef GWORLD_3D
	__host__ GWorld(float w, float h, float d)
	{
		this->width = w;
		this->height = h;
		this->depth = d;
		size_t sizeAgArray = MAX_AGENT_NO*sizeof(int);
		size_t sizeCellArray = CELL_NO*sizeof(int);

		cudaMalloc((void**)&this->allAgents, MAX_AGENT_NO*sizeof(GAgent*));
		cudaMalloc((void**)&neighborIdx, sizeAgArray);
		cudaMalloc((void**)&cellIdxStart, sizeCellArray);
		cudaMalloc((void**)&cellIdxEnd, sizeCellArray);
	}
	//distance utility
	__device__ float stz(float z) const;
	__device__ float tdz(float az, float bz) const;
#endif
private:
	__device__ bool iterContinue(iterInfo &info) const;
	template<class dataUnion> __device__ void setSMem(iterInfo &info) const;
	__device__ void calcPtrAndBoarder(iterInfo &info) const;
};

class GScheduler
{
public:
	GAgent **agentPtrArray;
	__host__ GScheduler(){
		cudaMalloc((void**)&this->agentPtrArray, modelHostParams.MAX_AGENT_NO * sizeof(GAgent*));
	}
};

class GModel
{
public:
	GWorld *world, *worldHost;
	GScheduler *scheduler, *schedulerHost;
	GModel *model;
public:
	__host__ GModel() 
	{
		schedulerHost = new GScheduler();
		util::hostAllocCopyToDevice<GScheduler>(schedulerHost, &scheduler);

		//worldHost = new GWorld(modelHostParams.WIDTH, modelHostParams.HEIGHT);
		//util::hostAllocCopyToDevice< GWorld > (worldHost, &world);
	}
	__host__ virtual void start() = 0;
	__host__ virtual void preStep() = 0;
	__host__ virtual void step() = 0;
	__host__ virtual void stop() = 0;
};

__global__ void initRandom(GRandom *random, int nElemMax);
class GRandom {
	curandState_t *rState;
public:
	__host__ GRandom(int nElemMax) {
		printf("GRandom size: %d\n", sizeof(curandState_t));
		cudaMalloc((void**)&rState, nElemMax * sizeof(curandState_t));
		GRandom *devSelf;
		util::hostAllocCopyToDevice<GRandom>(this, &devSelf);
		int gSize = GRID_SIZE(nElemMax);
		initRandom<<<gSize, BLOCK_SIZE>>>(devSelf, nElemMax);
		cudaFree(devSelf);
	}
	__host__ ~GRandom()
	{
		cudaFree(rState);
	}
	__device__ void init(){
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		curand_init(modelDevParams.RANDOM_SEED, idx, 0, &this->rState[idx]);
	}
	__device__ float uniform(){
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		float res = curand_uniform(&rState[idx]);
		return res;
	}
	__device__ float gaussian(){
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		return curand_normal(&this->rState[idx]);
	}
	friend __global__ void initRandom(GRandom *random, int nElemMax);
};
__global__ void initRandom(GRandom *random, int nElemMax)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < nElemMax)	{
		curand_init(modelDevParams.RANDOM_SEED, idx, 0, &random->rState[idx]);
	}
}

template<class Agent, class AgentData> class AgentPool
{
	cudaStream_t *poolStream;
public:
	size_t shareDataSize;
	/* objects to be deleted will be marked as delete */
	int *delMark;
	/* pointer array, elements are pointers points to elements in data array
	Since pointers are light weighted, it will be manipulated in sorting, etc. */
	int *dataIdxArray;
	/* check if elements are inserted or deleted from pool*/
	bool modified;
	unsigned int numElem;
	unsigned int numElemMax;
	unsigned int incCount;
	unsigned int decCount;
	/* keeping the actual Agents */
	Agent **agentPtrArray;
	/* Agent array*/
	Agent *agentArray;
	/* ptrArray to agent data*/
	AgentData *dataArray;
	/* ptrArray to agent data copy*/
	AgentData *dataCopyArray;

	//Pool implementation
	__device__ int agentSlot()
	{
		return atomicInc(&incCount, numElemMax-numElem) + numElem;
	}
	__device__ int add(Agent *o, int agentSlot)
	{
		this->modified = true;
		this->delMark[agentSlot] = false;
		this->agentPtrArray[agentSlot] = o;
		return this->dataIdxArray[agentSlot];
	}
	__device__ int dataSlot(int agentSlot)
	{
		return this->dataIdxArray[agentSlot];
	}
	__device__ Agent* agentInSlot(int dataSlot) {
		return &this->agentArray[dataSlot];
	}
	__device__ AgentData* dataInSlot(int dataSlot){
		return &this->dataArray[dataSlot];
	}
	__device__ AgentData* dataCopyInSlot(int dataSlot){
		return &this->dataCopyArray[dataSlot];
	}
	__device__ void remove(int agentSlot)
	{
		bool del = atomicCAS(&this->delMark[agentSlot], false, true);
		if (del == false) 
			atomicInc(&decCount, numElem);

		this->modified = true;
	}
	__host__ void alloc(int nElem, int nElemMax){
		printf("AgentPool::alloc: Agent size: %d, AgentData size: %d\n", sizeof(Agent), sizeof(AgentData));
		this->numElem = nElem;
		this->numElemMax = nElemMax;
		this->incCount = 0;
		this->decCount = 0;
		this->modified = false;
		cudaMalloc((void**)&this->delMark, nElemMax * sizeof(int));
		cudaMalloc((void**)&this->agentArray, nElemMax * sizeof(Agent));
		cudaMalloc((void**)&this->agentPtrArray, nElemMax * sizeof(Agent*));
		cudaMalloc((void**)&this->dataArray, nElemMax * sizeof(AgentData));
		cudaMalloc((void**)&this->dataCopyArray, nElemMax * sizeof(AgentData));
		cudaMalloc((void**)&this->dataIdxArray, nElemMax * sizeof(int));
		cudaMemset(this->agentPtrArray, 0x00, nElemMax * sizeof(Agent*));
		cudaMemset(this->dataIdxArray, 0x00, nElemMax * sizeof(int));
		cudaMemset(this->delMark, 1, nElemMax * sizeof(int));
	}
	__host__ bool cleanup(AgentPool<Agent, AgentData> *pDev)
	{
		typedef thrust::device_ptr<void*> tdp_voidStar;
		typedef thrust::device_ptr<int> tdp_int;
		cudaMemcpy(this, pDev, sizeof(AgentPool<Agent, AgentData>), cudaMemcpyDeviceToHost);

		this->incCount = 0;
		this->decCount = 0;
		bool poolModifiedLocal = this->modified;
		this->modified = false;
		
		//if (poolModifiedLocal && this->numElem > 0) {
		int gSize = GRID_SIZE(this->numElemMax);
		//agentPoolUtil::cleanupDevice<<<gSize, BLOCK_SIZE>>>(pDev);
		agentPoolUtil::genHash<<<gSize, BLOCK_SIZE>>>(pDev, hash);

		if (poolModifiedLocal == false)
			return false;

		int *dataIdxArrayLocal = this->dataIdxArray;
		int *delMarkLocal = this->delMark;
		void **agentPtrArrayLocal = (void**)this->agentPtrArray;

		tdp_int thrustDelMark = thrust::device_pointer_cast(delMarkLocal);
		tdp_voidStar thrustAgentPtrArray = thrust::device_pointer_cast(agentPtrArrayLocal);
		tdp_int thrustDataIdxArray = thrust::device_pointer_cast(dataIdxArrayLocal);
		tdp_int thrustHash = thrust::device_pointer_cast(hash);

		thrust::tuple<tdp_voidStar, tdp_int> val = thrust::make_tuple(thrustAgentPtrArray, thrustDataIdxArray);
		thrust::tuple<tdp_int, tdp_int> key = thrust::make_tuple(thrustDelMark, thrustHash);
		thrust::zip_iterator< thrust::tuple<tdp_voidStar, tdp_int> > valFirst = thrust::make_zip_iterator(val);
		thrust::zip_iterator< thrust::tuple<tdp_int, tdp_int> > keyFirst = thrust::make_zip_iterator(key);
		thrust::sort_by_key(keyFirst, keyFirst + this->numElemMax, valFirst);

		this->numElem = this->numElemMax - thrust::reduce(thrustDelMark, thrustDelMark + this->numElemMax);
		cudaMemcpy(pDev, this, sizeof(AgentPool<Agent, AgentData>), cudaMemcpyHostToDevice);
		//}

		return poolModifiedLocal; 
	}
	__host__ void registerPool(GWorld *worldHost, GScheduler *schedulerHost, AgentPool<Agent, AgentData> *pDev)
	{
		// clean up the slots
		// sort the agents based on the agent hash
		cleanup(pDev);

		// copy the sorted agents to world
		if (numElem > 0) {
			Agent **worldPtrArray = (Agent**)worldHost->allAgents;
			cudaMemcpy(worldPtrArray + modelHostParams.AGENT_NO, agentPtrArray, numElem * sizeof(Agent*), cudaMemcpyDeviceToDevice);
		}
		getLastCudaError("registerPool");
		modelHostParams.AGENT_NO += numElem;
	}
	__host__ void swapPool() {
		if (numElem > 0) {
			int gSize = GRID_SIZE(numElem);
			agentPoolUtil::swapKernel<<<gSize, BLOCK_SIZE>>>(numElem, agentPtrArray);
		}
	}
	__host__ AgentPool(int nElem, int nElemMax, size_t sizeOfSharedData, cudaStream_t *pStream){
		if (nElemMax < nElem)
			nElemMax = nElem;
		this->shareDataSize = sizeOfSharedData + sizeof(int);
		this->shareDataSize *= BLOCK_SIZE;
		//cudaStreamCreate(&poolStream);
		this->poolStream = pStream;
		this->alloc(nElem, nElemMax);
		int gSize = GRID_SIZE(nElemMax);
		agentPoolUtil::initPoolIdxArray<<<gSize, BLOCK_SIZE>>>(dataIdxArray, delMark, nElemMax);
	}
	__host__ int stepPoolAgent(GModel *model)
	{
		if (numElem <= 0)
			return 0;
		int gSize = GRID_SIZE(numElem);
		agentPoolUtil::stepKernel<<<gSize, BLOCK_SIZE, this->shareDataSize, *poolStream>>>(numElem, this->agentPtrArray, model);
		return numElem;
	}
};

//GWorld
__device__ GAgent* GWorld::obtainAgent(int idx) const {
	GAgent *ag = NULL;
	if (idx < modelDevParams.AGENT_NO && idx >= 0){
		ag = this->allAgents[idx];
	} 
	return ag;
}
__device__ float GWorld::stx(float x, float width) const{
	if (x >= 0) {
		if (x >= width)
			x = x - width;
	} else
		x = x + width;
	return x;
}
__device__ float GWorld::sty(float y, float height) const {
	if (y >= 0) {
		if (y >= height)
			y = y - height;
	} else
		y = y + height;
	return y;

}
__device__ float GWorld::tdx(float ax, float bx) const {
	float width = this->width;
	if(fabs(ax - bx) <= width / 2)
		return ax - bx;

	float dx = stx(ax, width) - stx(bx, width);
	if (dx * 2 > width)
		return dx - width;
	if (dx * 2 < -width)
		return dx + width;
	return 0;
}
__device__ float GWorld::tdy(float ay, float by) const {
	float height = this->height;
	if(fabs(ay - by) <= height / 2)
		return ay - by;

	float dy = stx(ay, height) - stx(by, height);
	if (dy * 2 > height)
		return dy - height;
	if (dy * 2 < - height)
		return dy + height;
	return 0;
}
__device__ int sharedMin(volatile int* data, int tid, int idx, float loc, float range, float discr)
{
	int index = (int)((loc-range)/discr);
	if (index < 0) index = 0;
	int lane = tid & 31;
	int wid = tid >> 5;
	//__syncthreads();
	data[tid] = index;
	if (lane >= 1) if(data[tid] < data[tid-1]) data[tid-1] = data[tid];
	if (lane >= 2) if(data[tid] < data[tid-2]) data[tid-2] = data[tid];
	if (lane >= 4) if(data[tid] < data[tid-4]) data[tid-4] = data[tid];
	if (lane >= 8) if(data[tid] < data[tid-8]) data[tid-8] = data[tid];
	if (lane >= 16) if(data[tid] < data[tid-16]) data[tid-16] = data[tid];
	return data[wid * warpSize];
}
__device__ int sharedMax(volatile int* data, int tid, int idx, float loc, float range, float discr)
{
	int index = (int)((loc+range)/discr);
	if (index >= modelDevParams.CNO_PER_DIM) index = modelDevParams.CNO_PER_DIM - 1;
	int lane = tid & 31;
	int wid = tid >> 5;
	//__syncthreads();
	data[tid] = index;
	if (lane >= 1) if(data[tid] > data[tid-1]) data[tid-1] = data[tid];
	if (lane >= 2) if(data[tid] > data[tid-2]) data[tid-2] = data[tid];
	if (lane >= 4) if(data[tid] > data[tid-4]) data[tid-4] = data[tid];
	if (lane >= 8) if(data[tid] > data[tid-8]) data[tid-8] = data[tid];
	if (lane >= 16) if(data[tid] > data[tid-16]) data[tid-16] = data[tid];
	return data[wid * warpSize];
}
__device__ void GWorld::neighborQueryInit(const FLOATn &agLoc, float range, iterInfo &info) const {
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	info.ptrInWorld = -1;
	info.boarder = -1;
	info.ptrInSmem = 0;

	info.cellHead.x = sharedMin(gsimSmem, tid, idx, agLoc.x, range, modelDevParams.CLEN_X);
	info.cellTail.x = sharedMax(gsimSmem, tid, idx, agLoc.x, range, modelDevParams.CLEN_X);
	info.cellHead.y = sharedMin(gsimSmem, tid, idx, agLoc.y, range, modelDevParams.CLEN_Y);
	info.cellTail.y = sharedMax(gsimSmem, tid, idx, agLoc.y, range, modelDevParams.CLEN_Y);
#ifdef GWORLD_3D
	info.cellHead.z = sharedMin(gsimSmem, tid, idx, agLoc.z, range, modelDevParams.CLEN_Z);
	info.cellTail.z = sharedMin(gsimSmem, tid, idx, agLoc.z, range, modelDevParams.CLEN_Z);
#endif

	info.cellCur.x = info.cellHead.x;
	info.cellCur.y = info.cellHead.y;
#ifdef GWORLD_3D
	info.cellCur.z = info.cellHead.z;
#endif

#ifdef GSIM_DEBUG
	if (info.cellCur.x < 0) printf("neighborQueryInit: tid:%d cellCur.x: %d pos: %f\n", tid, info.cellCur.x, agLoc.x);
	if (info.cellCur.y < 0) printf("neighborQueryInit: tid:%d cellCur.y: %d pos: %f\n", tid, info.cellCur.y, agLoc.y);
#ifdef GWORLD_3D
	if (info.cellCur.z < 0) printf("neighborQueryInit: tid:%d cellCur.z: %d pos: %f\n", tid, info.cellCur.z, agLoc.z);
#endif
#endif
	this->calcPtrAndBoarder(info);
}
__device__ void GWorld::neighborQueryReset(iterInfo &info) const{
	info.ptrInWorld = -1;
	info.boarder = -1;
	info.ptrInSmem = 0;
	info.cellCur = info.cellHead;
	this->calcPtrAndBoarder(info);
}
__device__ void GWorld::calcPtrAndBoarder(iterInfo &info) const {
#ifdef GWORLD_3D
	int hash = util::zcode(info.cellCur.x, info.cellCur.y, info.cellCur.z);
#else
	int hash = util::zcode(info.cellCur.x, info.cellCur.y);
#endif
	if(hash < modelDevParams.CELL_NO && hash>=0)
	{
		info.ptrInWorld = this->cellIdxStart[hash];
		info.boarder = this->cellIdxEnd[hash];
	}
}
template<class dataUnion> __device__ dataUnion *GWorld::nextAgentDataFromSharedMem(iterInfo &info) const {
	dataUnion *unionArray = (dataUnion*)&gsimSmem[blockDim.x];

	const int tid = threadIdx.x;
	const int lane = tid & 31;

	if (!iterContinue(info))
		return NULL;

	setSMem<dataUnion>(info);

	dataUnion *elem = &unionArray[tid - lane + info.ptrInSmem];
	info.ptrInSmem++;
	info.ptrInWorld++;

	return elem;
}
__device__ GAgentData_t *GWorld::nextAgentData(iterInfo &info) const 
{
	if (!iterContinue(info))
		return NULL;

	GAgent *ag = this->obtainAgent(info.ptrInWorld);
	info.ptrInWorld++;
	return ag->data;
}
template<class dataUnion> __device__ GAgent *GWorld::nextAgent(iterInfo &info) const
{
	if (!iterContinue(info))
		return NULL;

	setSMem<dataUnion>(info);

	GAgent *ag = this->obtainAgent(info.ptrInWorld);
	info.ptrInWorld++;
	return ag;
}
__device__ bool GWorld::iterContinue(iterInfo &info) const
{
	while (info.ptrInWorld>=info.boarder) {
		info.ptrInSmem = 0;
		info.cellCur.x++;
		if(info.cellCur.x>info.cellTail.x){
			info.cellCur.x = info.cellHead.x;
			info.cellCur.y++;
			if(info.cellCur.y>info.cellTail.y) {
#ifdef GWORLD_3D
				info.cellCur.y = info.cellHead.y;
				info.cellCur.z++;
				if (info.cellCur.z > info.cellTail.z)
					return false;
#else
				return false;
#endif
			}
		}
		this->calcPtrAndBoarder(info);
	}
	return true;
}
template<class dataUnion> __device__ void GWorld::setSMem(iterInfo &info) const
{
	dataUnion *unionArray = (dataUnion*)&gsimSmem[blockDim.x];
	//dataUnion *unionArray = (dataUnion*)&gsimSmem[0];
	int tid = threadIdx.x;
	int lane = tid & 31;
	int wid = tid >> 5;

	if (info.ptrInSmem == 32)
		info.ptrInSmem = 0;

#define NUM_AGENT_IN_SMEM 32
	if (info.ptrInSmem == 0 && lane == 0) {
		for (int i = 0; i < NUM_AGENT_IN_SMEM; i++) {
			dataUnion &elem = unionArray[wid * 32 + i];
			int idxSmem = info.ptrInWorld + i;
			if (idxSmem < info.boarder && idxSmem >= 0) {
				GAgent *ag = this->obtainAgent(idxSmem);
				elem.putDataInSmem(ag);
			}
		}
	}

}
#ifdef GWORLD_3D
__device__ float GWorld::stz(const float z) const{
	float res = z;
	if (z >= 0) {
		if (z >= this->depth)
			res = z - this->depth;
	} else
		res = z + this->depth;
	if (res == this->depth)
		res = 0;
	return res;
}
__device__ float GWorld::tdz(float az, float bz) const {
	float dz = abs(az-bz);
	if (dz < this->depth / 2)
		return dz;
	else
		return this->depth - dz;
}
#endif

void errorHandler(GWorld *world_h);
//namespace continuous2D Utility
__global__ void generateHash(int *hash, GAgent **agentPtrArray, FLOATn *pos, int numAgent)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAgent) {
		GAgent *ag = agentPtrArray[idx];
		hash[idx] = ag->locHash();
#ifdef _DEBUG
		FLOATn myLoc = ag->data->loc;
		pos[idx] = myLoc;
#endif
	}
}
__global__ void generateCellIdx(int *hash, GWorld *c2d, int numAgent)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAgent && idx > 0) {
		int hash_idx = hash[idx];
		int hash_previdx = hash[idx-1];
		if (hash_idx != hash_previdx) {
			c2d->cellIdxStart[hash_idx] = idx;
			c2d->cellIdxEnd[hash_previdx] = idx;
		}
	}
	if (idx == 0) {
		c2d->cellIdxStart[hash[0]] = idx;
		c2d->cellIdxEnd[hash[numAgent-1]] = numAgent;
	}
}
void sortHash(int *hash, GAgent **ptrArray, int numAgent)
{
	thrust::device_ptr<void*> id_ptr((void**)ptrArray);
	thrust::device_ptr<int> hash_ptr(hash);
	typedef thrust::device_vector<void*>::iterator ValIter;
	typedef thrust::device_vector<int>::iterator KeyIter;
	KeyIter key_begin(hash_ptr);
	KeyIter key_end(hash_ptr + numAgent);
	ValIter val_begin(id_ptr);
	thrust::sort_by_key(key_begin, key_end, val_begin);
}
void util::genNeighbor(GWorld *world, GWorld *world_h, int numAgent, cudaStream_t poolStream)
{
	if (world != NULL) {
		int bSize = BLOCK_SIZE;
		int gSize = GRID_SIZE(numAgent);

		cudaMemsetAsync(hash, 0xff, numAgent * sizeof(int), poolStream);
		cudaMemsetAsync(world_h->cellIdxStart, 0xff, modelHostParams.CELL_NO*sizeof(int), poolStream);
		cudaMemsetAsync(world_h->cellIdxEnd, 0xff, modelHostParams.CELL_NO*sizeof(int), poolStream);

		generateHash<<<gSize, bSize, 0, poolStream>>>(hash, world_h->allAgents, pos, numAgent);
		if(cudaSuccess != cudaGetLastError()) { errorHandler(world_h); }
		sortHash(hash, world_h->allAgents, numAgent);
		if(cudaSuccess != cudaGetLastError()) { errorHandler(world_h); }
		generateCellIdx<<<gSize, bSize, 0, poolStream>>>(hash, world, numAgent);
		if(cudaSuccess != cudaGetLastError()) { errorHandler(world_h); }
	}
}
template<class Type> void util::hostAllocCopyToDevice(Type *hostPtr, Type **devPtr)//device ptrInWorld must be double star
{
	size_t size = sizeof(Type);
	cudaMalloc(devPtr, size);
	cudaMemcpy(*devPtr, hostPtr, size, cudaMemcpyHostToDevice);
	getLastCudaError("copyHostToDevice");
}
__device__ int util::zcode(int x, int y)
{
	x &= 0x0000ffff;					// x = ---- ---- ---- ---- fedc ba98 7654 3210
	y &= 0x0000ffff;					// x = ---- ---- ---- ---- fedc ba98 7654 3210
	x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	y = (y ^ (y << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	y = (y ^ (y << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	y = (y ^ (y << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	y = (y ^ (y << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	return x | (y << 1);
}
__device__ int util::zcode(int x, int y, int z)
{
	x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0

	y &= 0x000003ff;                  // y = ---- ---- ---- ---- ---- --98 7654 3210
	y = (y ^ (y << 16)) & 0xff0000ff; // y = ---- --98 ---- ---- ---- ---- 7654 3210
	y = (y ^ (y <<  8)) & 0x0300f00f; // y = ---- --98 ---- ---- 7654 ---- ---- 3210
	y = (y ^ (y <<  4)) & 0x030c30c3; // y = ---- --98 ---- 76-- --54 ---- 32-- --10
	y = (y ^ (y <<  2)) & 0x09249249; // y = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0

	z &= 0x000003ff;                  // z = ---- ---- ---- ---- ---- --98 7654 3210
	z = (z ^ (z << 16)) & 0xff0000ff; // z = ---- --98 ---- ---- ---- ---- 7654 3210
	z = (z ^ (z <<  8)) & 0x0300f00f; // z = ---- --98 ---- ---- 7654 ---- ---- 3210
	z = (z ^ (z <<  4)) & 0x030c30c3; // z = ---- --98 ---- 76-- --54 ---- 32-- --10
	z = (z ^ (z <<  2)) & 0x09249249; // z = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return (z << 2) | (y << 1) | x;
}

//execution logic
void readConfig(char *config_file){
	std::ifstream fin;
	fin.open(config_file);
	std::string rec;
	char *cstr, *p;
	cstr = (char *)malloc(100 * sizeof(char));


	int discr = 0;

	while (!fin.eof()) {
		std::getline(fin, rec);
		std::strcpy(cstr, rec.c_str());
		if(strcmp(cstr,"")==0)
			break;
		p=strtok(cstr, "=");
		if(strcmp(p, "AGENT_NO")==0){
			p=strtok(NULL, "=");
			modelHostParams.AGENT_NO = atoi(p);
		}
		if(strcmp(p, "MAX_AGENT_NO")==0){
			p=strtok(NULL, "=");
			modelHostParams.MAX_AGENT_NO = atoi(p);
		}
		if(strcmp(p, "WIDTH")==0){
			p=strtok(NULL, "=");
			modelHostParams.WIDTH = atoi(p);
		}
		if(strcmp(p, "HEIGHT")==0){
			p=strtok(NULL, "=");
			modelHostParams.HEIGHT = atoi(p);
		}
		if(strcmp(p, "DEPTH")==0){
			p=strtok(NULL, "=");
			modelHostParams.DEPTH = atoi(p);
		}
		if(strcmp(p, "DISCRETI")==0){
			p=strtok(NULL, "=");
			discr = atoi(p);
		}
		if(strcmp(p, "STEPS")==0){
			p=strtok(NULL, "=");
			STEPS = atoi(p);
		}
		if(strcmp(p, "VISUALIZE")==0){
			p=strtok(NULL, "=");
			VISUALIZE = atoi(p);
		}
		if(strcmp(p, "BLOCK_SIZE")==0){
			p=strtok(NULL, "=");
			BLOCK_SIZE = atoi(p);
		}
		if(strcmp(p, "HEAP_SIZE")==0){
			p=strtok(NULL, "=");
			HEAP_SIZE = atoi(p);
		}
		if(strcmp(p, "STACK_SIZE")==0){
			p=strtok(NULL, "=");
			STACK_SIZE = atoi(p);
		}
		if(strcmp(p, "RANDOM_SEED")==0){
			p=strtok(NULL, "=");
			int randomSeed = atoi(p);
		}
	}
	free(cstr);
	fin.close();

	modelHostParams.CNO_PER_DIM = (int)pow((float)2, discr);
	modelHostParams.CELL_NO = modelHostParams.CNO_PER_DIM * modelHostParams.CNO_PER_DIM;
	modelHostParams.CLEN_X = modelHostParams.WIDTH/modelHostParams.CNO_PER_DIM;
	modelHostParams.CLEN_Y = modelHostParams.HEIGHT/modelHostParams.CNO_PER_DIM;

#ifdef GWORLD_3D
	CELL_NO *= CNO_PER_DIM_H;
	float modelHostParams.CLEN_Z = DEPTH_H/(float)CNO_PER_DIM_H;
#endif

	printf("model params:\n");
	printf("\tmodelHostParams.AGENT_NO:%d\n", modelHostParams.AGENT_NO);
	printf("\tmodelHostParams.CELL_NO:%d\n", modelHostParams.CELL_NO);
	printf("\tmodelHostParams.CLEN_X:%f\n", modelHostParams.CLEN_X);
	printf("\tmodelHostParams.CLEN_Y:%f\n", modelHostParams.CLEN_Y);
	printf("\tmodelHostParams.CLEN_Z:%f\n", modelHostParams.CLEN_Z);
	printf("\tmodelHostParams.CNO_PER_DIM:%d\n", modelHostParams.CNO_PER_DIM);
	printf("\tmodelHostParams.WIDTH:%f\n", modelHostParams.WIDTH);
	printf("\tmodelHostParams.DEPTH:%f\n", modelHostParams.DEPTH);
	printf("\tmodelHostParams.HEIGHT:%f\n", modelHostParams.HEIGHT);
	printf("\tmodelHostParams.MAX_AGENT_NO:%d\n", modelHostParams.MAX_AGENT_NO);

	cudaMemcpyToSymbol(modelDevParams, &modelHostParams, sizeof(modelConstants));
}
size_t sizeOfSmem = 0;
template<class SharedMemoryData> void init(char *configFile)
{
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	getLastCudaError("setting cache preference");
	readConfig(configFile);

	size_t pVal;
	cudaDeviceGetLimit(&pVal, cudaLimitMallocHeapSize);
	printf("cudaLimitMallocHeapSize: %d\n", pVal);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, HEAP_SIZE);
	cudaDeviceGetLimit(&pVal, cudaLimitMallocHeapSize);
	printf("cudaLimitMallocHeapSize: %d\n", pVal);

	cudaDeviceGetLimit(&pVal, cudaLimitStackSize);
	printf("cudaLimitStackSize: %d\n", pVal);
	cudaDeviceSetLimit(cudaLimitStackSize, STACK_SIZE);
	cudaDeviceGetLimit(&pVal, cudaLimitStackSize);
	printf("cudaLimitStackSize: %d\n", pVal);

	printf("steps: %d\n", STEPS);

	sizeOfSmem = sizeof(SharedMemoryData) + sizeof(int);
	sizeOfSmem *= BLOCK_SIZE;
	printf("SharedMemoryData size: %d, Per block shared memory size: %d\n", sizeof(SharedMemoryData), sizeOfSmem);

	agentColor colorHost;
	colorHost.blue	=	make_uchar4(0, 0, 128, 0);
	colorHost.green =	make_uchar4(0, 128, 0, 128);
	colorHost.red	=	make_uchar4(128, 0, 0, 0);
	colorHost.yellow =	make_uchar4(128, 128, 0, 0);
	colorHost.white	=	make_uchar4(255, 255, 255, 0);
	colorHost.black =	make_uchar4(0, 0, 0, 0);
	cudaMemcpyToSymbol(colorConfigs, &colorHost, sizeof(agentColor));
}
void errorHandler(GWorld *world_h)
{
	FLOATn *pos_h = new FLOATn[modelHostParams.AGENT_NO];
	int *hash_h = new int[modelHostParams.AGENT_NO];
	int *cidx_h = new int[modelHostParams.CELL_NO];
	cudaMemcpy(hash_h, hash, modelHostParams.AGENT_NO * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(cidx_h, world_h->cellIdxStart, modelHostParams.CELL_NO * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(pos_h, pos, sizeof(FLOATn)*modelHostParams.AGENT_NO, cudaMemcpyDeviceToHost);

	std::fstream fout;
	char *outfname = new char[30];
	sprintf(outfname, "out_genNeighbor_neighborIdx.txt");
	fout.open(outfname, std::ios::out);
	for (unsigned int i = 0; i < modelHostParams.AGENT_NO; i++){
		fout 
			<< hash_h[i] << " " 
			<< pos_h[i].x << " " 
			<< pos_h[i].y << " " 
#ifdef GWORLD_3D
			<< pos_h[i].z 
#endif
			<<std::endl;
		fout.flush();
	}
	fout.close();
	sprintf(outfname, "out_genNeighbor_cellIdx.txt");
	fout.open(outfname, std::ios::out);
	for (int i = 0; i < modelHostParams.CELL_NO; i++){
		fout << cidx_h[i] <<std::endl;
		fout.flush();
	}
	fout.close();
	printf("terminated unexpectly\n");
	system("PAUSE");
	exit(-1);
}
void doLoop(GModel *mHost){
	cudaMalloc((void**)&hash, modelHostParams.MAX_AGENT_NO*sizeof(int));
	cudaMalloc((void**)&pos, sizeof(FLOATn)*modelHostParams.MAX_AGENT_NO);

	mHost->start();

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	std::fstream fout;
	char *outfname = new char[30];
	sprintf(outfname, "doLoop count.txt");
	fout.open(outfname, std::ios::out);

	for (; stepCountHost<STEPS; stepCountHost++){
		//if ((i%(STEPS/100))==0) 
		//printf("STEP:%d ", stepCountHost);
		cudaMemcpyToSymbol(stepCount, &stepCountHost, sizeof(int));
		cudaEventRecord(start, 0);

		modelHostParams.AGENT_NO = 0;

		mHost->preStep();
		//util::genNeighbor(mHost->world, mHost->worldHost, modelHostParams.AGENT_NO);

		mHost->step();

		cudaDeviceSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		fout<<time<<std::endl;
	}
	fout.close();
	mHost->stop();
	cudaFree(hash);
	printf("finally total agent is %d\n", modelHostParams.AGENT_NO);
	system("PAUSE");

}
#endif