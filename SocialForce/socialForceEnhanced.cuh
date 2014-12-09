#ifndef SOCIAL_FORCE_CUH
#define SOCIAL_FORCE_CUH

#include "gsimcore.cuh"
#include "gsimvisual.cuh"

#define DIST(ax, ay, bx, by) sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by))

class SocialForceRoomModel;
class SocialForceRoomAgent;
class SocialForceRoomClone;

#define CLONE
//#define _DEBUG
//#define VALIDATE

typedef struct SocialForceRoomAgentData : public GAgentData_t {
	double2 goal;
	double2 velocity;
	double v0;
	double mass;
	__device__ void putDataInSmem(GAgent *ag);
};

struct obstacleLine
{
	double sx;
	double sy;
	double ex;
	double ey;

	__host__ void init(double sxx, double syy, double exx, double eyy)
	{
		sx = sxx;
		sy = syy;
		ex = exx;
		ey = eyy;
	}

	__device__ double pointToLineDist(float2 loc) 
	{
		double a, b;
		return this->pointToLineDist(loc, a, b, 0);
	}

	__device__ double pointToLineDist(float2 loc, double &crx, double &cry, int id) 
	{
		double d = DIST(sx, sy, ex, ey);
		double t0 = ((ex - sx) * (loc.x - sx) + (ey - sy) * (loc.y - sy)) / (d * d);
	
		if(t0 < 0){
			d = sqrt((loc.x - sx) * (loc.x - sx) + (loc.y - sy) * (loc.y - sy));
		}else if(t0 > 1){
			d = sqrt((loc.x - ex) * (loc.x - ex) + (loc.y - ey) * ( loc.y - ey));
		}else{
			d = sqrt(
				(loc.x - (sx + t0 * ( ex  - sx))) * (loc.x - (sx + t0 * ( ex  - sx))) +
				(loc.y - (sy + t0 * ( ey  - sy))) * (loc.y - (sy + t0 * ( ey  - sy)))
				);
		}
		crx = sx + t0 * (ex - sx);
		cry = sy + t0 * (ey - sy);

		//if (stepCount == 0 && id == 263) {
		//	printf("cross: (%f, %f)\n", crx, cry);
		//	printf("t0: %f, d: %f\n", t0, d);
		//}

		return d;
	}

	__device__ int intersection2LineSeg(double p0x, double p0y, double p1x, double p1y, double &ix, double &iy)
	{
		double s1x, s1y, s2x, s2y;
		s1x = p1x - p0x;
		s1y = p1y - p0y;
		s2x = ex - sx;
		s2y = ey - sy;

		double s, t;
		s = (-s1y * (p0x - sx) + s1x * (p0y - sy)) / (-s2x * s1y + s1x * s2y);
		t = ( s2x * (p0y - sy) - s2y * (p0x - sx)) / (-s2x * s1y + s1x * s2y);

		if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
		{
			// Collision detected
			if (ix != NULL)
				ix = p0x + (t * s1x);
			if (iy != NULL)
				iy = p0y + (t * s1y);
			return 1;
		}
		return 0; // No collision
	}
};

#define	tao 0.5
#define	A 2000
#define	B 0.1
#define	k1 (1.2 * 100000)
#define k2 (2.4 * 100000)
#define	maxv 3

#define NUM_WALLS 10
#define NUM_GATES 3

__constant__ double2 gateLocs[NUM_GATES];
__constant__ obstacleLine walls[NUM_WALLS];
//__constant__ double gateSizes[NUM_GATE_0_CHOICES]; 

#define NUM_GATE_0_CHOICES 2
#define NUM_GATE_1_CHOICES 2
#define NUM_GATE_2_CHOICES 2
__constant__ double gate0Sizes[NUM_GATE_0_CHOICES];
__constant__ double gate1Sizes[NUM_GATE_1_CHOICES];
__constant__ double gate2Sizes[NUM_GATE_2_CHOICES];

#define GATE_LINE_NUM 2
#define LEFT_GATE_SIZE 2
#ifdef CLONE
#define RIGHT_GATE_SIZE_A 2
#else
#define RIGHT_GATE_SIZE_A 6
#endif

#define MONITOR_STEP 38 
#define MONITOR_ID 1990 
#define CLONE_COMPARE
#define CLONE_PERCENT 0.5

#define TIMER_START(cloneStream) ;
	//cudaEventRecord(timerStart, cloneStream);

#define TIMER_END(cloneid, stage, cloneStream) ;
	//cudaEventRecord(timerStop, cloneStream); \
	cudaEventSynchronize(timerStop); \
	cudaEventElapsedTime(&time, timerStart, timerStop); \
	printf ("cloneid: %d, stage %s time: %f ms\n", cloneid, stage, time);

#ifdef VALIDATE
__device__ uint throughput;
int throughputHost;
#endif

#ifdef _DEBUG
	SocialForceRoomAgentData *dataHost;
	SocialForceRoomAgentData *dataCopyHost;
	int *dataIdxArrayHost;
#endif


__global__ void addAgentsOnDevice(GRandom *myRandom, int numAgent, SocialForceRoomClone *clone0);
__global__ void replaceOriginalWithClone(SocialForceRoomClone *childClone, int numClonedAgent);
__global__ void cloneKernel(SocialForceRoomClone *fatherClone, 
							SocialForceRoomClone *childClone,
							int numAgentLocal, int cloneid);
__global__ void compareOriginAndClone(SocialForceRoomClone *childClone, int numClonedAgents);


#ifdef CLONE
class SocialForceRoomClone {
private:
	static int cloneCount;
	cudaStream_t cloneStream;
public:
	AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData> *agents, *agentsHost;
	GWorld *clonedWorld, *clonedWorldHost;
	SocialForceRoomAgent **unsortedAgentPtrArray;
	int cloneid;
	uchar4 color;
	
	SocialForceRoomClone *cloneDev;
	int cloneidArray[NUM_GATES];
	int cloneMasks[NUM_GATES];
	int cloneLevel;

	__host__ SocialForceRoomClone(int num, int *cloneidArrayVal) {
		cudaStreamCreate(&cloneStream);

		agentsHost = new AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData>(0, num * 2, sizeof(SocialForceRoomAgentData));
		util::hostAllocCopyToDevice<AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData> >(agentsHost, &agents);
		
		clonedWorldHost = new GWorld();
		util::hostAllocCopyToDevice<GWorld>(clonedWorldHost, &clonedWorld);

		//alloc untouched agent array
		cudaMalloc((void**)&unsortedAgentPtrArray, modelHostParams.MAX_AGENT_NO * sizeof(SocialForceRoomAgent*) );

		cloneid = cloneCount++;
		
		int r = rand();
		memcpy(&color, &r, sizeof(uchar4));

		memcpy(this->cloneidArray, cloneidArrayVal, NUM_GATES * sizeof(int));

		for (cloneLevel = NUM_GATES-1; cloneLevel >= 0; cloneLevel--)
			if (cloneidArray[cloneLevel] != 0)
				break;

		for (int i = 0; i < NUM_GATES; i++) {
			cloneMasks[i] = 1;
			cloneMasks[i] = cloneMasks[i] << cloneidArray[i];
			cloneMasks[i] = cloneMasks[i] >> 1;
		}
		
		util::hostAllocCopyToDevice<SocialForceRoomClone>(this, &this->cloneDev);

	}
	__host__ void stepPhase1(SocialForceRoomClone *fatherClone);
	__host__ void stepPhase2(SocialForceRoomModel *modelHost);
	__host__ void stepPhase3();
	__host__ void stepPhase4();
};
#endif

__global__ void inspect(SocialForceRoomClone *childClone, SocialForceRoomClone *fatherClone) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < childClone->agents->numElem) {
		SocialForceRoomAgent *ag = childClone->agents->agentPtrArray[idx];
		SocialForceRoomAgent *ag2 = fatherClone->agents->agentPtrArray[idx];
	}
}

class SocialForceRoomModel : public GModel {
public:
	GRandom *random, *randomHost;
	cudaEvent_t timerStart, timerStop;

	std::fstream fout;
#ifdef CLONE
	SocialForceRoomClone **clones;
	int numChoicesPerGate[NUM_GATES];
	int numClones;
#endif
	__host__ int encode(int *cloneidArrayVal)
	{
		int ret = 0;
		int factor = 1;
		for (int i = 0; i < NUM_GATES; i++) {
			ret += factor * cloneidArrayVal[i];
			factor *= numChoicesPerGate[i];
		}
		return ret;
	}
	__host__ void decode(int cloneidCode, int *cloneid) 
	{
		int factor = 1;
		for (int i = 0; i < NUM_GATES; i++)
			factor *= numChoicesPerGate[i];
		for (int i = 0; i < NUM_GATES; i++) {
			factor /= numChoicesPerGate[NUM_GATES-i-1];
			int r = cloneidCode / factor;
			cloneidCode = cloneidCode - r * factor;
			cloneid[NUM_GATES-i-1] = r;
		}
	}
	__host__ void fatherCloneidArray(const int *childVal, int *fatherVal) {
		memcpy(fatherVal, childVal, NUM_GATES * sizeof(int));
		for (int i = NUM_GATES-1; i >= 0; i--) {
			if (fatherVal[i] != 0) {
				fatherVal[i] = 0;
				return;
			}
		}
	}
	__host__ SocialForceRoomModel(char **modelArgs) {
		int num = modelHostParams.AGENT_NO;
		char *outfname = new char[30];
#ifdef _DEBUG
#ifdef CLONE
		sprintf(outfname, "agent_%d_clone.txt", num);
#else
		sprintf(outfname, "agent_%d_gate_%d_single.txt", num, RIGHT_GATE_SIZE_A);
#endif
#else
#ifdef CLONE
		sprintf(outfname, "throughput_clone.txt", num);
#else
		sprintf(outfname, "throughput_single_%d.txt", num, RIGHT_GATE_SIZE_A);
#endif
#endif
		fout.open(outfname, std::ios::out);

		double *gateSizesHost = (double*)malloc(sizeof(double) * NUM_GATE_0_CHOICES);
		srand(time(NULL));

		for (int j = 0; j < NUM_GATE_0_CHOICES; j++)
			gateSizesHost[j] = (j + 1);
		cudaMemcpyToSymbol(gate0Sizes, &gateSizesHost[0], NUM_GATE_0_CHOICES * sizeof(double));
		for (int j = 0; j < NUM_GATE_1_CHOICES; j++)
			gateSizesHost[j] = (j + 1);
		cudaMemcpyToSymbol(gate1Sizes, &gateSizesHost[0], NUM_GATE_1_CHOICES * sizeof(double));
		for (int j = 0; j < NUM_GATE_2_CHOICES; j++)
			gateSizesHost[j] = (j + 1);
		cudaMemcpyToSymbol(gate2Sizes, &gateSizesHost[0], NUM_GATE_2_CHOICES * sizeof(double));

		//init obstacles
		obstacleLine wallsHost[NUM_WALLS];
		float wLocal = modelHostParams.WIDTH;
		float hLocal = modelHostParams.HEIGHT;
		float gLocal = LEFT_GATE_SIZE;
		wallsHost[0].init(0.1 * wLocal, 0.09 * hLocal, 0.1 * wLocal, 0.91 * hLocal );
		wallsHost[1].init(0.09 * wLocal, 0.1 * hLocal, 0.91 * wLocal, 0.1 * hLocal);
		wallsHost[2].init(0.9 * wLocal, 0.09 * hLocal, 0.9 * wLocal, 0.3 * hLocal);
		wallsHost[3].init(0.9 * wLocal, 0.3 * hLocal, 0.9 * wLocal, 0.91 * hLocal);
		wallsHost[4].init(0.09 * wLocal, 0.9 * hLocal, 0.91 * wLocal, 0.9 * hLocal);
		wallsHost[5].init(0.5 * wLocal, 0.7 * hLocal, 0.5 * wLocal, 0.91 * hLocal);
		wallsHost[6].init(0.09 * wLocal, 0.5 * hLocal, 0.3 * wLocal, 0.5 * hLocal);
		wallsHost[7].init(0.5 * wLocal, 0.09 * hLocal, 0.5 * wLocal, 0.3 * hLocal);
		wallsHost[8].init(0.5 * wLocal, 0.3 * hLocal, 0.5 * wLocal, 0.7 * hLocal);
		wallsHost[9].init(0.3 * wLocal, 0.5 * hLocal, 0.91 * wLocal, 0.5 * hLocal);
		cudaMemcpyToSymbol(walls, &wallsHost, NUM_WALLS * sizeof(obstacleLine));

		double2 gateLocsHost[NUM_GATES];
		gateLocsHost[0] = make_double2(0.5 * wLocal, 0.7 * hLocal);
		gateLocsHost[1] = make_double2(0.3 * wLocal, 0.5 * hLocal);
		gateLocsHost[2] = make_double2(0.5 * wLocal, 0.3 * hLocal);
		cudaMemcpyToSymbol(gateLocs, &gateLocsHost, NUM_GATES * sizeof(double2));

		numClones = 1;
		for(int i = 0; i < NUM_GATES; i++) {
			numChoicesPerGate[i] = 2;
			numClones *= numChoicesPerGate[i];
		}
		
		int cloneidArrayValLocal[NUM_GATES];
		clones = (SocialForceRoomClone **)malloc(numClones * sizeof(SocialForceRoomClone*));
		for (int i = 0; i < numClones; i++) {
			this->decode(i, cloneidArrayValLocal);
			clones[i] = new SocialForceRoomClone(modelHostParams.AGENT_NO, cloneidArrayValLocal);
		}

		//init utility
		randomHost = new GRandom(modelHostParams.MAX_AGENT_NO);
		util::hostAllocCopyToDevice<GRandom>(randomHost, &random);

		util::hostAllocCopyToDevice<SocialForceRoomModel>(this, (SocialForceRoomModel**)&this->model);
		getLastCudaError("INIT");

	}
	__host__ void start()
	{
		int numAgentLocal = modelHostParams.AGENT_NO;

		//debug info
#if defined(_DEBUG)
		//alloc debug output
		dataHost = (SocialForceRoomAgentData*)malloc(sizeof(SocialForceRoomAgentData) * numAgentLocal);
		dataCopyHost = (SocialForceRoomAgentData*)malloc(sizeof(SocialForceRoomAgentData) * numAgentLocal);
		dataIdxArrayHost = new int[numAgentLocal];
#elif defined (VALIDATE)
		throughputHost = 0;
		cudaMemcpyToSymbol(throughput, &throughputHost, sizeof(int));
#endif

		//add original agents
		int gSize = GRID_SIZE(numAgentLocal);
		addAgentsOnDevice<<<gSize, BLOCK_SIZE>>>(random, numAgentLocal, clones[0]->cloneDev);
		cudaMemcpy(clones[0]->unsortedAgentPtrArray, clones[0]->agentsHost->agentPtrArray,
				modelHostParams.AGENT_NO * sizeof(void*), cudaMemcpyDeviceToDevice);
		for (int i = 1; i < numClones; i++) {
			cudaMemcpy(clones[i]->unsortedAgentPtrArray, clones[0]->agentsHost->agentPtrArray,
				modelHostParams.AGENT_NO * sizeof(void*), cudaMemcpyDeviceToDevice);
			cudaMemcpy(clones[i]->clonedWorldHost->allAgents, clones[0]->agentsHost->agentPtrArray,
				modelHostParams.AGENT_NO * sizeof(void*), cudaMemcpyDeviceToDevice);
		}

		//timer related
		cudaEventCreate(&timerStart);
		cudaEventCreate(&timerStop);
		cudaEventRecord(timerStart, 0);

		getLastCudaError("start");

	}
	__host__ void preStep()
	{
		//switch world
		
#ifdef _WIN32
#ifdef CLONE
		int chosen = (GSimVisual::clicks + 7) % numClones;
		GSimVisual::getInstance().setWorld(clones[chosen]->clonedWorld, clones[chosen]->agentsHost->numElem);
		printf("Chosen clone: %d. Config: [%d, %d, %d]\n", 
			chosen, 
			clones[chosen]->cloneidArray[0], 
			clones[chosen]->cloneidArray[1], 
			clones[chosen]->cloneidArray[2]);
#endif
#endif
		getLastCudaError("copyHostToDevice");
	}
	__host__ void step()
	{
		float time = 0;
		cudaEvent_t timerStart, timerStop;
		cudaEventCreate(&timerStart);
		cudaEventCreate(&timerStop);

		TIMER_START(0);
		//1. run the original copy
		clones[0]->stepPhase1(NULL);
		clones[0]->stepPhase2(this);
		TIMER_END(0, "0", 0);

#ifdef CLONE
		//2. run the clones
		int childVal[NUM_GATES];
		int fatherVal[NUM_GATES];
		for (int i = 1; i < numClones; i++) {
			decode(i, childVal);
			fatherCloneidArray(childVal, fatherVal);
			int code = encode(fatherVal);
			SocialForceRoomClone *fatherClone = clones[code];
			clones[i]->stepPhase1(fatherClone);
			clones[i]->stepPhase2(this);
			clones[i]->stepPhase3();
		}
#endif

		//debug info, print the real data of original agents and cloned agents, or throughputs

		//5. swap data and dataCopy
		clones[0]->agentsHost->swapPool();
#ifdef CLONE
		for (int i = 1; i < numClones; i++) {
			clones[i]->agentsHost->swapPool();
		}
#endif

		//paint related stuff
#ifdef _WIN32
		GSimVisual::getInstance().animate();
#endif
		getLastCudaError("step");
	}
	__host__ void stop()
	{
		//fout.close();

		float time;
		cudaDeviceSynchronize();
		cudaEventRecord(timerStop, 0);
		cudaEventSynchronize(timerStop);
		cudaEventElapsedTime(&time, timerStart, timerStop);
		std::cout<<time<<std::endl;
#ifdef _WIN32
		GSimVisual::getInstance().stop();
#endif
	}
};

#ifdef CLONE
int SocialForceRoomClone::cloneCount = 0;
__host__ void SocialForceRoomClone::stepPhase1(SocialForceRoomClone *fatherClone) {

	if (fatherClone == NULL) {
		this->agentsHost->registerPool(this->clonedWorldHost, NULL, this->agents);
		util::genNeighbor(this->clonedWorld, this->clonedWorldHost, this->agentsHost->numElem);
		return;
	}

	int numFatherAgent = fatherClone->agentsHost->numElem;
	if (numFatherAgent == 0)
		return;

	float time = 0;
	cudaEvent_t timerStart, timerStop;
	cudaEventCreate(&timerStart);
	cudaEventCreate(&timerStop);

	//1.1 clone agents
	int gSize = GRID_SIZE(numFatherAgent);
	cudaDeviceSynchronize();
	TIMER_START(cloneStream);
	cloneKernel<<<gSize, BLOCK_SIZE, 0, cloneStream>>>(fatherClone->cloneDev, this->cloneDev, numFatherAgent, cloneid);
	TIMER_END(cloneid, "1",cloneStream);


	//2. run the cloned copy
	//2.1. register the cloned agents to the c1loned world
	TIMER_START(cloneStream);
	cudaMemcpyAsync(this->unsortedAgentPtrArray,
		fatherClone->unsortedAgentPtrArray, 
		modelHostParams.MAX_AGENT_NO * sizeof(void*), 
		cudaMemcpyDeviceToDevice, 
		cloneStream);
	cudaMemcpyAsync(this->clonedWorldHost->allAgents, 
		fatherClone->unsortedAgentPtrArray,
		modelHostParams.MAX_AGENT_NO * sizeof(void*),
		cudaMemcpyDeviceToDevice,
		cloneStream);
	TIMER_END(cloneid, "2",cloneStream);

	inspect<<<gSize, BLOCK_SIZE>>>(this->cloneDev, fatherClone->cloneDev);

	TIMER_START(cloneStream);
	//necessary cleanup, reason: 1. accumulative clean up of the step, 2. prepare the following step
	this->agentsHost->cleanup(this->agents);
	TIMER_END(cloneid, "2.1",cloneStream);
	getLastCudaError("stepstepPhase1");

	int numChildAgents = this->agentsHost->numElem;
	if (numChildAgents == 0)
		return;
	
	TIMER_START(cloneStream);
	replaceOriginalWithClone<<<gSize, BLOCK_SIZE, 0, cloneStream>>>(this->cloneDev,	numChildAgents);
	cudaMemcpyAsync(this->clonedWorldHost->allAgents, 
		this->unsortedAgentPtrArray,
		modelHostParams.MAX_AGENT_NO * sizeof(void*),
		cudaMemcpyDeviceToDevice,
		cloneStream);
	TIMER_END(cloneid, "2.2",cloneStream);

	//2.2. sort world and worldClone
	TIMER_START(cloneStream);
	util::genNeighbor(this->clonedWorld, this->clonedWorldHost, numChildAgents);
	TIMER_END(cloneid, "2.3",cloneStream);
	getLastCudaError("stepstepPhase2");
}
__host__ void SocialForceRoomClone::stepPhase2(SocialForceRoomModel *modelHost) {
	float time = 0;
	cudaEvent_t timerStart, timerStop;
	cudaEventCreate(&timerStart);
	cudaEventCreate(&timerStop);

	int numAgentsB = this->agentsHost->numElem;
	if (numAgentsB == 0)
		return;

	int gSize = GRID_SIZE(numAgentsB);
	//2.3. step the cloned copy
	TIMER_START(cloneStream);
	this->agentsHost->stepPoolAgent(modelHost->model, cloneStream);
	TIMER_END(cloneid, "2.4",cloneStream);

}
__host__ void SocialForceRoomClone::stepPhase3() {
	float time = 0;
	cudaEvent_t timerStart, timerStop;
	cudaEventCreate(&timerStart);
	cudaEventCreate(&timerStop);

	int numAgentsB = this->agentsHost->numElem;
	if (numAgentsB == 0)
		return;

	int gSize = GRID_SIZE(numAgentsB);

#ifdef CLONE_COMPARE
	//3. double check
	TIMER_START(cloneStream);
	compareOriginAndClone<<<gSize, BLOCK_SIZE, 0, cloneStream>>>(this->cloneDev, numAgentsB);
	TIMER_END(cloneid, "3",cloneStream);
#endif

	cudaEventDestroy(timerStart);
	cudaEventDestroy(timerStop);
	getLastCudaError("stepstepPhase3");
}
#endif

__device__ double correctCrossBoader(double val, double limit)
{
	if (val > limit)
		return limit-0.001;
	else if (val < 0)
		return 0;
	return val;
}
class SocialForceRoomAgent : public GAgent {
public:
	GRandom *random;
	GWorld *myWorld;

	int id;
	int cloneid;
#ifdef CLONE
	int flagCloning[NUM_GATES];
	int flagCloned[NUM_GATES];
	int cloneidArray[NUM_GATES];
#endif

	SocialForceRoomAgent *myOrigin;
	//double gateSize;

	__device__ void computeIndivSocialForceRoom(const SocialForceRoomAgentData &myData, const SocialForceRoomAgentData &otherData, double2 &fSum){
		double cMass = 100;
		//my data
		const float2& loc = myData.loc;
		const double2& goal = myData.goal;
		const double2& velo = myData.velocity;
		const double& v0 = myData.v0;
		const double& mass = myData.mass;
		//other's data
		const float2& locOther = otherData.loc;
		const double2& goalOther = otherData.goal;
		const double2& veloOther = otherData.velocity;
		const double& v0Other = otherData.v0;
		const double& massOther = otherData.mass;

		double d = 1e-15 + sqrt((loc.x - locOther.x) * (loc.x - locOther.x) + (loc.y - locOther.y) * (loc.y - locOther.y));
		double dDelta = mass / cMass + massOther / cMass - d;
		double fExp = A * exp(dDelta / B);
		double fKg = dDelta < 0 ? 0 : k1 *dDelta;
		double nijx = (loc.x - locOther.x) / d;
		double nijy = (loc.y - locOther.y) / d;
		double fnijx = (fExp + fKg) * nijx;
		double fnijy = (fExp + fKg) * nijy;
		double fkgx = 0;
		double fkgy = 0;
		if (dDelta > 0) {
			double tix = - nijy;
			double tiy = nijx;
			fkgx = k2 * dDelta;
			fkgy = k2 * dDelta;
			double vijDelta = (veloOther.x - velo.x) * tix + (veloOther.y - velo.y) * tiy;
			fkgx = fkgx * vijDelta * tix;
			fkgy = fkgy * vijDelta * tiy;
		}
		fSum.x += fnijx + fkgx;
		fSum.y += fnijy + fkgy;
	}
	__device__ void computeForceWithWall(const SocialForceRoomAgentData &dataLocal, obstacleLine &wall, const int &cMass, double2 &fSum) {
		double diw, crx, cry;
		const float2 &loc = dataLocal.loc;
		
		diw = wall.pointToLineDist(loc, crx, cry, this->id);
		double virDiw = DIST(loc.x, loc.y, crx, cry);

		//if (stepCount == MONITOR_STEP && this->id == 263) {
		//	printf("dist: %f, cross: (%f, %f)\n", diw, crx, cry);
		//}

		double niwx = (loc.x - crx) / virDiw;
		double niwy = (loc.y - cry) / virDiw;
		double drw = dataLocal.mass / cMass - diw;
		double fiw1 = A * exp(drw / B);
		if (drw > 0)
			fiw1 += k1 * drw;
		double fniwx = fiw1 * niwx;
		double fniwy = fiw1 * niwy;

		double fiwKgx = 0, fiwKgy = 0;
		if (drw > 0)
		{
			double fiwKg = k2 * drw * (dataLocal.velocity.x * (-niwy) + dataLocal.velocity.y * niwx);
			fiwKgx = fiwKg * (-niwy);
			fiwKgy = fiwKg * niwx;
		}

		fSum.x += fniwx - fiwKgx;
		fSum.y += fniwy - fiwKgy;
	}
	__device__ void computeWallImpaction(const SocialForceRoomAgentData &dataLocal, obstacleLine &wall, const double2 &newVelo, const double &tick, double &mint){
		double crx, cry, tt;
		const float2 &loc = dataLocal.loc;
		int ret = wall.intersection2LineSeg(
			loc.x, 
			loc.y, 
			loc.x + 0.5 * newVelo.x * tick,
			loc.y + 0.5 * newVelo.y * tick,
			crx,
			cry
			);
		if (ret == 1) 
		{
			if (fabs(crx - loc.x) > 0)
				tt = (crx - loc.x) / (newVelo.x * tick);
			else
				tt = (crx - loc.y) / (newVelo.y * tick + 1e-20);
			if (tt < mint)
				mint = tt;
		}
	}
	__device__ void computeDirection(const SocialForceRoomAgentData &dataLocal, double2 &dvt) {
		//my data
		const float2& loc = dataLocal.loc;
		const double2& goal = dataLocal.goal;
		const double2& velo = dataLocal.velocity;
		const double& v0 = dataLocal.v0;
		const double& mass = dataLocal.mass;
		
		dvt.x = 0;	dvt.y = 0;
		double2 diff; diff.x = 0; diff.y = 0;
		double d0 = sqrt((loc.x - goal.x) * (loc.x - goal.x) + (loc.y - goal.y) * (loc.y - goal.y));
		diff.x = v0 * (goal.x - loc.x) / d0;
		diff.y = v0 * (goal.y - loc.y) / d0;
		dvt.x = (diff.x - velo.x) / tao;
		dvt.y = (diff.y - velo.y) / tao;
	}
	__device__ void computeSocialForceRoom(SocialForceRoomAgentData &dataLocal, double2 &fSum) {
		GWorld *world = this->myWorld;
		iterInfo info;

		fSum.x = 0; fSum.y = 0;
		SocialForceRoomAgentData *otherData, otherDataLocal;
		double ds = 0;

		int neighborCount = 0;

		world->neighborQueryInit(dataLocal.loc, 6, info);
		otherData = world->nextAgentDataFromSharedMem<SocialForceRoomAgentData>(info);
		while (otherData != NULL) {
			otherDataLocal = *otherData;
			SocialForceRoomAgent *otherPtr = (SocialForceRoomAgent*)otherData->agentPtr;
			ds = length(otherDataLocal.loc - dataLocal.loc);
			if (ds < 6 && ds > 0 ) {
				neighborCount++;
				computeIndivSocialForceRoom(dataLocal, otherDataLocal, fSum);
#ifdef CLONE
				for (int i = 0; i < NUM_GATES; i++) {
					this->flagCloning[i] |= otherPtr->flagCloned[i];
				}
				//if (stepCount == 204 && dataLocal.id == 4554) {
				//	printf("[%d, %d]", this->cloneid, otherPtr->cloned[0]);
				//}
#endif
				/*
				if (stepCount == 0 && dataLocal.id == 263) {
					printf("%d, %d, [%f, %f], [%f, %f], [%f, %f]\n", stepCount, otherDataLocal.id, otherDataLocal.loc.x, otherDataLocal.loc.y, otherDataLocal.velocity.x, otherDataLocal.velocity.y, otherDataLocal.goal.x, otherDataLocal.goal.y);
				}
				*/
			}
			otherData = world->nextAgentDataFromSharedMem<SocialForceRoomAgentData>(info);
		}
	}
	__device__ void chooseNewGoal(const float2 &newLoc, double epsilon, double2 &newGoal) {
		
		double2 center = make_double2(modelDevParams.WIDTH / 2, modelDevParams.HEIGHT / 2);
		if (newLoc.x < center.x && newLoc.y < center.y) {
			if ((newLoc.x + epsilon >= 0.5 * modelDevParams.WIDTH) 
				//&& (newLoc.y + epsilon > 0.3 * modelDevParams.HEIGHT - gateSize) 
				//&& (newLoc.y - epsilon < 0.3 * modelDevParams.HEIGHT + gateSize)
					) 
			{
				newGoal.x = 0.9 * modelDevParams.WIDTH;
				newGoal.y = 0.3 * modelDevParams.HEIGHT;
			}
		}
		else if (newLoc.x > center.x && newLoc.y < center.y) {
			if ((newLoc.x + epsilon >= 0.9 * modelDevParams.WIDTH) 
				//&& (newLoc.y + epsilon > 0.3 * modelDevParams.HEIGHT - gateSize) 
				//&& (newLoc.y - epsilon < 0.3 * modelDevParams.HEIGHT + gateSize)
					) 
			{
				//newGoal.x = modelDevParams.WIDTH;
				//newGoal.y = 0;
			}
		}
		else if (newLoc.x < center.x && newLoc.y > center.y) {
			if ((newLoc.y - epsilon <= 0.5 * modelDevParams.HEIGHT) 
				//&& (newLoc.x + epsilon > 0.3 * modelDevParams.WIDTH - gateSize) 
				//&& (newLoc.x - epsilon < 0.3 * modelDevParams.WIDTH + gateSize)
					) 
			{
				newGoal.x = 0.5 * modelDevParams.WIDTH;
				newGoal.y = 0.3 * modelDevParams.HEIGHT;
			}
		}
		else if (newLoc.x > center.x && newLoc.y > center.y) {
			if ((newLoc.x - epsilon <= 0.5 * modelDevParams.WIDTH) 
				//&& (newLoc.y + epsilon > 0.7 * modelDevParams.WIDTH - gateSize) 
				//&& (newLoc.y - epsilon < 0.7 * modelDevParams.WIDTH + gateSize)
					) 
			{
				newGoal.x = 0.3 * modelDevParams.WIDTH;
				newGoal.y = 0.5 * modelDevParams.HEIGHT;
			}
		}
	}
	__device__ void alterWall(obstacleLine &wall, int wallId) {
		double gateSize;
		if (9 == wallId)	{int choice = cloneidArray[1]; gateSize = gate1Sizes[choice]; wall.sx += gateSize;}
		if (6 == wallId)	{int choice = cloneidArray[1]; gateSize = gate1Sizes[choice]; wall.ex -= gateSize;}
		if (7 == wallId)	{int choice = cloneidArray[2]; gateSize = gate2Sizes[choice]; wall.ey -= gateSize;}
		if (8 == wallId)	{int choice = cloneidArray[2]; gateSize = gate2Sizes[choice]; wall.sy += gateSize;
								 choice = cloneidArray[0]; gateSize = gate0Sizes[choice]; wall.ey -= gateSize;}
		if (5 == wallId)	{int choice = cloneidArray[0]; gateSize = gate0Sizes[choice]; wall.sy += gateSize;}
	}
	__device__ void alterGate(obstacleLine &gate, int i) {
		double2 gateLoc = gateLocs[i];
		gate.sx = gate.ex = gateLoc.x;
		gate.sy = gate.ey = gateLoc.y;
		double gateSize;
		if (i == 0) {gateSize = gate0Sizes[NUM_GATE_0_CHOICES-1]; gate.sy -= gateSize; gate.ey += gateSize;} 
		if (i == 1) {gateSize = gate1Sizes[NUM_GATE_1_CHOICES-1]; gate.sx -= gateSize; gate.ex += gateSize;} 
		if (i == 2) {gateSize = gate2Sizes[NUM_GATE_2_CHOICES-1]; gate.sy -= gateSize; gate.ey += gateSize;} 
	}
	__device__ void step(GModel *model){
		double cMass = 100;

		SocialForceRoomAgentData dataLocal = *(SocialForceRoomAgentData*)this->data;

		const float2& loc = dataLocal.loc;
		const double2& goal = dataLocal.goal;
		const double2& velo = dataLocal.velocity;
		const double& v0 = dataLocal.v0;
		const double& mass = dataLocal.mass;

		//compute the direction
		double2 dvt;
		computeDirection(dataLocal, dvt);

		//compute force with other agents
		double2 fSum; 
		computeSocialForceRoom(dataLocal, fSum);
#ifdef VALIDATE
		if (stepCount == MONITOR_STEP && this->id == MONITOR_ID) {
			printf("fSum: (%f, %f)\n", fSum.x, fSum.y);
		}
#endif

		//compute force with wall
		for (int i = 0; i < NUM_WALLS; i++) {
			obstacleLine wall = walls[i];
			alterWall(wall, i);
			computeForceWithWall(dataLocal, wall, cMass, fSum);
		}

#ifdef VALIDATE
		if (stepCount == MONITOR_STEP && this->id == MONITOR_ID) {
			printf("fSum: (%f, %f)\n", fSum.x, fSum.y);
		}
#endif

#ifdef CLONE
		//decision point A: impaction from wall
		int gateMask = ~0;
		for (int i = 0; i < NUM_GATES; i++) {
			obstacleLine gate;
			alterGate(gate, i);
			if(gate.pointToLineDist(loc) < 2) {
				//current agent impacted by a chosing gate;
				this->flagCloning[i] = gateMask;
			}
		}
#endif

		//sum up
		dvt.x += fSum.x / mass;
		dvt.y += fSum.y / mass;

#ifdef VALIDATE
		if (stepCount == MONITOR_STEP && this->id == MONITOR_ID) {
			printf("dvt: (%f, %f)\n", dvt.x, dvt.y);
		}
#endif
		double2 newVelo = dataLocal.velocity;
		float2 newLoc = dataLocal.loc;
		double2 newGoal = dataLocal.goal;

#ifdef VALIDATE
		if (stepCount == MONITOR_STEP && this->id == MONITOR_ID) {
			printf("oldVelo: (%f, %f)\n", newVelo.x, newVelo.y);
		}
#endif

		double tick = 0.1;
		newVelo.x += dvt.x * tick * (1);// + this->random->gaussian() * 0.1);
		newVelo.y += dvt.y * tick * (1);// + this->random->gaussian() * 0.1);
		double dv = sqrt(newVelo.x * newVelo.x + newVelo.y * newVelo.y);

		if (dv > maxv) {
			newVelo.x = newVelo.x * maxv / dv;
			newVelo.y = newVelo.y * maxv / dv;
		}

		double mint = 1;
		for (int i = 0; i < NUM_WALLS; i++) {
			obstacleLine wall = walls[i];
			alterWall(wall, i);
			computeWallImpaction(dataLocal, wall, newVelo, tick, mint);
		}
		
#ifdef VALIDATE
		if (stepCount == MONITOR_STEP && this->id == MONITOR_ID) {
			printf("dv: %f\n", dv);
			printf("newVelo: (%f, %f)\n", newVelo.x, newVelo.y);
		}
#endif
		newVelo.x *= mint;
		newVelo.y *= mint;
		newLoc.x += newVelo.x * tick;
		newLoc.y += newVelo.y * tick;

#ifdef VALIDATE
		if (stepCount == MONITOR_STEP && this->id == MONITOR_ID) {
			printf("mint: %f\n", mint);
			printf("newVelo: (%f, %f)\n", newVelo.x, newVelo.y);
		}
#endif
		double goalTemp = goal.x;

		chooseNewGoal(newLoc, mass/cMass, newGoal);

		newLoc.x = correctCrossBoader(newLoc.x, modelDevParams.WIDTH);
		newLoc.y = correctCrossBoader(newLoc.y, modelDevParams.HEIGHT);

		SocialForceRoomAgentData dataCopyLocal = dataLocal;
		dataCopyLocal.loc = newLoc;
		dataCopyLocal.velocity = newVelo;
		dataCopyLocal.goal = newGoal;

		
#ifdef VALIDATE
		if (stepCount == MONITOR_STEP && this->id == MONITOR_ID) {
			printf("BEF: %d, %d, [%f, %f], [%f, %f], [%f, %f]\n", stepCount, this->id, dataLocal.loc.x, dataLocal.loc.y, dataLocal.velocity.x, dataLocal.velocity.y, dataLocal.goal.x, dataLocal.goal.y);
			printf("AFT: %d, %d, [%f, %f], [%f, %f], [%f, %f]\n", stepCount, this->id, dataCopyLocal.loc.x, dataCopyLocal.loc.y, dataCopyLocal.velocity.x, dataCopyLocal.velocity.y, dataCopyLocal.goal.x, dataCopyLocal.goal.y);
		}
#endif
		

		*(SocialForceRoomAgentData*)this->dataCopy = dataCopyLocal;
	}
	__device__ void fillSharedMem(void *dataPtr){
		SocialForceRoomAgentData *dataSmem = (SocialForceRoomAgentData*)dataPtr;
		SocialForceRoomAgentData *dataAgent = (SocialForceRoomAgentData*)this->data;
		*dataSmem = *dataAgent;
	}
	__device__ void init(GWorld *myW, GRandom *myR, int dataSlot,
		AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData> *myPool) {
		this->myWorld = myW;
		this->random = myR;
		this->color = colorConfigs.green;
		this->id = dataSlot;

		this->cloneid = 0;
#ifdef CLONE
		for (int i = 0; i < NUM_GATES; i++) {
			this->cloneidArray[i] = 0;
			this->flagCloning[i] = 0;
			this->flagCloned[i] = 0;
		}
#endif
		this->myOrigin = NULL;

		SocialForceRoomAgentData dataLocal; //= &sfModel->originalAgents->dataArray[dataSlot];

		dataLocal.agentPtr = this;
		dataLocal.loc.x = (0.5 + 0.4 * this->random->uniform()) * modelDevParams.WIDTH - 0.1;
		dataLocal.loc.y = (0.5 + 0.4 * this->random->uniform()) * modelDevParams.HEIGHT - 0.1;
		dataLocal.velocity.x = 2;//4 * (this->random->uniform()-0.5);
		dataLocal.velocity.y = 2;//4 * (this->random->uniform()-0.5);

		dataLocal.v0 = 2;
		dataLocal.mass = 50;

		dataLocal.goal = make_double2(0.5 * modelDevParams.WIDTH, 0.7 * modelDevParams.HEIGHT);
		//chooseNewGoal(dataLocal.loc, 0, dataLocal.goal);

		this->data = myPool->dataInSlot(dataSlot);
		this->dataCopy = myPool->dataCopyInSlot(dataSlot);
		*(SocialForceRoomAgentData*)this->data = dataLocal;
		*(SocialForceRoomAgentData*)this->dataCopy = dataLocal;
	}

#ifdef CLONE
	__device__ void initNewClone(const SocialForceRoomAgent &agent, 
		SocialForceRoomAgent *originPtr, 
		int dataSlot, 
		SocialForceRoomClone *clone,
		int cloneid) 
	{
		this->myWorld = clone->clonedWorld;
		this->color = clone->color;

		this->cloneid = cloneid;
		this->id = agent.id;
		for (int i = 0; i < NUM_GATES; i++) {
			this->cloneidArray[i] = clone->cloneidArray[i];
			this->flagCloning[i] = originPtr->flagCloning[i]; // or 0?
			this->flagCloned[i] = originPtr->flagCloned[i]; // or 0?
		}
		this->myOrigin = originPtr;

		SocialForceRoomAgentData dataLocal, dataCopyLocal;

		dataLocal = *(SocialForceRoomAgentData*)agent.data;
		dataCopyLocal = *(SocialForceRoomAgentData*)agent.dataCopy;

		dataLocal.agentPtr = this;
		dataCopyLocal.agentPtr = this;

		if (stepCount % 2 == 0) {
			this->data = clone->agents->dataInSlot(dataSlot);
			this->dataCopy = clone->agents->dataCopyInSlot(dataSlot);
		} else {
			this->data = clone->agents->dataCopyInSlot(dataSlot);
			this->dataCopy = clone->agents->dataInSlot(dataSlot);
		}

		*(SocialForceRoomAgentData*)this->data = dataLocal;
		*(SocialForceRoomAgentData*)this->dataCopy = dataCopyLocal;
	}
#endif
};
__device__ void SocialForceRoomAgentData::putDataInSmem(GAgent *ag){
	*this = *(SocialForceRoomAgentData*)ag->data;
}
__global__ void addAgentsOnDevice(GRandom *myRandom, int numAgent, SocialForceRoomClone *clone0)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAgent){
		int dataSlot = idx;
		SocialForceRoomAgent *ag = clone0->agents->agentInSlot(dataSlot);
		ag->init(clone0->clonedWorld, myRandom, dataSlot, clone0->agents);
		clone0->agents->add(ag, dataSlot);
	}
}
#ifdef CLONE
__global__ void cloneKernel(SocialForceRoomClone *fatherClone, 
							SocialForceRoomClone *childClone,
							int numAgentLocal, int cloneid)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAgentLocal){ // user init step
		SocialForceRoomAgent ag, *agPtr = fatherClone->agents->agentPtrArray[idx];
		ag = *agPtr;

		int cloneLevelLocal = childClone->cloneLevel;
		int cloneMaskLocal = childClone->cloneMasks[cloneLevelLocal];
		int cloneDecision = ~ag.flagCloned[cloneLevelLocal] & cloneMaskLocal & ag.flagCloning[cloneLevelLocal];
		if (cloneDecision > 0) {
			ag.flagCloned[cloneLevelLocal] |= cloneMaskLocal;
			ag.flagCloning[cloneLevelLocal] &= ~cloneMaskLocal;
			*agPtr = ag;

			AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData> *childAgentPool = childClone->agents;

			int agentSlot = childAgentPool->agentSlot();
			int dataSlot =childAgentPool->dataSlot(agentSlot);

			SocialForceRoomAgent *ag2 = childAgentPool->agentInSlot(dataSlot);
			ag2->initNewClone(ag, agPtr, dataSlot, childClone, cloneid);
			childAgentPool->add(ag2, agentSlot);
		}
	}
}

__global__ void replaceOriginalWithClone(SocialForceRoomClone* childClone, int numClonedAgent)
{
	if (childClone->unsortedAgentPtrArray == NULL) {
		printf("cloneid: [%d, %d, %d]", childClone->cloneidArray[0],childClone->cloneidArray[1],childClone->cloneidArray[2]);
		return;
	}
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numClonedAgent){
		SocialForceRoomAgent *ag = childClone->agents->agentPtrArray[idx];
		childClone->unsortedAgentPtrArray[ag->id] = ag;
	}
}

__global__ void compareOriginAndClone(SocialForceRoomClone *childClone, int numClonedAgents) 
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numClonedAgents) {
		SocialForceRoomAgent *clonedAg = childClone->agents->agentPtrArray[idx];
		SocialForceRoomAgent *originalAg = clonedAg->myOrigin;
		SocialForceRoomAgentData clonedAgData = *(SocialForceRoomAgentData*) clonedAg->dataCopy;
		SocialForceRoomAgentData originalAgData = *(SocialForceRoomAgentData*) originalAg->dataCopy;

		bool match;
		//compare equivalence of two copies of data;
#define DELTA DBL_EPSILON
		double diffLocX = abs(clonedAgData.loc.x - originalAgData.loc.x);
		double diffLocY = abs(clonedAgData.loc.y - originalAgData.loc.y);
		double diffVelX = abs(clonedAgData.velocity.x - originalAgData.velocity.x);
		double diffVelY = abs(clonedAgData.velocity.y - originalAgData.velocity.y);
		match = (diffLocX <= DELTA)
			&& (diffLocY <= DELTA)
			&& (diffVelX <= DELTA)
			&& (diffVelY <= DELTA); 
			//&& (clonedAgData.goal.x - originalAgData.goal.x == DELTA)
			//&& (clonedAgData.goal.y - originalAgData.goal.y == DELTA);
		if (match) {
			//remove from cloned set, reset clone state to non-cloned
			childClone->agents->remove(idx);
			int cloneLevelLocal = childClone->cloneLevel;
			int cloneMaskLocal = childClone->cloneMasks[cloneLevelLocal];

			atomicAnd(&originalAg->flagCloned[cloneLevelLocal], ~cloneMaskLocal);
			atomicAnd(&originalAg->flagCloning[cloneLevelLocal], ~cloneMaskLocal);
		}
		/*
#ifdef _DEBUG
		else {
			originalAg->color = colorConfigs.blue;
			printf("step: %d\n", stepCount);
			printf("\t origin:%d, clone:%d", originalAgData.id, clonedAgData.id);
			printf("\t diffLocSqr:[%f, %f]", diffLocX * diffLocX, diffLocY * diffLocY);
			printf("\t diffVelSqr:[%f, %f]", diffVelX * diffVelX, diffVelY * diffVelY);
		}
#endif
		*/
	}
}
#endif

#endif