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

#define NUM_CLONE 2
#define NUM_WALLS 10
#define NUM_GATES 4
#define CHOSEN_CLONE_ID NUM_CLONE
__constant__ double2 gateLocs[NUM_GATES];
__constant__ obstacleLine walls[NUM_WALLS];
__constant__ double gateSizes[(NUM_CLONE + 1) * NUM_GATES]; 
__constant__ uchar4 colors[NUM_CLONE];

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


__global__ void addAgentsOnDevice(GRandom *myRandom, int numAgent, GWorld *world,
								  AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData> *agents);

__global__ void replaceOriginalWithClone(GAgent **originalAgents, SocialForceRoomAgent **clonedAgents, 
										 int numClonedAgent);

__global__ void cloneKernel(SocialForceRoomAgent **originalAgents, 
							AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData> *pool, 
							int numAgentLocal, GWorld *world, int cloneid);

__global__ void compareOriginAndClone(AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData> *clonedPool, 
									  GWorld *clonedWorld, int numClonedAgents, int cloneid) ;


#ifdef CLONE
class SocialForceRoomClone {
private:
	static int cloneCount;
	cudaStream_t cloneStream;
public:
	AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData> *agents, *agentsHost;
	GWorld *clonedWorld, *clonedWorldHost;
	SocialForceRoomAgent **agentPtrArrayUnsorted;
	int cloneid;

	__host__ SocialForceRoomClone(int num) {
		cudaStreamCreate(&cloneStream);

		agentsHost = new AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData>(0, num * 2, sizeof(SocialForceRoomAgentData));
		util::hostAllocCopyToDevice<AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData> >(agentsHost, &agents);
		
		clonedWorldHost = new GWorld();
		util::hostAllocCopyToDevice<GWorld>(clonedWorldHost, &clonedWorld);

		//alloc untouched agent array
		cudaMalloc((void**)&agentPtrArrayUnsorted, num * sizeof(SocialForceRoomAgent*) );

		cloneid = cloneCount++;
	}

	__host__ void start(SocialForceRoomAgent** originalAgents, int num) {
		//copy the init agents to untouched agent list
		cudaMemcpy(agentPtrArrayUnsorted, originalAgents, num * sizeof(SocialForceRoomAgent*), cudaMemcpyDeviceToDevice);
	}

	__host__ void stepPhase1(SocialForceRoomAgent** originalAgents, int num);
	
	__host__ void stepPhase2(SocialForceRoomModel *modelHost);

	__host__ void stepPhase3();

	__host__ void stepPhase4();
};
#endif
class SocialForceRoomModel : public GModel {
public:
	GRandom *random, *randomHost;
	cudaEvent_t timerStart, timerStop;
	SocialForceRoomClone *originalClone;

	std::fstream fout;
#ifdef CLONE
	SocialForceRoomClone *clones[NUM_CLONE];
#endif

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

		double *gateSizesHost = (double*)malloc(sizeof(double) * (1 + NUM_CLONE) * NUM_GATES);
		uchar4 *colorsHost = (uchar4*)malloc(sizeof(uchar4) * NUM_CLONE);
		srand(time(NULL));
		for (int i = 0; i < NUM_CLONE + 1; i++) {
			for (int j = 0; j < NUM_GATES; j++)
				gateSizesHost[i * NUM_GATES + j] = (i + 1) * 2;
		}
		cudaMemcpyToSymbol(gateSizes, &gateSizesHost[0], (1 + NUM_CLONE) * NUM_GATES * sizeof(double));

		for (int i = 0; i < NUM_CLONE; i++) {
			int r = rand();
			memcpy(&colorsHost[i], &r, sizeof(uchar4));
		}
		cudaMemcpyToSymbol(colors, &colorsHost[0], NUM_CLONE * sizeof(uchar4));

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
		gateLocsHost[0] = make_double2(0.9 * wLocal, 0.3 * hLocal);
		gateLocsHost[1] = make_double2(0.5 * wLocal, 0.7 * hLocal);
		gateLocsHost[2] = make_double2(0.3 * wLocal, 0.5 * hLocal);
		gateLocsHost[3] = make_double2(0.5 * wLocal, 0.3 * hLocal);
		cudaMemcpyToSymbol(gateLocs, &gateLocsHost, NUM_GATES * sizeof(double2));

		originalClone = new SocialForceRoomClone(modelHostParams.AGENT_NO);
		for(int i = 0; i < NUM_CLONE; i++) {
			clones[i] = new SocialForceRoomClone(modelHostParams.AGENT_NO);
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
		addAgentsOnDevice<<<gSize, BLOCK_SIZE>>>(random, numAgentLocal, 
			originalClone->clonedWorld, originalClone->agents);

		//initialize unsorted agent array in clones with original agents
#ifdef CLONE
		for (int i = 0; i < NUM_CLONE; i++) {
			clones[i]->start(originalClone->agentsHost->agentPtrArray, numAgentLocal);
		}
#endif

		//paint related
#ifdef _WIN32
		GSimVisual::getInstance().setWorld(this->world);
#endif
		//timer related
		cudaEventCreate(&timerStart);
		cudaEventCreate(&timerStop);
		cudaEventRecord(timerStart, 0);

		getLastCudaError("start");

	}

	__host__ void preStep()
	{
		//switch world
		int numInst = NUM_CLONE + 1;
		
#ifdef _WIN32
		if (GSimVisual::clicks % numInst == 0)
			GSimVisual::getInstance().setWorld(originalClone->clonedWorld);
#ifdef CLONE
		else {
			int chosen = GSimVisual::clicks % numInst;
			GSimVisual::getInstance().setWorld(clones[chosen-1]->clonedWorld);
		}
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
		originalClone->stepPhase1(NULL, modelHostParams.AGENT_NO);
		originalClone->stepPhase2(this);
		TIMER_END(0, "0", 0);

		//2. run the clones
#ifdef CLONE
		//for (int i = 0; i < NUM_CLONE; i++) {
			//clones[i]->step(this->originalAgentsHost->agentPtrArray, numAgentLocal, this);
		//}
		for (int i = 0; i < NUM_CLONE; i++) {
			clones[i]->stepPhase1(originalClone->agentsHost->agentPtrArray, modelHostParams.AGENT_NO);
		}
		for (int i = 0; i < NUM_CLONE; i++) {
			clones[i]->stepPhase2(this);
		}
		for (int i = 0; i < NUM_CLONE; i++) {
			clones[i]->stepPhase3();
		}
#endif

		//debug info, print the real data of original agents and cloned agents, or throughputs

		//5. swap data and dataCopy
		originalClone->agentsHost->swapPool();
#ifdef CLONE
		for (int i = 0; i < NUM_CLONE; i++) {
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
__host__ void SocialForceRoomClone::stepPhase1(SocialForceRoomAgent** originalAgents, int num) {

	if (originalAgents == NULL) {
		this->agentsHost->registerPool(this->clonedWorldHost, NULL, this->agents);
		util::genNeighbor(this->clonedWorld, this->clonedWorldHost, this->agentsHost->numElem);
		return;
	}

	float time = 0;
	cudaEvent_t timerStart, timerStop;
	cudaEventCreate(&timerStart);
	cudaEventCreate(&timerStop);

	//1.1 clone agents
	int numAgentLocal = num;
	int gSize = GRID_SIZE(numAgentLocal);
	cudaDeviceSynchronize();
	TIMER_START(cloneStream);
	cloneKernel<<<gSize, BLOCK_SIZE, 0, cloneStream>>>(originalAgents, agents, numAgentLocal, clonedWorld, cloneid);
	TIMER_END(cloneid, "1",cloneStream);


	//2. run the cloned copy
	//2.1. register the cloned agents to the c1loned world
	TIMER_START(cloneStream);
	cudaMemcpyAsync(clonedWorldHost->allAgents,
		agentPtrArrayUnsorted, 
		num * sizeof(void*), 
		cudaMemcpyDeviceToDevice, cloneStream);
	TIMER_END(cloneid, "2",cloneStream);

	TIMER_START(cloneStream);
	this->agentsHost->cleanup(this->agents);
	TIMER_END(cloneid, "2.1",cloneStream);
	getLastCudaError("stepstepPhase1");

	int numAgentsB = this->agentsHost->numElem;
	if (numAgentsB != 0) {
		int gSize = GRID_SIZE(numAgentsB);

		TIMER_START(cloneStream);
		replaceOriginalWithClone<<<gSize, BLOCK_SIZE, 0, cloneStream>>>(
			clonedWorldHost->allAgents, 
			this->agentsHost->agentPtrArray, 
			numAgentsB);
		TIMER_END(cloneid, "2.2",cloneStream);

		//2.2. sort world and worldClone
		TIMER_START(cloneStream);
		util::genNeighbor(this->clonedWorld, this->clonedWorldHost, modelHostParams.AGENT_NO);
		TIMER_END(cloneid, "2.3",cloneStream);
		getLastCudaError("stepstepPhase2");
	}
}
__host__ void SocialForceRoomClone::stepPhase2(SocialForceRoomModel *modelHost) {
	float time = 0;
	cudaEvent_t timerStart, timerStop;
	cudaEventCreate(&timerStart);
	cudaEventCreate(&timerStop);

	int numAgentsB = this->agentsHost->numElem;
	if (numAgentsB != 0) {
		int gSize = GRID_SIZE(numAgentsB);
		//2.3. step the cloned copy
		TIMER_START(cloneStream);
		this->agentsHost->stepPoolAgent(modelHost->model, cloneStream);
		TIMER_END(cloneid, "2.4",cloneStream);
	}

}
__host__ void SocialForceRoomClone::stepPhase3() {
	float time = 0;
	cudaEvent_t timerStart, timerStop;
	cudaEventCreate(&timerStart);
	cudaEventCreate(&timerStop);

	int numAgentsB = this->agentsHost->numElem;
	if (numAgentsB != 0) {
		int gSize = GRID_SIZE(numAgentsB);

#ifdef CLONE_COMPARE
		//3. double check
		TIMER_START(cloneStream);
		compareOriginAndClone<<<gSize, BLOCK_SIZE, 0, cloneStream>>>(this->agents, clonedWorld, numAgentsB, cloneid);
		TIMER_END(cloneid, "3",cloneStream);
#endif
		cudaEventDestroy(timerStart);
		cudaEventDestroy(timerStop);
		getLastCudaError("stepstepPhase3");
	}
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
	bool cloned[NUM_CLONE];
	bool cloning[NUM_CLONE];
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
				for (int i = 0; i < NUM_CLONE; i++)
					if (otherPtr->cloned[i] == true) // decision point B: impaction from neighboring agent
						this->cloning[i] = true;
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
	__device__ void alterWall(obstacleLine &wall, int i) {
		double gateSize;
		int base = cloneid * NUM_GATES;
		if (2 == i)	{gateSize = gateSizes[base]; wall.ey += gateSize;}
		if (3 == i)	{gateSize = gateSizes[base]; wall.sy -= gateSize;}
		if (9 == i)	{gateSize = gateSizes[base+2]; wall.sx += gateSize;}
		if (6 == i)	{gateSize = gateSizes[base+2]; wall.ex -= gateSize;}
		if (7 == i)	{gateSize = gateSizes[base+3]; wall.ey -= gateSize;}
		if (8 == i)	{gateSize = gateSizes[base+3]; wall.sy += gateSize;
					 gateSize = gateSizes[base+3]; wall.ey -= gateSize;}
		if (5 == i)	{gateSize = gateSizes[base+1]; wall.sy += gateSize;}
	}
	__device__ void alterGate(obstacleLine &gate, int i) {
		double2 gateLoc = gateLocs[i];
		gate.sx = gate.ex = gateLoc.x;
		gate.sy = gate.ey = gateLoc.y;
		double gateSize = gateSizes[NUM_CLONE * NUM_GATES];
		if (i == 0) {gate.sy -= gateSize; gate.ey += gateSize;} 
		if (i == 1) {gate.sy -= gateSize; gate.ey += gateSize;} 
		if (i == 3) {gate.sy -= gateSize; gate.ey += gateSize;} 
		if (i == 2) {gate.sx -= gateSize; gate.ex += gateSize;} 
	}
	__device__ void step(GModel *model){
		SocialForceRoomModel *sfModel = (SocialForceRoomModel*)model;
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
		for (int i = 0; i < NUM_GATES; i++) {
			obstacleLine gate;
			alterGate(gate, i);
			if(gate.pointToLineDist(loc) < 2) {
				for (int i = 0; i < NUM_CLONE; i++)
					if (this->cloned[i] == false)
						this->cloning[i] = true;
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

		/*
		if (goal.x < 0.5 * modelDevParams.WIDTH
			&& (newLoc.x - mass/cMass <= 0.25 * modelDevParams.WIDTH) 
			&& (newLoc.y - mass/cMass > 0.5 * modelDevParams.HEIGHT - gateSize) 
			&& (newLoc.y - mass/cMass < 0.5 * modelDevParams.HEIGHT + gateSize)) 
		{
			newGoal.x = 0;
		}

		if (goal.x > 0.5 * modelDevParams.WIDTH
			&& (newLoc.x + mass/cMass >= 0.75 * modelDevParams.WIDTH) 
			&& (newLoc.y - mass/cMass > 0.5 * modelDevParams.HEIGHT - gateSize) 
			&& (newLoc.y - mass/cMass < 0.5 * modelDevParams.HEIGHT + gateSize)) 
		{
			newGoal.x = modelDevParams.WIDTH;
#ifdef VALIDATE
#ifdef CLONE
			if (goalTemp != newGoal.x && this->cloneid == CHOSEN_CLONE_ID) 
#else
			if (goalTemp != newGoal.x) 	
#endif
			{
				atomicInc(&throughput, 8192);
			}
#endif
		}
		*/


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
		
		for (int i = 0; i < NUM_CLONE; i++) {
			this->cloned[i] = false;
			this->cloning[i] = false;
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
		GWorld *clonedWorld,
		AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData> *pool,
		int cloneid) 
	{
		this->myWorld = clonedWorld;
		this->color = colors[cloneid-1];

		this->cloneid = cloneid;
		this->id = agent.id;
		for (int i = 0; i < NUM_CLONE; i++){
			this->cloned[i] = false;
			this->cloning[i] = false;
		}
		this->myOrigin = originPtr;

		SocialForceRoomAgentData dataLocal, dataCopyLocal;

		dataLocal = *(SocialForceRoomAgentData*)agent.data;
		dataCopyLocal = *(SocialForceRoomAgentData*)agent.dataCopy;

		dataLocal.agentPtr = this;
		dataCopyLocal.agentPtr = this;

		if (stepCount % 2 == 0) {
			this->data = pool->dataInSlot(dataSlot);
			this->dataCopy = pool->dataCopyInSlot(dataSlot);
		} else {
			this->data = pool->dataCopyInSlot(dataSlot);
			this->dataCopy = pool->dataInSlot(dataSlot);
		}

		*(SocialForceRoomAgentData*)this->data = dataLocal;
		*(SocialForceRoomAgentData*)this->dataCopy = dataCopyLocal;
	}
#endif
};

__device__ void SocialForceRoomAgentData::putDataInSmem(GAgent *ag){
	*this = *(SocialForceRoomAgentData*)ag->data;
}

__global__ void addAgentsOnDevice(GRandom *myRandom, int numAgent, GWorld *world,
								  AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData> *agents)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAgent){
		int dataSlot = idx;
		SocialForceRoomAgent *ag = agents->agentInSlot(dataSlot);
		ag->init(world, myRandom, dataSlot, agents);
		agents->add(ag, dataSlot);
	}
}

#ifdef CLONE
__global__ void cloneKernel(SocialForceRoomAgent **originalAgents, 
							AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData> *pool, 
							int numAgentLocal, GWorld *world, int cloneid)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAgentLocal){ // user init step
		SocialForceRoomAgent ag, *agPtr = originalAgents[idx];
		ag = *agPtr;

		int cloneidLocal = cloneid - 1;
		if( ag.cloning[cloneidLocal] == true && ag.cloned[cloneidLocal] == false) {
			ag.cloned[cloneidLocal] = true;
			ag.cloning[cloneidLocal] = false;
			*agPtr = ag;

			int agentSlot = pool->agentSlot();
			int dataSlot = pool->dataSlot(agentSlot);

			SocialForceRoomAgent *ag2 = pool->agentInSlot(dataSlot);
			ag2->initNewClone(ag, agPtr, dataSlot, world, pool, cloneid);
			pool->add(ag2, agentSlot);
		}
	}
}

__global__ void replaceOriginalWithClone(GAgent **originalAgents, SocialForceRoomAgent **clonedAgents, int numClonedAgent)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numClonedAgent){
		SocialForceRoomAgent *ag = clonedAgents[idx];
		originalAgents[ag->id] = ag;
	}
}

__global__ void compareOriginAndClone(
	AgentPool<SocialForceRoomAgent, SocialForceRoomAgentData> *clonedPool,
	GWorld *clonedWorld,
	int numClonedAgents,
	int cloneid) 
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numClonedAgents) {
		SocialForceRoomAgent *clonedAg = clonedPool->agentPtrArray[idx];
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
			clonedPool->remove(idx);
			int cloneidLocal = cloneid - 1;
			originalAg->cloned[cloneidLocal] = false;
			originalAg->cloning[cloneidLocal] = false;
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