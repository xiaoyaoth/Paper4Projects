#ifndef SOCIAL_FORCE_CUH
#define SOCIAL_FORCE_CUH

#include "gsimcore.cuh"
#include "gsimvisual.cuh"

#define DIST(ax, ay, bx, by) sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by))

class SocialForceModel;
class SocialForceAgent;
class SocialForceClone;

typedef struct SocialForceAgentData : public GAgentData_t {
	float2 goal;
	float2 velocity;
	float v0;
	float mass;
	//for debug
	int id;
	int neibCount;
	__device__ void putDataInSmem(GAgent *ag);
};

struct obstacleLine
{
	float sx;
	float sy;
	float ex;
	float ey;

	__host__ void init(float sxx, float syy, float exx, float eyy)
	{
		sx = sxx;
		sy = syy;
		ex = exx;
		ey = eyy;
	}

	__device__ float pointToLineDist(float2 loc) 
	{
		float a, b;
		return this->pointToLineDist(loc, a, b);
	}

	__device__ float pointToLineDist(float2 loc, float &crx, float &cry) 
	{
		float d = DIST(sx, sy, ex, ey);
		float t0 = ((ex - sx) * (loc.x - sx) + (ey - sy) * (loc.y - sy)) / (d * d);

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
		return d;
	}

	__device__ int intersection2LineSeg(float p0x, float p0y, float p1x, float p1y, float &ix, float &iy)
	{
		float s1x, s1y, s2x, s2y;
		s1x = p1x - p0x;
		s1y = p1y - p0y;
		s2x = ex - sx;
		s2y = ey - sy;

		float s, t;
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

#define NUM_CLONE 1
#define NUM_WALLS 8
#define CHOSEN_CLONE_ID NUM_CLONE
__constant__ int numAgent;
__constant__ obstacleLine holeB;
__constant__ obstacleLine walls[NUM_WALLS];

__device__ uint throughput;
int throughputHost;
//std::fstream fout;
//char *outfname;
#ifdef _DEBUG
	SocialForceAgentData *dataHost;
	SocialForceAgentData *dataCopyHost;
	int *dataIdxArrayHost;
#endif

#define GATE_LINE_NUM 2
#define LEFT_GATE_SIZE 2
#define RIGHT_GATE_SIZE_A 2
#define RIGHT_GATE_SIZE_B 4
#define RIGHT_GATE_SIZE_C 6

#define MONITOR_STEP 41
#define CLONE
#define CLONE_COMPARE
#define CLONE_PERCENT 0.5

__global__ void addAgentsOnDevice(SocialForceModel *sfModel);

__global__ void replaceOriginalWithClone(GAgent **originalAgents, SocialForceAgent **clonedAgents, int numClonedAgent);

__global__ void cloneKernel(SocialForceAgent **originalAgents, AgentPool<SocialForceAgent, SocialForceAgentData> *pool, int numAgentLocal, GWorld *world, int cloneid);

__global__ void compareOriginAndClone(AgentPool<SocialForceAgent, SocialForceAgentData> *clonedPool, GWorld *clonedWorld, int numClonedAgents, int cloneid) ;

#ifdef CLONE
class SocialForceClone {
private:
	static int cloneCount;
public:
	AgentPool<SocialForceAgent, SocialForceAgentData> *agents, *agentsHost;
	GWorld *clonedWorld, *clonedWorldHost;
	SocialForceAgent **agentPtrArrayUnsorted;
	int cloneid;

	__host__ SocialForceClone(int num) {
		agentsHost = new AgentPool<SocialForceAgent, SocialForceAgentData>(0, num, sizeof(SocialForceAgentData));
		util::hostAllocCopyToDevice<AgentPool<SocialForceAgent, SocialForceAgentData> >(agentsHost, &agents);
		
		clonedWorldHost = new GWorld();
		util::hostAllocCopyToDevice<GWorld>(clonedWorldHost, &clonedWorld);

		//alloc untouched agent array
		cudaMalloc((void**)&agentPtrArrayUnsorted, num * sizeof(SocialForceAgent*) );

		cloneid = cloneCount++;
	}

	__host__ void start(SocialForceAgent** originalAgents, int num) {
		//copy the init agents to untouched agent list
		cudaMemcpy(agentPtrArrayUnsorted, originalAgents, num * sizeof(SocialForceAgent*), cudaMemcpyDeviceToDevice);
	}

	__host__ void step(SocialForceAgent** originalAgents, int num, SocialForceModel *modelHost);
};
#endif
class SocialForceModel : public GModel {
public:
	GRandom *random, *randomHost;
	cudaEvent_t timerStart, timerStop;

	AgentPool<SocialForceAgent, SocialForceAgentData> *originalAgents, *originalAgentsHost;

	std::fstream fout;
#ifdef CLONE
	SocialForceClone *clone1;
	//SocialForceClone *clone2;
#endif

	__host__ SocialForceModel(char **modelArgs) {
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

		cudaMemcpyToSymbol(numAgent, &num, sizeof(int));

		//init obstacles
		obstacleLine gateHost[NUM_WALLS], holeHost;
		gateHost[0].init(0.25 * modelHostParams.WIDTH, -20, 0.25 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT - LEFT_GATE_SIZE);
		gateHost[1].init(0.25 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT + LEFT_GATE_SIZE, 0.25 * modelHostParams.WIDTH, modelHostParams.HEIGHT + 20);

		gateHost[2].init(0.75 * modelHostParams.WIDTH, -20, 0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT - RIGHT_GATE_SIZE_A);
		gateHost[3].init(0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT + RIGHT_GATE_SIZE_A, 0.75 * modelHostParams.WIDTH, modelHostParams.HEIGHT + 20);

		gateHost[4].init(0.75 * modelHostParams.WIDTH, -20, 0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT - RIGHT_GATE_SIZE_B);
		gateHost[5].init(0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT + RIGHT_GATE_SIZE_B, 0.75 * modelHostParams.WIDTH, modelHostParams.HEIGHT + 20);
		gateHost[6].init(0.75 * modelHostParams.WIDTH, -20, 0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT - RIGHT_GATE_SIZE_C);
		gateHost[7].init(0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT + RIGHT_GATE_SIZE_C, 0.75 * modelHostParams.WIDTH, modelHostParams.HEIGHT + 20);

		cudaMemcpyToSymbol(walls, &gateHost, NUM_WALLS * sizeof(obstacleLine));

		holeHost.init(	0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT - RIGHT_GATE_SIZE_C,
			0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT + RIGHT_GATE_SIZE_C);
		cudaMemcpyToSymbol(holeB, &holeHost, sizeof(obstacleLine));

		//init agent pool
		originalAgentsHost = new AgentPool<SocialForceAgent, SocialForceAgentData>(num, modelHostParams.MAX_AGENT_NO, sizeof(SocialForceAgentData));
		util::hostAllocCopyToDevice<AgentPool<SocialForceAgent, SocialForceAgentData> >(originalAgentsHost, &originalAgents);

		//init world
		worldHost = new GWorld();
		util::hostAllocCopyToDevice<GWorld>(worldHost, &world);

		//init utility
		randomHost = new GRandom(modelHostParams.MAX_AGENT_NO);
		util::hostAllocCopyToDevice<GRandom>(randomHost, &random);

		util::hostAllocCopyToDevice<SocialForceModel>(this, (SocialForceModel**)&this->model);
	}

	__host__ void start()
	{
		int numAgentLocal = this->originalAgentsHost->numElem;
#ifdef CLONE
		//init clone
		clone1 = new SocialForceClone(numAgentLocal);
		//clone2 = new SocialForceClone(numAgentLocal);
#endif

		//debug info
#if defined(_DEBUG)
		//alloc debug output
		dataHost = (SocialForceAgentData*)malloc(sizeof(SocialForceAgentData) * numAgentLocal);
		dataCopyHost = (SocialForceAgentData*)malloc(sizeof(SocialForceAgentData) * numAgentLocal);
		dataIdxArrayHost = new int[numAgentLocal];
#else
		throughputHost = 0;
		cudaMemcpyToSymbol(throughput, &throughputHost, sizeof(int));
#endif

		//add original agents
		int gSize = GRID_SIZE(numAgentLocal);
		addAgentsOnDevice<<<gSize, BLOCK_SIZE>>>((SocialForceModel*)this->model);

		//initialize unsorted agent array in clones with original agents
#ifdef CLONE
		this->clone1->start(this->originalAgentsHost->agentPtrArray, numAgentLocal);
		//this->clone2->start(this->originalAgentsHost->agentPtrArray, numAgentLocal);
#endif

		//paint related
#ifdef _WIN32
		GSimVisual::getInstance().setWorld(this->world);
#endif
		//timer related
		cudaEventCreate(&timerStart);
		cudaEventCreate(&timerStop);
		cudaEventRecord(timerStart, 0);
	}

	__host__ void preStep()
	{
		//switch world
		int numInst = NUM_CLONE + 1;
#ifdef _WIN32
		if (GSimVisual::clicks % numInst == 0)
			GSimVisual::getInstance().setWorld(this->world);
#ifdef CLONE
		else if (GSimVisual:: clicks % numInst == 1)
			GSimVisual::getInstance().setWorld(this->clone1->clonedWorld);
		//else if (GSimVisual:: clicks % numInst == 2)
			//GSimVisual::getInstance().setWorld(this->clone2->clonedWorld);
#endif
#endif
		getLastCudaError("copyHostToDevice");
	}

	__host__ void step()
	{
		int numAgentLocal = this->originalAgentsHost->numElem;
		//1. run the original copy
		this->originalAgentsHost->registerPool(this->worldHost, this->schedulerHost, this->originalAgents);
		util::genNeighbor(this->world, this->worldHost, this->originalAgentsHost->numElem);
		cudaMemcpyToSymbol(modelDevParams, &modelHostParams, sizeof(modelConstants));
		this->originalAgentsHost->stepPoolAgent(this->model);

		//2. run the clones
#ifdef CLONE
		clone1->step(this->originalAgentsHost->agentPtrArray, numAgentLocal, this);
		//clone2->step(this->originalAgentsHost->agentPtrArray, numAgentLocal, this);
#endif

		//debug info, print the real data of original agents and cloned agents, or throughputs
#if defined(_DEBUG)
			//if (stepCountHost != MONITOR_STEP)
			//goto SKIP_DEBUG_OUT_OF_AGENT_ARRAY;
			fout<<"step:"<<stepCountHost<<std::endl;
			int numElemHost = this->originalAgentsHost->numElem;
			std::cout<<stepCountHost<<" "<<numElemHost;

			if (stepCountHost % 2 == 0)
				cudaMemcpy(dataHost, this->originalAgentsHost->dataCopyArray, sizeof(SocialForceAgentData) * numAgentLocal, cudaMemcpyDeviceToHost);
			else
				cudaMemcpy(dataHost, this->originalAgentsHost->dataArray, sizeof(SocialForceAgentData) * numAgentLocal, cudaMemcpyDeviceToHost);

			cudaMemcpy(dataIdxArrayHost, this->originalAgentsHost->dataIdxArray, sizeof(int) * numAgentLocal, cudaMemcpyDeviceToHost);

			for(int i = 0; i < numElemHost; i ++) {
				int dataIdx = dataIdxArrayHost[i];
				fout << dataHost[dataIdx].id
					<< "\t" << dataHost[dataIdx].neibCount 
					<< "\t" << dataHost[dataIdx].loc.x 
					<< "\t" << dataHost[dataIdx].loc.y 
					<< "\t"	<< dataHost[dataIdx].velocity.x 
					<< "\t" << dataHost[dataIdx].velocity.y 
					<< "\t" << std::endl;
				fout.flush();
			}
			fout <<"-------------------"<<std::endl;

#if defined(CLONE)
			numElemHost = this->clone1->agentsHost->numElem;
			std::cout<<" "<<numElemHost<<std::endl;
			if (stepCountHost % 2 == 0) {
				cudaMemcpy(dataHost, this->clone1->agentsHost->dataCopyArray, sizeof(SocialForceAgentData) * numAgentLocal, cudaMemcpyDeviceToHost);
				cudaMemcpy(dataCopyHost, this->clone1->agentsHost->dataArray, sizeof(SocialForceAgentData) * numAgentLocal, cudaMemcpyDeviceToHost);
			} else {
				cudaMemcpy(dataHost, this->clone1->agentsHost->dataArray, sizeof(SocialForceAgentData) * numAgentLocal, cudaMemcpyDeviceToHost);
				cudaMemcpy(dataCopyHost, this->clone1->agentsHost->dataCopyArray, sizeof(SocialForceAgentData) * numAgentLocal, cudaMemcpyDeviceToHost);
			}

			cudaMemcpy(dataIdxArrayHost, this->clone1->agentsHost->dataIdxArray, sizeof(int) * numAgentLocal, cudaMemcpyDeviceToHost);

			for(int i = 0; i < numElemHost; i ++) {
				int dataIdx = dataIdxArrayHost[i];
				fout << dataHost[dataIdx].id
					<< "\t" << dataHost[dataIdx].neibCount 
					<< "\t" << dataHost[dataIdx].loc.x 
					<< "\t" << dataHost[dataIdx].loc.y
					<< "\t"	<< dataHost[dataIdx].velocity.x 
					<< "\t" << dataHost[dataIdx].velocity.y 
					<< "\t" << std::endl;
				fout.flush();
			}
			//fout <<"-------------------"<<std::endl;
			//for(int i = 0; i < numElemHost; i ++) {
			//	int dataIdx = dataIdxArrayHost[i];
			//	fout << dataCopyHost[dataIdx].id
			//		<< "\t" << dataCopyHost[dataIdx].loc.x 
			//		<< "\t" << dataCopyHost[dataIdx].loc.y
			//		<< "\t"	<< dataCopyHost[dataIdx].velocity.x 
			//		<< "\t" << dataCopyHost[dataIdx].velocity.y 
			//		<< "\t" << std::endl;
			//	fout.flush();
			//}
			fout <<"==================="<<std::endl<<std::endl;
			fout.flush();
#endif

#else
		cudaMemcpyFromSymbol(&throughputHost, throughput, sizeof(int));
		fout<<throughputHost<<std::endl;
		fout.flush();
#endif

		//5. swap data and dataCopy
		this->originalAgentsHost->swapPool();
#ifdef CLONE
		this->clone1->agentsHost->swapPool();
		//this->clone2->agentsHost->swapPool();
#endif
		getLastCudaError("step");

		//paint related stuff
#ifdef _WIN32
		GSimVisual::getInstance().animate();
#endif

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
int SocialForceClone::cloneCount = 1;
__host__ void SocialForceClone::step(SocialForceAgent** originalAgents, int num, SocialForceModel *modelHost) {

	//1.1 clone agents
	int numAgentLocal = num;
	int gSize = GRID_SIZE(numAgentLocal);
	cloneKernel<<<gSize, BLOCK_SIZE>>>(originalAgents, agents, numAgentLocal, clonedWorld, cloneid);

	//2. run the cloned copy
	//2.1. register the cloned agents to the c1loned world
	cudaMemcpy(clonedWorldHost->allAgents,
		agentPtrArrayUnsorted, 
		num * sizeof(void*), 
		cudaMemcpyDeviceToDevice);

	this->agentsHost->cleanup(this->agents);
	int numAgentsB = this->agentsHost->numElem;
	if (numAgentsB != 0) {
		gSize = GRID_SIZE(numAgentsB);

		replaceOriginalWithClone<<<gSize, BLOCK_SIZE>>>(
			clonedWorldHost->allAgents, 
			this->agentsHost->agentPtrArray, 
			numAgentsB);

		//2.2. sort world and worldClone
		util::genNeighbor(this->clonedWorld, this->clonedWorldHost, modelHostParams.AGENT_NO);

		//2.3. step the cloned copy
		this->agentsHost->stepPoolAgent(modelHost->model);

#ifdef CLONE_COMPARE
		//3. double check
		compareOriginAndClone<<<gSize, BLOCK_SIZE>>>(this->agents, clonedWorld, numAgentsB, cloneid);

		//4. clean pool again, since some agents are removed
		this->agentsHost->cleanup(this->agents);
#endif

		getLastCudaError("step:clone");
	}

}
#endif

__device__ float correctCrossBoader(float val, float limit)
{
	if (val > limit)
		return limit-0.001;
	else if (val < 0)
		return 0;
	return val;
}

class SocialForceAgent : public GAgent {
public:
	GRandom *random;
	GWorld *myWorld;

	int id;
#ifdef CLONE
	int cloneid;
	bool cloned[NUM_CLONE];
	bool cloning[NUM_CLONE];
#endif

	SocialForceAgent *myOrigin;
	obstacleLine *myWall;
	float gateSize;


	__device__ void computeIndivSocialForce(const SocialForceAgentData &myData, const SocialForceAgentData &otherData, float2 &fSum){
		float cMass = 100;
		//my data
		const float2& loc = myData.loc;
		const float2& goal = myData.goal;
		const float2& velo = myData.velocity;
		const float& v0 = myData.v0;
		const float& mass = myData.mass;
		//other's data
		const float2& locOther = otherData.loc;
		const float2& goalOther = otherData.goal;
		const float2& veloOther = otherData.velocity;
		const float& v0Other = otherData.v0;
		const float& massOther = otherData.mass;

		float d = 1e-15 + sqrt((loc.x - locOther.x) * (loc.x - locOther.x) + (loc.y - locOther.y) * (loc.y - locOther.y));
		float dDelta = mass / cMass + massOther / cMass - d;
		float fExp = A * exp(dDelta / B);
		float fKg = dDelta < 0 ? 0 : k1 *dDelta;
		float nijx = (loc.x - locOther.x) / d;
		float nijy = (loc.y - locOther.y) / d;
		float fnijx = (fExp + fKg) * nijx;
		float fnijy = (fExp + fKg) * nijy;
		float fkgx = 0;
		float fkgy = 0;
		if (dDelta > 0) {
			float tix = - nijy;
			float tiy = nijx;
			fkgx = k2 * dDelta;
			fkgy = k2 * dDelta;
			float vijDelta = (veloOther.x - velo.x) * tix + (veloOther.y - velo.y) * tiy;
			fkgx = fkgx * vijDelta * tix;
			fkgy = fkgy * vijDelta * tiy;
		}
		fSum.x += fnijx + fkgx;
		fSum.y += fnijy + fkgy;
	}
	__device__ void computeForceWithWall(const SocialForceAgentData &dataLocal, obstacleLine &wall, const int &cMass, float2 &fSum) {
		float diw, crx, cry;
		const float2 &loc = dataLocal.loc;
		diw = wall.pointToLineDist(loc, crx, cry);
		float virDiw = DIST(loc.x, loc.y, crx, cry);
		float niwx = (loc.x - crx) / virDiw;
		float niwy = (loc.y - cry) / virDiw;
		float drw = dataLocal.mass / cMass - diw;
		float fiw1 = A * exp(drw / B);
		if (drw > 0)
			fiw1 += k1 * drw;
		float fniwx = fiw1 * niwx;
		float fniwy = fiw1 * niwy;

		float fiwKgx = 0, fiwKgy = 0;
		if (drw > 0)
		{
			float fiwKg = k2 * drw * (dataLocal.velocity.x * (-niwy) + dataLocal.velocity.y * niwx);
			fiwKgx = fiwKg * (-niwy);
			fiwKgy = fiwKg * niwx;
		}

		fSum.x += fniwx - fiwKgx;
		fSum.y += fniwy - fiwKgy;
	}
	__device__ void computeWallImpaction(const SocialForceAgentData &dataLocal, obstacleLine &wall, const float2 &newVelo, const float &tick, float &mint){
		float crx, cry, tt;
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
	__device__ void computeDirection(const SocialForceAgentData &dataLocal, float2 &dvt) {
		//my data
		const float2& loc = dataLocal.loc;
		const float2& goal = dataLocal.goal;
		const float2& velo = dataLocal.velocity;
		const float& v0 = dataLocal.v0;
		const float& mass = dataLocal.mass;
		
		dvt.x = 0;	dvt.y = 0;
		float2 diff; diff.x = 0; diff.y = 0;
		float d0 = sqrt((loc.x - goal.x) * (loc.x - goal.x) + (loc.y - goal.y) * (loc.y - goal.y));
		diff.x = v0 * (goal.x - loc.x) / d0;
		diff.y = v0 * (goal.y - loc.y) / d0;
		dvt.x = (diff.x - velo.x) / tao;
		dvt.y = (diff.y - velo.y) / tao;
	}
	__device__ void computeSocialForce(SocialForceAgentData &dataLocal, float2 &fSum) {
		GWorld *world = this->myWorld;
		iterInfo info;

		fSum.x = 0; fSum.y = 0;
		SocialForceAgentData *otherData, otherDataLocal;
		float ds = 0;

		int neighborCount = 0;

		world->neighborQueryInit(dataLocal.loc, 6, info);
		otherData = world->nextAgentDataFromSharedMem<SocialForceAgentData>(info);
		while (otherData != NULL) {
			otherDataLocal = *otherData;
			SocialForceAgent *otherPtr = (SocialForceAgent*)otherData->agentPtr;
			ds = length(otherDataLocal.loc - dataLocal.loc);
			if (ds < 6 && ds > 0 ) {
				neighborCount++;
				computeIndivSocialForce(dataLocal, otherDataLocal, fSum);
#ifdef CLONE
				for (int i = 0; i < NUM_CLONE; i++)
					if (otherPtr->cloned[i] == true) // decision point B: impaction from neighboring agent
						this->cloning[i] = true;
#endif
			}
			otherData = world->nextAgentDataFromSharedMem<SocialForceAgentData>(info);
		}
		dataLocal.neibCount = neighborCount;
	}

	__device__ void step(GModel *model){
		SocialForceModel *sfModel = (SocialForceModel*)model;
		float cMass = 100;

		SocialForceAgentData dataLocal = *(SocialForceAgentData*)this->data;

		const float2& loc = dataLocal.loc;
		const float2& goal = dataLocal.goal;
		const float2& velo = dataLocal.velocity;
		const float& v0 = dataLocal.v0;
		const float& mass = dataLocal.mass;

		//compute the direction
		float2 dvt;
		computeDirection(dataLocal, dvt);

		//compute force with other agents
		float2 fSum; 
		computeSocialForce(dataLocal, fSum);

		//compute force with wall
		computeForceWithWall(dataLocal, myWall[0], cMass, fSum);
		computeForceWithWall(dataLocal, myWall[1], cMass, fSum);

#ifdef CLONE
		//decision point A: impaction from wall
		if(holeB.pointToLineDist(loc) < 20) {
			for (int i = 0; i < NUM_CLONE; i++)
				if (this->cloned[i] == false)
					this->cloning[i] = true;
		}
#endif

		//sum up
		dvt.x += fSum.x / mass;
		dvt.y += fSum.y / mass;

		float2 newVelo = dataLocal.velocity;
		float2 newLoc = dataLocal.loc;
		float2 newGoal = dataLocal.goal;

		float tick = 0.1;
		newVelo.x += dvt.x * tick * (1);// + this->random->gaussian() * 0.1);
		newVelo.y += dvt.y * tick * (1);// + this->random->gaussian() * 0.1);
		float dv = sqrt(newVelo.x * newVelo.x + newVelo.y * newVelo.y);

		if (dv > maxv) {
			newVelo.x = newVelo.x * maxv / dv;
			newVelo.y = newVelo.y * maxv / dv;
		}

		float mint = 1;
		computeWallImpaction(dataLocal, myWall[0], newVelo, tick, mint);
		computeWallImpaction(dataLocal, myWall[1], newVelo, tick, mint);

		newVelo.x *= mint;
		newVelo.y *= mint;
		newLoc.x += newVelo.x * tick;
		newLoc.y += newVelo.y * tick;

		float goalTemp = goal.x;

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
#ifdef CLONE
			if (goalTemp != newGoal.x && this->cloneid == CHOSEN_CLONE_ID) 
#else
			if (goalTemp != newGoal.x) 	
#endif
				atomicInc(&throughput, 8192);
		}

		newLoc.x = correctCrossBoader(newLoc.x, modelDevParams.WIDTH);
		newLoc.y = correctCrossBoader(newLoc.y, modelDevParams.HEIGHT);

		SocialForceAgentData dataCopy = dataLocal;
		dataCopy.loc = newLoc;
		dataCopy.velocity = newVelo;
		dataCopy.goal = newGoal;

		*(SocialForceAgentData*)this->dataCopy = dataCopy;
	}

	__device__ void fillSharedMem(void *dataPtr){
		SocialForceAgentData *dataSmem = (SocialForceAgentData*)dataPtr;
		SocialForceAgentData *dataAgent = (SocialForceAgentData*)this->data;
		*dataSmem = *dataAgent;
	}

	__device__ void init(SocialForceModel *sfModel, int dataSlot) {
		this->myWorld = sfModel->world;
#ifdef NDEBUG
		this->random = sfModel->random;
#endif
		this->color = colorConfigs.green;
		this->id = dataSlot;

#ifdef CLONE
		this->cloneid = 0;
		for (int i = 0; i < NUM_CLONE; i++) {
			this->cloned[i] = false;
			this->cloning[i] = false;
		}
#endif
		this->myOrigin = NULL;

		SocialForceAgentData dataLocal; //= &sfModel->originalAgents->dataArray[dataSlot];

		dataLocal.agentPtr = this;
		dataLocal.id = dataSlot;
#ifdef NDEBUG
		dataLocal.loc.x = (0.25 + 0.5 * this->random->uniform()) * modelDevParams.WIDTH - 0.1;
		dataLocal.loc.y = this->random->uniform() * modelDevParams.HEIGHT;
#else
		float sqrtNumAgent = sqrt((float)numAgent);
		float x = (float)(dataSlot % (int)sqrtNumAgent) / sqrtNumAgent;
		float y = (float)(dataSlot / (int)sqrtNumAgent) / sqrtNumAgent;
		dataLocal.loc.x = (0.3 + x * 0.4) * modelDevParams.WIDTH;
		dataLocal.loc.y = y * modelDevParams.HEIGHT;
#endif
		dataLocal.velocity.x = 2;//4 * (this->random->uniform()-0.5);
		dataLocal.velocity.y = 2;//4 * (this->random->uniform()-0.5);

		dataLocal.v0 = 2;
		dataLocal.mass = 50;


		float2 goal1 = make_float2(0.25 * modelDevParams.WIDTH, 0.50 * modelDevParams.HEIGHT);
		float2 goal2 = make_float2(0.75 * modelDevParams.WIDTH, 0.50 * modelDevParams.HEIGHT);
#ifdef NDEBUG
		if(dataLocal.loc.x < (0.75 - 0.5 * CLONE_PERCENT) * modelDevParams.WIDTH) {
			dataLocal.goal = goal1;
			myWall = &walls[0];
			gateSize = LEFT_GATE_SIZE;
		} else 
#endif
		{
			dataLocal.goal = goal2;
			myWall = &walls[2];
			gateSize = RIGHT_GATE_SIZE_A;
		}

		this->data = sfModel->originalAgents->dataInSlot(dataSlot);
		this->dataCopy = sfModel->originalAgents->dataCopyInSlot(dataSlot);
		*(SocialForceAgentData*)this->data = dataLocal;
		*(SocialForceAgentData*)this->dataCopy = dataLocal;
	}

#ifdef CLONE
	__device__ void initNewClone(const SocialForceAgent &agent, 
		SocialForceAgent *originPtr, 
		int dataSlot, 
		GWorld *clonedWorld,
		AgentPool<SocialForceAgent, SocialForceAgentData> *pool,
		int cloneid) 
	{
		this->myWorld = clonedWorld;
		if (cloneid == 1) {
			this->color = colorConfigs.red;
			this->gateSize = RIGHT_GATE_SIZE_B;
		} else if (cloneid == 2) {
			this->color = colorConfigs.yellow;
			this->gateSize = RIGHT_GATE_SIZE_C;
		}
		
		this->cloneid = cloneid;
		this->id = agent.id;
		for (int i = 0; i < NUM_CLONE; i++){
			this->cloned[i] = false;
			this->cloning[i] = false;
		}
		this->myOrigin = originPtr;
		this->myWall = &walls[cloneid * 2 + 2];

		SocialForceAgentData dataLocal, dataCopyLocal;

		dataLocal = *(SocialForceAgentData*)agent.data;
		dataCopyLocal = *(SocialForceAgentData*)agent.dataCopy;

		dataLocal.agentPtr = this;
		dataCopyLocal.agentPtr = this;

		if (stepCount % 2 == 0) {
			this->data = pool->dataInSlot(dataSlot);
			this->dataCopy = pool->dataCopyInSlot(dataSlot);
		} else {
			this->data = pool->dataCopyInSlot(dataSlot);
			this->dataCopy = pool->dataInSlot(dataSlot);
		}

		*(SocialForceAgentData*)this->data = dataLocal;
		*(SocialForceAgentData*)this->dataCopy = dataCopyLocal;
	}
#endif
};

__device__ void SocialForceAgentData::putDataInSmem(GAgent *ag){
	*this = *(SocialForceAgentData*)ag->data;
}

__global__ void addAgentsOnDevice(SocialForceModel *sfModel){
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAgent){ // user init step
		//Add agent here
		int dataSlot = idx;
		SocialForceAgent *ag = sfModel->originalAgents->agentInSlot(dataSlot);
		ag->init(sfModel, dataSlot);
		sfModel->originalAgents->add(ag, dataSlot);

	}
}

#ifdef CLONE
__global__ void cloneKernel(SocialForceAgent **originalAgents, AgentPool<SocialForceAgent, SocialForceAgentData> *pool, int numAgentLocal, GWorld *world, int cloneid)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAgentLocal){ // user init step
		SocialForceAgent ag, *agPtr = originalAgents[idx];
		ag = *agPtr;

		int cloneidLocal = cloneid - 1;
		if( ag.cloning[cloneidLocal] == true && ag.cloned[cloneidLocal] == false) {
			ag.cloned[cloneidLocal] = true;
			ag.cloning[cloneidLocal] = false;
			*agPtr = ag;

			int agentSlot = pool->agentSlot();
			int dataSlot = pool->dataSlot(agentSlot);

			SocialForceAgent *ag2 = pool->agentInSlot(dataSlot);
			ag2->initNewClone(ag, agPtr, dataSlot, world, pool, cloneid);
			pool->add(ag2, agentSlot);
		}
	}
}

__global__ void replaceOriginalWithClone(GAgent **originalAgents, SocialForceAgent **clonedAgents, int numClonedAgent)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numClonedAgent){
		SocialForceAgent *ag = clonedAgents[idx];
		originalAgents[ag->id] = ag;
	}
}

__global__ void compareOriginAndClone(
	AgentPool<SocialForceAgent, SocialForceAgentData> *clonedPool,
	GWorld *clonedWorld,
	int numClonedAgents,
	int cloneid) 
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numClonedAgents) {
		SocialForceAgent *clonedAg = clonedPool->agentPtrArray[idx];
		SocialForceAgent *originalAg = clonedAg->myOrigin;
		SocialForceAgentData clonedAgData = *(SocialForceAgentData*) clonedAg->dataCopy;
		SocialForceAgentData originalAgData = *(SocialForceAgentData*) originalAg->dataCopy;

		bool match;
		//compare equivalence of two copies of data;
#define DELTA FLT_EPSILON
		float diffLocX = clonedAgData.loc.x - originalAgData.loc.x;
		float diffLocY = clonedAgData.loc.y - originalAgData.loc.y;
		float diffVelX = clonedAgData.velocity.x - originalAgData.velocity.x;
		float diffVelY = clonedAgData.velocity.y - originalAgData.velocity.y;
		match = (diffLocX * diffLocX <= DELTA)
			&& (diffLocY * diffLocY <= DELTA)
			&& (diffVelX * diffVelX <= DELTA)
			&& (diffVelY * diffVelY <= DELTA); 
			//&& (clonedAgData.goal.x - originalAgData.goal.x == DELTA)
			//&& (clonedAgData.goal.y - originalAgData.goal.y == DELTA);
		if (match) {
			//remove from cloned set, reset clone state to non-cloned
			clonedPool->remove(idx);
			int cloneidLocal = cloneid - 1;
			originalAg->cloned[cloneidLocal] = false;
			originalAg->cloning[cloneidLocal] = false;
		}
#ifdef _DEBUG
		else {
			originalAg->color = colorConfigs.blue;
			printf("step: %d\n", stepCount);
			printf("\t origin:%d, clone:%d", originalAgData.id, clonedAgData.id);
			printf("\t diffLocSqr:[%f, %f]", diffLocX * diffLocX, diffLocY * diffLocY);
			printf("\t diffVelSqr:[%f, %f]", diffVelX * diffVelX, diffVelY * diffVelY);
		}
#endif
	}
}
#endif

#endif