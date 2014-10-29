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
	int id;
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

__constant__ int numAgent;
int numAgentHost;

__constant__ obstacleLine gateOne[2];
__constant__ obstacleLine gateTwoA[2];
__constant__ obstacleLine gateTwoB[2];
__constant__ obstacleLine holeB;

__device__ uint throughput;
int throughputHost;
//std::fstream fout;
//char *outfname;

#define GATE_LINE_NUM 2
#define LEFT_GATE_SIZE 2
#define RIGHT_GATE_SIZE_A 2
#define RIGHT_GATE_SIZE_B 4

#define MONITOR_STEP 41
#define CLONE
#define CLONE_COMPARE
#define CLONE_PERCENT 0.5

__global__ void addAgentsOnDevice(SocialForceModel *sfModel);

__global__ void replaceOriginalWithClone(
	GAgent **originalAgents, 
	SocialForceAgent **clonedAgents, 
	int numClonedAgent);

__global__ void cloneKernel(SocialForceModel *sfModel, 
							int numAgentLocal);

__global__ void compareOriginAndClone(
	AgentPool<SocialForceAgent, SocialForceAgentData> *clonedPool,
	GWorld *clonedWorld,
	int numClonedAgents) ;

class SocialForceModel : public GModel {
public:
	GRandom *random, *randomHost;
	cudaEvent_t timerStart, timerStop;

	AgentPool<SocialForceAgent, SocialForceAgentData> *agentsA, *agentsAHost;

#ifdef CLONE
	AgentPool<SocialForceAgent, SocialForceAgentData> *agentsB, *agentsBHost;
	GWorld *clonedWorld, *clonedWorldHost;
	SocialForceAgent **agentPtrArrayUnsorted;
#endif

#ifdef _DEBUG
	SocialForceAgentData *dataHost;
	SocialForceAgentData *dataCopyHost;
	int *dataIdxArrayHost;
#endif

	std::fstream fout;

	__host__ SocialForceModel(char **modelArgs) {
		int num = atoi(modelArgs[0]);
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

		numAgentHost = num;
		cudaMemcpyToSymbol(numAgent, &num, sizeof(int));

		//init obstacles
		obstacleLine gateHost[2], holeHost;
		gateHost[0].init(0.25 * modelHostParams.WIDTH, -20, 0.25 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT - LEFT_GATE_SIZE);
		gateHost[1].init(0.25 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT + LEFT_GATE_SIZE, 0.25 * modelHostParams.WIDTH, modelHostParams.HEIGHT + 20);
		cudaMemcpyToSymbol(gateOne, &gateHost, 2 * sizeof(obstacleLine));

		gateHost[0].init(0.75 * modelHostParams.WIDTH, -20, 0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT - RIGHT_GATE_SIZE_A);
		gateHost[1].init(0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT + RIGHT_GATE_SIZE_A, 0.75 * modelHostParams.WIDTH, modelHostParams.HEIGHT + 20);
		cudaMemcpyToSymbol(gateTwoA, &gateHost, 2 * sizeof(obstacleLine));

		gateHost[0].init(0.75 * modelHostParams.WIDTH, -20, 0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT - RIGHT_GATE_SIZE_B);
		gateHost[1].init(0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT + RIGHT_GATE_SIZE_B, 0.75 * modelHostParams.WIDTH, modelHostParams.HEIGHT + 20);
		cudaMemcpyToSymbol(gateTwoB, &gateHost, 2 * sizeof(obstacleLine));

		holeHost.init(	0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT - RIGHT_GATE_SIZE_B,
			0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT + RIGHT_GATE_SIZE_B);
		cudaMemcpyToSymbol(holeB, &holeHost, sizeof(obstacleLine));

		//init agent pool
		agentsAHost = new AgentPool<SocialForceAgent, SocialForceAgentData>(numAgentHost, numAgentHost, sizeof(SocialForceAgentData));
		util::hostAllocCopyToDevice<AgentPool<SocialForceAgent, SocialForceAgentData> >(agentsAHost, &agentsA);

#ifdef CLONE
		agentsBHost = new AgentPool<SocialForceAgent, SocialForceAgentData>(0, numAgentHost, sizeof(SocialForceAgentData));
		util::hostAllocCopyToDevice<AgentPool<SocialForceAgent, SocialForceAgentData> >(agentsBHost, &agentsB);
#endif
		//init world
		worldHost = new GWorld();
		util::hostAllocCopyToDevice<GWorld>(worldHost, &world);

#ifdef CLONE
		clonedWorldHost = new GWorld();
		util::hostAllocCopyToDevice<GWorld>(clonedWorldHost, &clonedWorld);

		//alloc untouched agent array
		cudaMalloc((void**)&agentPtrArrayUnsorted, num * sizeof(SocialForceAgent*) );
#endif

		//init utility
		randomHost = new GRandom(modelHostParams.MAX_AGENT_NO);
		util::hostAllocCopyToDevice<GRandom>(randomHost, &random);

		util::hostAllocCopyToDevice<SocialForceModel>(this, (SocialForceModel**)&this->model);
	}

	__host__ void start()
	{

#if defined(_DEBUG)
		//alloc debug output
		int numAgent = this->agentsAHost->numElem;
		dataHost = (SocialForceAgentData*)malloc(sizeof(SocialForceAgentData) * numAgent);
		dataCopyHost = (SocialForceAgentData*)malloc(sizeof(SocialForceAgentData) * numAgent);
		dataIdxArrayHost = new int[numAgent];
#else
		throughputHost = 0;
		cudaMemcpyToSymbol(throughput, &throughputHost, sizeof(int));
#endif

		int AGENT_NO = this->agentsAHost->numElem;
		int gSize = GRID_SIZE(AGENT_NO);
		addAgentsOnDevice<<<gSize, BLOCK_SIZE>>>((SocialForceModel*)this->model);
#ifdef CLONE
		//copy the init agents to untouched agent list
		cudaMemcpy(agentPtrArrayUnsorted, 
			this->agentsAHost->agentPtrArray, 
			AGENT_NO * sizeof(SocialForceAgent*), 
			cudaMemcpyDeviceToDevice);
#endif

#ifdef _WIN32
		GSimVisual::getInstance().setWorld(this->world);
#endif
		cudaEventCreate(&timerStart);
		cudaEventCreate(&timerStop);
		cudaEventRecord(timerStart, 0);
	}

	__host__ void preStep()
	{

#ifdef _WIN32
		if (GSimVisual::clicks % 2 == 0)
			GSimVisual::getInstance().setWorld(this->world);
#ifdef CLONE
		else
			GSimVisual::getInstance().setWorld(this->clonedWorld);
#endif
#endif
		getLastCudaError("copyHostToDevice");
	}

	__host__ void step()
	{
		//1. run the original copy
		this->agentsAHost->registerPool(this->worldHost, this->schedulerHost, this->agentsA);
		util::genNeighbor(this->world, this->worldHost, this->agentsAHost->numElem);
		cudaMemcpyToSymbol(modelDevParams, &modelHostParams, sizeof(modelConstants));
		this->agentsAHost->stepPoolAgent(this->model);

#ifdef CLONE	
		//1.1 clone agents
		int numAgentLocal = this->agentsAHost->numElem;
		int gSize = GRID_SIZE(numAgentLocal);
		cloneKernel<<<gSize, BLOCK_SIZE>>>((SocialForceModel*)this->model, numAgentLocal);

		//2. run the cloned copy
		//2.1. register the cloned agents to the c1loned world
		cudaMemcpy(clonedWorldHost->allAgents,
			agentPtrArrayUnsorted, 
			numAgentHost * sizeof(void*), 
			cudaMemcpyDeviceToDevice);

		this->agentsBHost->cleanup(this->agentsB);
		int numAgentsB = this->agentsBHost->numElem;
		if (numAgentsB != 0) {
			gSize = GRID_SIZE(numAgentsB);

			replaceOriginalWithClone<<<gSize, BLOCK_SIZE>>>(
				clonedWorldHost->allAgents, 
				this->agentsBHost->agentPtrArray, 
				numAgentsB);

			//2.2. sort world and worldClone
			util::genNeighbor(this->clonedWorld, this->clonedWorldHost, modelHostParams.AGENT_NO);

			//2.3. step the cloned copy
			this->agentsBHost->stepPoolAgent(this->model);

#ifdef CLONE_COMPARE
			//3. double check
			compareOriginAndClone<<<gSize, BLOCK_SIZE>>>(this->agentsB, clonedWorld, numAgentsB);

			//4. clean pool again, since some agents are removed
			this->agentsBHost->cleanup(this->agentsB);
#endif

#ifdef _DEBUG

			//4.1 demonstrate
			//numAgentsB = this->agentsBHost->numElem;
			//cudaMemcpy(clonedWorldHost->allAgents,
			//	agentPtrArrayUnsorted, 
			//	numAgentHost * sizeof(void*), 
			//	cudaMemcpyDeviceToDevice);

			//replaceOriginalWithClone<<<gSize, BLOCK_SIZE>>>(
			//	clonedWorldHost->allAgents, 
			//	this->agentsBHost->agentPtrArray, 
			//	numAgentsB);
#endif
		}

#endif

#if defined(_DEBUG)
			//if (stepCountHost != MONITOR_STEP)
			//goto SKIP_DEBUG_OUT_OF_AGENT_ARRAY;
			fout<<"step:"<<stepCountHost<<std::endl;
			int numElemMaxHost = this->agentsAHost->numElemMax;
			int numElemHost = this->agentsAHost->numElem;
			std::cout<<stepCountHost<<" "<<numElemHost;

			if (stepCountHost % 2 == 0)
				cudaMemcpy(dataHost, this->agentsAHost->dataCopyArray, sizeof(SocialForceAgentData) * numElemMaxHost, cudaMemcpyDeviceToHost);
			else
				cudaMemcpy(dataHost, this->agentsAHost->dataArray, sizeof(SocialForceAgentData) * numElemMaxHost, cudaMemcpyDeviceToHost);

			cudaMemcpy(dataIdxArrayHost, this->agentsAHost->dataIdxArray, sizeof(int) * numElemMaxHost, cudaMemcpyDeviceToHost);

			for(int i = 0; i < numElemHost; i ++) {
				int dataIdx = dataIdxArrayHost[i];
				fout << dataHost[dataIdx].id
					<< "\t" << dataHost[dataIdx].loc.x 
					<< "\t" << dataHost[dataIdx].loc.y 
					<< "\t"	<< dataHost[dataIdx].velocity.x 
					<< "\t" << dataHost[dataIdx].velocity.y 
					<< "\t" << std::endl;
				fout.flush();
			}
			fout <<"-------------------"<<std::endl;
#if defined(CLONE)
			numElemMaxHost = this->agentsBHost->numElemMax;
			numElemHost = this->agentsBHost->numElem;
			std::cout<<" "<<numElemHost<<std::endl;
			if (stepCountHost % 2 == 0) {
				cudaMemcpy(dataHost, this->agentsBHost->dataCopyArray, sizeof(SocialForceAgentData) * numElemMaxHost, cudaMemcpyDeviceToHost);
				cudaMemcpy(dataCopyHost, this->agentsBHost->dataArray, sizeof(SocialForceAgentData) * numElemMaxHost, cudaMemcpyDeviceToHost);
			} else {
				cudaMemcpy(dataHost, this->agentsBHost->dataArray, sizeof(SocialForceAgentData) * numElemMaxHost, cudaMemcpyDeviceToHost);
				cudaMemcpy(dataCopyHost, this->agentsBHost->dataCopyArray, sizeof(SocialForceAgentData) * numElemMaxHost, cudaMemcpyDeviceToHost);
			}

			cudaMemcpy(dataIdxArrayHost, this->agentsBHost->dataIdxArray, sizeof(int) * numElemMaxHost, cudaMemcpyDeviceToHost);

			for(int i = 0; i < numElemHost; i ++) {
				int dataIdx = dataIdxArrayHost[i];
				fout << dataHost[dataIdx].id
					<< "\t" << dataHost[dataIdx].loc.x 
					<< "\t" << dataHost[dataIdx].loc.y
					<< "\t"	<< dataHost[dataIdx].velocity.x 
					<< "\t" << dataHost[dataIdx].velocity.y 
					<< "\t" << std::endl;
				fout.flush();
			}
			fout <<"-------------------"<<std::endl;
			for(int i = 0; i < numElemHost; i ++) {
				int dataIdx = dataIdxArrayHost[i];
				fout << dataCopyHost[dataIdx].id
					<< "\t" << dataCopyHost[dataIdx].loc.x 
					<< "\t" << dataCopyHost[dataIdx].loc.y
					<< "\t"	<< dataCopyHost[dataIdx].velocity.x 
					<< "\t" << dataCopyHost[dataIdx].velocity.y 
					<< "\t" << std::endl;
				fout.flush();
			}
			fout <<"==================="<<std::endl<<std::endl;
			fout.flush();
#endif
#else
		cudaMemcpyFromSymbol(&throughputHost, throughput, sizeof(int));
		fout<<throughputHost<<std::endl;
		fout.flush();
#endif

		//5. swap data and dataCopy
		this->agentsAHost->swapPool();
#ifdef CLONE
		this->agentsBHost->swapPool();
#endif
		getLastCudaError("preStep");

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
	SocialForceModel *myModel;
	GWorld *myWorld;

	int id;
	uchar4 cloneid;
	bool cloned;
	bool cloning;
	SocialForceAgent *myOrigin;

	__device__ void computeSocialForce(const SocialForceAgentData &myData, const SocialForceAgentData &otherData, float2 &fSum){
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

	__device__ void step(GModel *model){
		SocialForceModel *sfModel = (SocialForceModel*)model;
		GWorld *world = this->myWorld;
		float width = world->width;
		float height = world->height;
		float cMass = 100;

		iterInfo info;

		SocialForceAgentData dataLocal = *(SocialForceAgentData*)this->data;

		const float2& loc = dataLocal.loc;
		const float2& goal = dataLocal.goal;
		const float2& velo = dataLocal.velocity;
		const float& v0 = dataLocal.v0;
		const float& mass = dataLocal.mass;

		//compute the direction
		float2 dvt;	dvt.x = 0;	dvt.y = 0;
		float2 diff; diff.x = 0; diff.y = 0;
		float d0 = sqrt((loc.x - goal.x) * (loc.x - goal.x) + (loc.y - goal.y) * (loc.y - goal.y));
		diff.x = v0 * (goal.x - loc.x) / d0;
		diff.y = v0 * (goal.y - loc.y) / d0;
		dvt.x = (diff.x - velo.x) / tao;
		dvt.y = (diff.y - velo.y) / tao;

		//compute force with other agents
		float2 fSum; fSum.x = 0; fSum.y = 0;
		SocialForceAgentData *otherData, otherDataLocal;
		float ds = 0;

		world->neighborQueryInit(loc, 6, info);
		otherData = world->nextAgentDataFromSharedMem<SocialForceAgentData>(info);
		while (otherData != NULL) {
			otherDataLocal = *otherData;
			SocialForceAgent *otherPtr = (SocialForceAgent*)otherData->agentPtr;
			bool otherCloned = otherPtr->cloned;
			ds = length(otherDataLocal.loc - loc);
			if (ds < 6 && ds > 0 ) {
				computeSocialForce(dataLocal, otherDataLocal, fSum);
				if (otherCloned == true) // decision point B: impaction from neighboring agent
					this->cloning = true;
			}
			otherData = world->nextAgentDataFromSharedMem<SocialForceAgentData>(info);
		}

		if (dataLocal.goal.x < 0.5 * modelDevParams.WIDTH) {
			computeForceWithWall(dataLocal, gateOne[0], cMass, fSum);
			computeForceWithWall(dataLocal, gateOne[1], cMass, fSum);
		}
#ifdef CLONE
		else if (this->cloneid.x == '0') {
			computeForceWithWall(dataLocal, gateTwoA[0], cMass, fSum);
			computeForceWithWall(dataLocal, gateTwoA[1], cMass, fSum);
		} else {
			computeForceWithWall(dataLocal, gateTwoB[0], cMass, fSum);
			computeForceWithWall(dataLocal, gateTwoB[1], cMass, fSum);
		}
#else
		else {
			computeForceWithWall(dataLocal, gateTwoA[0], cMass, fSum);
			computeForceWithWall(dataLocal, gateTwoA[1], cMass, fSum);
		}
#endif

#ifdef CLONE
		//decision point A: impaction from wall
		if(holeB.pointToLineDist(loc) < 20 && this->cloned == false) {
			this->cloning = true;
		}
#endif

		//sum up
		dvt.x += fSum.x / mass;
		dvt.y += fSum.y / mass;

		float2 &newVelo = dataLocal.velocity;
		float2 &newLoc = dataLocal.loc;
		float2 &newGoal = dataLocal.goal;

		float tick = 0.1;
		newVelo.x += dvt.x * tick * (1);// + this->random->gaussian() * 0.1);
		newVelo.y += dvt.y * tick * (1);// + this->random->gaussian() * 0.1);
		float dv = sqrt(newVelo.x * newVelo.x + newVelo.y * newVelo.y);

		if (dv > maxv) {
			newVelo.x = newVelo.x * maxv / dv;
			newVelo.y = newVelo.y * maxv / dv;
		}

		float mint = 1;

		if (goal.x < 0.5 * modelDevParams.WIDTH) {
			computeWallImpaction(dataLocal, gateOne[0], newVelo, tick, mint);
			computeWallImpaction(dataLocal, gateOne[1], newVelo, tick, mint);
		}
#ifdef CLONE
		else if (this->cloneid.x == '0') {
			computeWallImpaction(dataLocal, gateTwoA[0], newVelo, tick, mint);
			computeWallImpaction(dataLocal, gateTwoA[1], newVelo, tick, mint);
		} else {
			computeWallImpaction(dataLocal, gateTwoB[0], newVelo, tick, mint);
			computeWallImpaction(dataLocal, gateTwoB[1], newVelo, tick, mint);
		}
#else
		else {
			computeWallImpaction(dataLocal, gateTwoA[0], newVelo, tick, mint);
			computeWallImpaction(dataLocal, gateTwoA[1], newVelo, tick, mint);
		}
#endif

		newVelo.x *= mint;
		newVelo.y *= mint;
		newLoc.x += newVelo.x * tick;
		newLoc.y += newVelo.y * tick;

		float goalTemp = goal.x;

		if (goal.x < 0.5 * modelDevParams.WIDTH
			&& (newLoc.x - mass/cMass <= 0.25 * modelDevParams.WIDTH) 
			&& (newLoc.y - mass/cMass > 0.5 * modelDevParams.HEIGHT - LEFT_GATE_SIZE) 
			&& (newLoc.y - mass/cMass < 0.5 * modelDevParams.HEIGHT + LEFT_GATE_SIZE)) 
		{
			newGoal.x = 0;
			//if (goalTemp != newGoal.x) 
			//	atomicInc(&throughput, 8192);
		}

		float rightGateSize;
#ifdef CLONE
		if (this->cloneid.x == '0')
			rightGateSize = RIGHT_GATE_SIZE_A;
		else
			rightGateSize = RIGHT_GATE_SIZE_B;
#else
		rightGateSize = RIGHT_GATE_SIZE_A;
#endif

		if (goal.x > 0.5 * modelDevParams.WIDTH
			&& (newLoc.x + mass/cMass >= 0.75 * modelDevParams.WIDTH) 
			&& (newLoc.y - mass/cMass > 0.5 * modelDevParams.HEIGHT - rightGateSize) 
			&& (newLoc.y - mass/cMass < 0.5 * modelDevParams.HEIGHT + rightGateSize)) 
		{
			newGoal.x = modelDevParams.WIDTH;
#ifdef CLONE
			if (goalTemp != newGoal.x && this->cloneid.x != '0') 
#else
			if (goalTemp != newGoal.x) 	
#endif
				atomicInc(&throughput, 8192);

			
		}

		newLoc.x = correctCrossBoader(newLoc.x, width);
		newLoc.y = correctCrossBoader(newLoc.y, height);

		*(SocialForceAgentData*)this->dataCopy = dataLocal;
#ifndef CLONE
		if (dataLocal.id == 11)
			printf("\tcloned: %d, %f, %f, %f, %f\n", dataLocal.id, dataLocal.loc.x, dataLocal.loc.y, dataLocal.velocity.x, dataLocal.velocity.y);
#endif
	}

	__device__ void fillSharedMem(void *dataPtr){
		SocialForceAgentData *dataSmem = (SocialForceAgentData*)dataPtr;
		SocialForceAgentData *dataAgent = (SocialForceAgentData*)this->data;
		*dataSmem = *dataAgent;
	}

	__device__ void init(SocialForceModel *sfModel, int dataSlot) {
		this->myModel = sfModel;
		this->myWorld = sfModel->world;
#ifdef NDEBUG
		this->random = sfModel->random;
#endif
		this->color = colorConfigs.green;

		if (dataSlot == 11)
			this->color = colorConfigs.blue;

		this->cloneid.x = '0';
		this->id = dataSlot;
		this->cloned = false;
		this->cloning = false;
		this->myOrigin = NULL;

		SocialForceAgentData dataLocal; //= &sfModel->agentsA->dataArray[dataSlot];

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
		if(dataLocal.loc.x < (0.75 - 0.5 * CLONE_PERCENT) * modelDevParams.WIDTH) 
			dataLocal.goal = goal1;
		else
#endif
			dataLocal.goal = goal2;

		this->data = sfModel->agentsA->dataInSlot(dataSlot);
		this->dataCopy = sfModel->agentsA->dataCopyInSlot(dataSlot);
		*(SocialForceAgentData*)this->data = dataLocal;
		*(SocialForceAgentData*)this->dataCopy = dataLocal;
	}

#ifdef CLONE
	__device__ void initNewClone(const SocialForceAgent &agent, SocialForceAgent *originPtr, int dataSlot) {
		this->myModel = agent.myModel;
		this->myWorld = myModel->clonedWorld;
		this->color = colorConfigs.red;

		this->cloneid.x = agent.cloneid.x + 1;
		this->id = agent.id;
		this->cloned = false;
		this->cloning = false;
		this->myOrigin = originPtr;

		SocialForceAgentData dataLocal, dataCopyLocal;

		dataLocal = *(SocialForceAgentData*)agent.data;
		dataCopyLocal = *(SocialForceAgentData*)agent.dataCopy;

		dataLocal.agentPtr = this;
		dataCopyLocal.agentPtr = this;

		if (stepCount % 2 == 0) {
			this->data = myModel->agentsB->dataInSlot(dataSlot);
			this->dataCopy = myModel->agentsB->dataCopyInSlot(dataSlot);
		} else {
			this->data = myModel->agentsB->dataCopyInSlot(dataSlot);
			this->dataCopy = myModel->agentsB->dataInSlot(dataSlot);
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
		SocialForceAgent *ag = sfModel->agentsA->agentInSlot(dataSlot);
		ag->init(sfModel, dataSlot);
		sfModel->agentsA->add(ag, dataSlot);

	}
}

#ifdef CLONE
__global__ void cloneKernel(SocialForceModel *sfModel, int numAgentLocal) 
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAgentLocal){ // user init step
		SocialForceAgent ag, *agPtr = &sfModel->agentsA->agentArray[idx];
		ag = *agPtr;
		AgentPool<SocialForceAgent, SocialForceAgentData> *agentsB = sfModel->agentsB;

		if( ag.cloning == true && ag.cloned == false) {
			ag.cloned = true;
			ag.cloning = false;
			*agPtr = ag;

			int agentSlot = agentsB->agentSlot();
			int dataSlot = agentsB->dataSlot(agentSlot);

			SocialForceAgent *ag2 = agentsB->agentInSlot(dataSlot);
			ag2->initNewClone(ag, agPtr, dataSlot);
			agentsB->add(ag2, agentSlot);
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
	int numClonedAgents) 
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numClonedAgents) {
		SocialForceAgent *clonedAg = clonedPool->agentPtrArray[idx];
		SocialForceAgent *originalAg = clonedAg->myOrigin;
		SocialForceAgentData clonedAgData = *(SocialForceAgentData*) clonedAg->dataCopy;
		SocialForceAgentData originalAgData = *(SocialForceAgentData*) originalAg->dataCopy;

		bool match;
		//compare equivalence of two copies of data;
#define DELTA 0.000001
		match = (abs(clonedAgData.loc.x - originalAgData.loc.x) <= DELTA)
			&& (abs(clonedAgData.loc.y - originalAgData.loc.y) <= DELTA)
			&& (abs(clonedAgData.velocity.x - originalAgData.velocity.x) <= DELTA)
			&& (abs(clonedAgData.velocity.y - originalAgData.velocity.y) <= DELTA)
			&& (abs(clonedAgData.goal.x - originalAgData.goal.x) <= DELTA)
			&& (abs(clonedAgData.goal.y - originalAgData.goal.y) <= DELTA);
		if (match) {
			//remove from cloned set, reset clone state to non-cloned
			clonedPool->remove(idx);
			originalAg->cloned = false;
			originalAg->cloning = false;
		}
#ifdef _DEBUG
		else {
			printf("step: %d", stepCount);
			printf("\torigin: %d, %f, %f, %f, %f\n", originalAgData.id, originalAgData.loc.x, originalAgData.loc.y, originalAgData.velocity.x, originalAgData.velocity.y);
			printf("\tcloned: %d, %f, %f, %f, %f\n", clonedAgData.id, clonedAgData.loc.x, clonedAgData.loc.y, clonedAgData.velocity.x, clonedAgData.velocity.y);
		}
#endif
	}
}
#endif

#endif