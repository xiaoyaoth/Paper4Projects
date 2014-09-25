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
#define GATE_LINE_NUM 2
#define GATE_SIZE_A 20
#define GATE_SIZE_B 20

#define MONITOR_STEP 75
#define CLONE

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
	AgentPool<SocialForceAgent, SocialForceAgentData> *agentsB, *agentsBHost;

	GWorld *clonedWorld, *clonedWorldHost;
	SocialForceAgent **agentPtrArrayUnsorted;

	SocialForceAgentData *dataHost;
	std::fstream fout;

	__host__ SocialForceModel(int num) {

		numAgentHost = num;
		cudaMemcpyToSymbol(numAgent, &num, sizeof(int));

		//init obstacles
		obstacleLine gateHost[2], holeHost;
		gateHost[0].init(0.25 * modelHostParams.WIDTH, -20, 0.25 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT - GATE_SIZE_A);
		gateHost[1].init(0.25 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT + GATE_SIZE_A, 0.25 * modelHostParams.WIDTH, modelHostParams.HEIGHT + 20);
		cudaMemcpyToSymbol(gateOne, &gateHost, 2 * sizeof(obstacleLine));

		gateHost[0].init(0.75 * modelHostParams.WIDTH, -20, 0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT - GATE_SIZE_A);
		gateHost[1].init(0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT + GATE_SIZE_A, 0.75 * modelHostParams.WIDTH, modelHostParams.HEIGHT + 20);
		cudaMemcpyToSymbol(gateTwoA, &gateHost, 2 * sizeof(obstacleLine));

		gateHost[0].init(0.75 * modelHostParams.WIDTH, -20, 0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT - GATE_SIZE_B);
		gateHost[1].init(0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT + GATE_SIZE_B, 0.75 * modelHostParams.WIDTH, modelHostParams.HEIGHT + 20);
		cudaMemcpyToSymbol(gateTwoB, &gateHost, 2 * sizeof(obstacleLine));

		holeHost.init(	0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT - GATE_SIZE_B,
			0.75 * modelHostParams.WIDTH, 0.5 * modelHostParams.HEIGHT + GATE_SIZE_B);
		cudaMemcpyToSymbol(holeB, &holeHost, sizeof(obstacleLine));

		//init agent pool
		agentsAHost = new AgentPool<SocialForceAgent, SocialForceAgentData>(numAgentHost, numAgentHost, sizeof(SocialForceAgentData));
		util::hostAllocCopyToDevice<AgentPool<SocialForceAgent, SocialForceAgentData> >(agentsAHost, &agentsA);

		agentsBHost = new AgentPool<SocialForceAgent, SocialForceAgentData>(0, numAgentHost, sizeof(SocialForceAgentData));
		util::hostAllocCopyToDevice<AgentPool<SocialForceAgent, SocialForceAgentData> >(agentsBHost, &agentsB);

		//init world
		worldHost = new GWorld();
		util::hostAllocCopyToDevice<GWorld>(worldHost, &world);

		clonedWorldHost = new GWorld();
		util::hostAllocCopyToDevice<GWorld>(clonedWorldHost, &clonedWorld);

		//init utility
		randomHost = new GRandom(modelHostParams.MAX_AGENT_NO);
		util::hostAllocCopyToDevice<GRandom>(randomHost, &random);

		util::hostAllocCopyToDevice<SocialForceModel>(this, (SocialForceModel**)&this->model);

		//alloc untouched agent array
		cudaMalloc((void**)&agentPtrArrayUnsorted, num * sizeof(SocialForceAgent*) );

		//alloc debug output
		dataHost = (SocialForceAgentData*)malloc(sizeof(SocialForceAgentData) * this->agentsAHost->numElem);

		char *outfname = new char[30];
		sprintf(outfname, "agent state.txt");
		fout.open(outfname, std::ios::out);
	}

	__host__ void start()
	{

		int AGENT_NO = this->agentsAHost->numElem;
		int gSize = GRID_SIZE(AGENT_NO);
		addAgentsOnDevice<<<gSize, BLOCK_SIZE>>>((SocialForceModel*)this->model);

		//copy the init agents to untouched agent list
		cudaMemcpy(agentPtrArrayUnsorted, 
			this->agentsAHost->agentPtrArray, 
			AGENT_NO * sizeof(SocialForceAgent*), 
			cudaMemcpyDeviceToDevice);

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
		else
			GSimVisual::getInstance().setWorld(this->clonedWorld);
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

	/*	

			//2.2. sort world and worldClone
			util::genNeighbor(this->clonedWorld, this->clonedWorldHost, modelHostParams.AGENT_NO);

			//2.3. step the cloned copy
			this->agentsBHost->stepPoolAgent(this->model);

			//3. double check
			compareOriginAndClone<<<gSize, BLOCK_SIZE>>>(this->agentsB, clonedWorld, numAgentsB);

			//4. clean pool again, since some agents are removed
			this->agentsBHost->cleanup(this->agentsB);
			
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
	*/
		}
		
#endif
		//5. swap data and dataCopy
		this->agentsAHost->swapPool();
		this->agentsBHost->swapPool();

		getLastCudaError("preStep");

#ifdef _WIN32
		GSimVisual::getInstance().animate();
#endif

#ifndef NDEBUG
		fout<<"step:"<<stepCountHost<<std::endl;
		int numAgent = this->agentsAHost->numElem;
		cudaMemcpy(dataHost, this->agentsAHost->dataArray, sizeof(SocialForceAgentData) * numAgent, cudaMemcpyDeviceToHost);
		for(int i = 0; i < numAgent; i ++) {
			fout << dataHost[i].id
				<< "\t" << dataHost[i].loc.x 
				<< "\t" << dataHost[i].loc.y 
				<< "\t"	<< dataHost[i].velocity.x 
				<< "\t" << dataHost[i].velocity.y 
				<< "\t" << std::endl;
			fout.flush();
		}
		fout <<"-------------------"<<std::endl;

		numAgent = this->agentsBHost->numElem;
		cudaMemcpy(dataHost, this->agentsBHost->dataArray, sizeof(SocialForceAgentData) * numAgent, cudaMemcpyDeviceToHost);
		for(int i = 0; i < numAgent; i ++) {
			fout << dataHost[i].id 
				<< "\t" << dataHost[i].loc.x 
				<< "\t" << dataHost[i].loc.y 
				<< "\t"	<< dataHost[i].velocity.x 
				<< "\t" << dataHost[i].velocity.y 
				<< "\t" << std::endl;
			fout.flush();
		}
		fout <<"==================="<<std::endl<<std::endl;
		fout.flush();
#endif
	}

	__host__ void stop()
	{
		fout.close();
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
#ifdef _DEBUG
		if (stepCount >= MONITOR_STEP) {
			int idx = threadIdx.x;
		}
#endif
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
		} else if (this->cloneid.x == '0') {
			computeForceWithWall(dataLocal, gateTwoA[0], cMass, fSum);
			computeForceWithWall(dataLocal, gateTwoA[1], cMass, fSum);
		} else {
			computeForceWithWall(dataLocal, gateTwoB[0], cMass, fSum);
			computeForceWithWall(dataLocal, gateTwoB[1], cMass, fSum);
		}

		//decision point A: impaction from wall
		if(holeB.pointToLineDist(loc) < 20 && this->cloned == false) {
			this->cloning = true;
		}

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
		} else if (this->cloneid.x == '0') {
			computeWallImpaction(dataLocal, gateTwoA[0], newVelo, tick, mint);
			computeWallImpaction(dataLocal, gateTwoA[1], newVelo, tick, mint);
		} else {
			computeWallImpaction(dataLocal, gateTwoB[0], newVelo, tick, mint);
			computeWallImpaction(dataLocal, gateTwoB[1], newVelo, tick, mint);
		}

		newVelo.x *= mint;
		newVelo.y *= mint;
		newLoc.x += newVelo.x * tick;
		newLoc.y += newVelo.y * tick;

		if (goal.x < 0.5 * modelDevParams.WIDTH
			&& (newLoc.x - mass/cMass <= 0.25 * modelDevParams.WIDTH) 
			&& (newLoc.y - mass/cMass > 0.5 * modelDevParams.HEIGHT - GATE_SIZE_A) 
			&& (newLoc.y - mass/cMass < 0.5 * modelDevParams.HEIGHT + GATE_SIZE_A)) 
		{
			newGoal.x = 0;
		}

		float rightGateSize;
		if (this->cloneid.x == '0')
			rightGateSize = GATE_SIZE_A;
		else
			rightGateSize = GATE_SIZE_B;

		if (goal.x > 0.5 * modelDevParams.WIDTH
			&& (newLoc.x + mass/cMass >= 0.25 * modelDevParams.WIDTH) 
			&& (newLoc.y - mass/cMass > 0.5 * modelDevParams.HEIGHT - rightGateSize) 
			&& (newLoc.y - mass/cMass < 0.5 * modelDevParams.HEIGHT + rightGateSize)) 
		{
			newGoal.x = modelDevParams.WIDTH;
		}

		newLoc.x = correctCrossBoader(newLoc.x, width);
		newLoc.y = correctCrossBoader(newLoc.y, height);

		*(SocialForceAgentData*)this->dataCopy = dataLocal;

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
		if(length(dataLocal.loc - goal1) < length(dataLocal.loc - goal2)) 
			dataLocal.goal = goal1;
		else
#endif
			dataLocal.goal = goal2;

		this->data = sfModel->agentsA->dataInSlot(dataSlot);
		this->dataCopy = sfModel->agentsA->dataCopyInSlot(dataSlot);
		*(SocialForceAgentData*)this->data = dataLocal;
		*(SocialForceAgentData*)this->dataCopy = dataLocal;
	}

	__device__ void initNewClone(SocialForceAgent *agent, int dataSlot) {
		this->myModel = agent->myModel;
		this->myWorld = myModel->clonedWorld;
		this->color = colorConfigs.red;

		this->cloneid.x = agent->cloneid.x + 1;
		this->id = agent->id;
		this->cloned = false;
		this->cloning = false;
		this->myOrigin = agent;

		SocialForceAgentData dataLocal;

		dataLocal = *(SocialForceAgentData*)agent->data;
		this->data = myModel->agentsB->dataInSlot(dataSlot);
		dataLocal.agentPtr = this;
		*(SocialForceAgentData*)this->data = dataLocal;

		//dataLocal = *(SocialForceAgentData*)agent->dataCopy;
		this->dataCopy = myModel->agentsB->dataCopyInSlot(dataSlot);
		*(SocialForceAgentData*)this->dataCopy = dataLocal;
	}
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

__global__ void replaceOriginalWithClone(GAgent **originalAgents, SocialForceAgent **clonedAgents, int numClonedAgent)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numClonedAgent){
		SocialForceAgent *ag = clonedAgents[idx];
#ifdef _DEBUG
		if (stepCount >= MONITOR_STEP) {
			int idx = threadIdx.x;	
		}
#endif
		originalAgents[ag->id] = ag;
	}
}

__global__ void cloneKernel(SocialForceModel *sfModel, int numAgentLocal) 
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAgentLocal){ // user init step
		SocialForceAgent *ag = &sfModel->agentsA->agentArray[idx];
		AgentPool<SocialForceAgent, SocialForceAgentData> *agentsB = sfModel->agentsB;

#ifdef _DEBUG
		if (stepCount >= MONITOR_STEP) {
			int aidx = threadIdx.x;
			agentsB->numElem = agentsB->numElem;
		}
#endif

		if( ag->cloning == true && ag->cloned == false) {

			ag->cloned = true;
			ag->cloning = false;

			int agentSlot = agentsB->agentSlot();
			int dataSlot = agentsB->dataSlot(agentSlot);

			SocialForceAgent *ag2 = agentsB->agentInSlot(dataSlot);
			ag2->initNewClone(ag, dataSlot);
			agentsB->add(ag2, agentSlot);
		}

#ifdef _DEBUG
		if (stepCount >= MONITOR_STEP) {
			int aidx = threadIdx.x;
			agentsB->numElem = agentsB->numElem;
		}
#endif
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
		SocialForceAgentData clonedAgData = *(SocialForceAgentData*) clonedAg->dataCopy;
		SocialForceAgent *originalAg = clonedAg->myOrigin;
		SocialForceAgentData originalAgData = *(SocialForceAgentData*) originalAg->dataCopy;
#ifdef _DEBUG
		if (stepCount >= MONITOR_STEP) {
			int aidx = threadIdx.x;
		}
#endif

		bool match;
		//compare equivalence of two copies of data;
		match = clonedAgData.loc.x == originalAgData.loc.x
			&& clonedAgData.loc.y == originalAgData.loc.y
			&& clonedAgData.velocity.x == originalAgData.velocity.x
			&& clonedAgData.velocity.y == originalAgData.velocity.y
			&& clonedAgData.goal.x == originalAgData.goal.x
			&& clonedAgData.goal.y == originalAgData.goal.y;
		if (match) {
			//remove from cloned set, reset clone state to non-cloned
			clonedPool->remove(idx);
			originalAg->cloned = false;
			originalAg->cloning = false;
#ifdef CLONE
		}
#else
			printf("%d: %d and %d match \n", stepCount, originalAg->id, clonedAg->id);
		} else {
			printf("%d: %d and %d not match \n", stepCount, originalAg->id, clonedAg->id);
		}
#endif
	}
}

#endif