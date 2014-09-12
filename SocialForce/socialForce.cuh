#ifndef SOCIAL_FORCE_CUH
#define SOCIAL_FORCE_CUH

#include "gsimcore.cuh"
#include "gsimvisual.cuh"

#define DIST(ax, ay, bx, by) sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by))

typedef struct SocialForceAgentData : public GAgentData_t {
	float2 goal;
	float2 velocity;
	float v0;
	float mass;
	int id;
	uchar4 cloneid;

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
#define GATE_SIZE_A 2
#define GATE_SIZE_B 20

class SocialForceModel;
class SocialForceAgent;
class SocialForceClone;

__global__ void addAgentsOnDevice(SocialForceModel *sfModel);
//__global__ void cloneBoundary(SocialForceModel *sfmodel);
//__global__ void cloneNeighbor(SocialForceModel *sfmodel);

//class SocialForceClone {
//public:
//	AgentPool<SocialForceAgent, SocialForceAgentData> *agentPool, *agentPoolHost;
//	int obsLineNum;
//	float gateSize;
//
//	__host__ SocialForceClone(int num, int numMax, float gs, int numLine) {
//		agentPoolHost = new AgentPool<SocialForceAgent, SocialForceAgentData>(num, numMax, sizeof(SocialForceAgentData));
//		util::hostAllocCopyToDevice<AgentPool<SocialForceAgent, SocialForceAgentData> >(agentPoolHost, &agentPool);
//
//		obsLineNum = numLine;
//		gateSize = gs;
//	}
//
//	__host__ void start(GModel *model) {
//		int AGENT_NO = this->agentPoolHost->numElem;
//		int gSize = GRID_SIZE(AGENT_NO);
//		addAgentsOnDevice<<<gSize, BLOCK_SIZE>>>((SocialForceModel*)model->model);
//	}
//
//	__host__ void preStep(GModel *model) {
//		this->agentPoolHost->registerPool(model->worldHost, model->schedulerHost, this->agentPool);
//	}
//
//	__host__ void step(GModel *model) {
//		int numStepped = 0;
//		numStepped += this->agentPoolHost->stepPoolAgent(model->model);
//	}
//
//	__host__ void stop() {
//
//	}
//
//	__device__ void condition(float2 loc, float2 velo, float mass, float cMass, float2 &fSum) {
//		for (int wallIdx = 0; wallIdx < obsLineNum; wallIdx++) {
//			float diw, crx, cry;
//			diw = obsLines[wallIdx].pointToLineDist(loc, crx, cry);
//			float virDiw = DIST(loc.x, loc.y, crx, cry);
//			float niwx = (loc.x - crx) / virDiw;
//			float niwy = (loc.y - cry) / virDiw;
//			float drw = mass / cMass - diw;
//			float fiw1 = A * exp(drw / B);
//			if (drw > 0)
//				fiw1 += k1 * drw;
//			float fniwx = fiw1 * niwx;
//			float fniwy = fiw1 * niwy;
//
//			float fiwKgx = 0, fiwKgy = 0;
//			if (drw > 0)
//			{
//				float fiwKg = k2 * drw * (velo.x * (-niwy) + velo.y * niwx);
//				fiwKgx = fiwKg * (-niwy);
//				fiwKgy = fiwKg * niwx;
//			}
//
//			fSum.x += fniwx - fiwKgx;
//			fSum.y += fniwy - fiwKgy;
//		}
//	}
//
//	__device__ void condition2(float2 loc, float2 newVelo, float tick, float &mint) {
//		for (int wallIdx = 0; wallIdx < obsLineNum; wallIdx++) 
//		{
//			float crx, cry, tt;
//			int ret = obsLines[wallIdx].intersection2LineSeg(
//				loc.x, 
//				loc.y, 
//				loc.x + 0.5 * newVelo.x * tick,
//				loc.y + 0.5 * newVelo.y * tick,
//				crx,
//				cry
//				);
//			if (ret == 1) 
//			{
//				if (fabs(crx - loc.x) > 0)
//					tt = (crx - loc.x) / (newVelo.x * tick);
//				else
//					tt = (crx - loc.y) / (newVelo.y * tick + 1e-20);
//				if (tt < mint)
//					mint = tt;
//			}
//		}
//	}
//
//	__device__ void condition3(float2 newLoc, float mass, float cMass, float2 &newGoal) {
//		if ((newLoc.x - mass/cMass <= 0.25 * modelDevParams.WIDTH) && (newLoc.y - mass/cMass > 0.5 * modelDevParams.HEIGHT - gateSize) && (newLoc.y - mass/cMass < 0.5 * modelDevParams.HEIGHT + gateSize)) 
//		{
//			newGoal.x = 0;
//		}
//	}
//};

class SocialForceModel : public GModel {
public:
	//AgentPool<SocialForceAgent, SocialForceAgentData> *agentPool, *agentPoolHost;
	GRandom *random, *randomHost;
	cudaEvent_t timerStart, timerStop;

	AgentPool<SocialForceAgent, SocialForceAgentData> *agentsA, *agentsAHost;
	AgentPool<SocialForceAgent, SocialForceAgentData> *agentsB, *agentsBHost;

	GWorld *cloneWorld, *cloneWorldHost;

	SocialForceAgentData *dataHost;
	std::fstream fout;
	
	__host__ SocialForceModel(int num) {

		numAgentHost = num;
		cudaMemcpyToSymbol(numAgent, &num, sizeof(int));

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


		agentsAHost = new AgentPool<SocialForceAgent, SocialForceAgentData>(numAgentHost, numAgentHost, sizeof(SocialForceAgentData));
		util::hostAllocCopyToDevice<AgentPool<SocialForceAgent, SocialForceAgentData> >(agentsAHost, &agentsA);

		agentsBHost = new AgentPool<SocialForceAgent, SocialForceAgentData>(0, numAgentHost, sizeof(SocialForceAgentData));
		util::hostAllocCopyToDevice<AgentPool<SocialForceAgent, SocialForceAgentData> >(agentsBHost, &agentsB);

		worldHost = new GWorld();
		util::hostAllocCopyToDevice<GWorld>(worldHost, &world);

		randomHost = new GRandom(modelHostParams.MAX_AGENT_NO);
		util::hostAllocCopyToDevice<GRandom>(randomHost, &random);

		util::hostAllocCopyToDevice<SocialForceModel>(this, (SocialForceModel**)&this->model);

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

#ifdef _WIN32
		GSimVisual::getInstance().setWorld(this->world);
#endif
		cudaEventCreate(&timerStart);
		cudaEventCreate(&timerStop);
		cudaEventRecord(timerStart, 0);
	}

	__host__ void preStep()
	{
#ifdef CLONE
		int AGENT_NO = this->clone1Host->agentPoolHost->numElem;
		int gSize = GRID_SIZE(AGENT_NO);
		cloneBoundary<<<gSize, BLOCK_SIZE>>>((SocialForceModel*)this->model);
#endif
		this->agentsAHost->registerPool(this->worldHost, this->schedulerHost, this->agentsA);
		this->agentsBHost->registerPool(this->worldHost, this->schedulerHost, this->agentsB);
		
		cudaMemcpyToSymbol(modelDevParams, &modelHostParams, sizeof(modelConstants));

#ifdef _WIN32
		GSimVisual::getInstance().animate();
#endif

		getLastCudaError("copyHostToDevice");
	}

	__host__ void step()
	{
#ifndef NDEBUG
		int numAgent = this->agentsAHost->numElem;
		cudaMemcpy(dataHost, this->agentsAHost->dataArray, sizeof(SocialForceAgentData) * numAgent, cudaMemcpyDeviceToHost);
		for(int i = 0; i < numAgent; i ++) {
			fout << i << "\t" << dataHost[i].loc.x << "\t" << dataHost[i].loc.y << "\t"
				<< dataHost[i].velocity.x << "\t" << dataHost[i].velocity.y << "\t" << std::endl;
			fout.flush();
		}
		fout <<"==========================================================="<<std::endl<<std::endl;
		fout.flush();
#endif

		this->agentsAHost->stepPoolAgent(this->model);
		this->agentsBHost->stepPoolAgent(this->model);

		

#ifdef CLONE
		int AGENT_NO = this->clone1Host->agentPoolHost->numElem;
		int gSize = GRID_SIZE(AGENT_NO);
		cloneNeighbor<<<gSize, BLOCK_SIZE>>>((SocialForceModel*)this->model);
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
		return limit-1;
	else if (val < 0)
		return 0;
	return val;
}

class SocialForceAgent : public GAgent {
public:
	GRandom *random;
	SocialForceModel *model;

	bool cloned;
	bool cloning;

	__device__ void init(SocialForceModel *sfModel, int dataSlot) {
		this->model = sfModel;
		this->random = sfModel->random;
		this->color = colorConfigs.green;
		this->cloned = false;
		this->cloning = false;

		SocialForceAgentData dataLocal; //= &sfModel->agentsA->dataArray[dataSlot];

		dataLocal.loc.x = (0.25 + 0.5 * this->random->uniform()) * modelDevParams.WIDTH - 0.1;
		dataLocal.loc.y = this->random->uniform() * modelDevParams.HEIGHT;

		dataLocal.velocity.x = 4 * (this->random->uniform()-0.5);
		dataLocal.velocity.y = 4 * (this->random->uniform()-0.5);

		dataLocal.v0 = 2;
		dataLocal.mass = 50;
		dataLocal.cloneid.x = '0';
		dataLocal.id = dataSlot;

		float2 goal1 = make_float2(0.25 * modelDevParams.WIDTH, 0.50 * modelDevParams.HEIGHT);
		float2 goal2 = make_float2(0.75 * modelDevParams.WIDTH, 0.50 * modelDevParams.HEIGHT);

		if(length(dataLocal.loc - goal1) < length(dataLocal.loc - goal2)) 
			dataLocal.goal = goal1;
		else
			dataLocal.goal = goal2;

		this->data = &sfModel->agentsA->dataArray[dataSlot];
		this->dataCopy = &sfModel->agentsA->dataCopyArray[dataSlot];
		*(SocialForceAgentData*)this->data = dataLocal;
		*(SocialForceAgentData*)this->dataCopy = dataLocal;
	}

	__device__ void initNewClone(SocialForceAgent *agent, int dataSlot) {
		this->model = agent->model;
		this->random = agent->random;
		this->color = colorConfigs.red;
		this->cloned = false;
		this->cloning = false;

		SocialForceAgentData *data = &this->model->agentsB->dataArray[dataSlot];
		SocialForceAgentData *dataCopy = &this->model->agentsB->dataCopyArray[dataSlot];

		*data = *(SocialForceAgentData*)agent->data;
		*dataCopy = *(SocialForceAgentData*)agent->dataCopy;

		data->cloneid.x++;
		dataCopy->cloneid.x++;

		this->data = data;
		this->dataCopy = dataCopy;
	}

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

	__device__ void step(GModel *model){
		SocialForceModel *sfModel = (SocialForceModel*)model;
		GWorld *world = sfModel->world;
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
			ds = length(otherDataLocal.loc - loc);
			if (ds < 6 && ds > 0 )
				if (otherDataLocal.cloneid.x >= dataLocal.cloneid.x && otherDataLocal.id != dataLocal.id) 
					computeSocialForce(dataLocal, otherDataLocal, fSum);
				else // decision point B: impaction from neighboring agent
					this->cloning = true; 
			otherData = world->nextAgentDataFromSharedMem<SocialForceAgentData>(info);
		}

		obstacleLine obsLines[2];

		if (goal.x < 0.5 * modelDevParams.WIDTH) {
			obsLines[0] = gateOne[0];
			obsLines[1] = gateOne[1];
		} else {
			obsLines[0] = gateTwoA[0];
			obsLines[1] = gateTwoA[1];
		}

		float diw, crx, cry;
		for (int wallIdx = 0; wallIdx < GATE_LINE_NUM; wallIdx++) {
			diw = obsLines[wallIdx].pointToLineDist(loc, crx, cry);
			float virDiw = DIST(loc.x, loc.y, crx, cry);
			float niwx = (loc.x - crx) / virDiw;
			float niwy = (loc.y - cry) / virDiw;
			float drw = mass / cMass - diw;
			float fiw1 = A * exp(drw / B);
			if (drw > 0)
				fiw1 += k1 * drw;
			float fniwx = fiw1 * niwx;
			float fniwy = fiw1 * niwy;

			float fiwKgx = 0, fiwKgy = 0;
			if (drw > 0)
			{
				float fiwKg = k2 * drw * (velo.x * (-niwy) + velo.y * niwx);
				fiwKgx = fiwKg * (-niwy);
				fiwKgy = fiwKg * niwx;
			}

			fSum.x += fniwx - fiwKgx;
			fSum.y += fniwy - fiwKgy;
		}

		//decision point A: impaction from wall
		if(holeB.pointToLineDist(loc, crx, cry) < 6) {
			this->cloning = true;
		}
			

		//sum up
		dvt.x += fSum.x / mass;
		dvt.y += fSum.y / mass;

		float2 newVelo = velo;
		float2 newLoc = loc;
		float2 newGoal = goal;
		float tick = 0.1;
		newVelo.x += dvt.x * tick * (1 + this->random->gaussian() * 0.1);
		newVelo.y += dvt.y * tick * (1 + this->random->gaussian() * 0.1);
		float dv = sqrt(newVelo.x * newVelo.x + newVelo.y * newVelo.y);

		if (dv > maxv) {
			newVelo.x = newVelo.x * maxv / dv;
			newVelo.y = newVelo.y * maxv / dv;
		}

		float mint = 1;

		for (int wallIdx = 0; wallIdx < GATE_LINE_NUM; wallIdx++) 
		{
			float crx, cry, tt;
			int ret = obsLines[wallIdx].intersection2LineSeg(
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

		if (goal.x > 0.5 * modelDevParams.WIDTH
			&& (newLoc.x + mass/cMass >= 0.25 * modelDevParams.WIDTH) 
			&& (newLoc.y - mass/cMass > 0.5 * modelDevParams.HEIGHT - GATE_SIZE_A) 
			&& (newLoc.y - mass/cMass < 0.5 * modelDevParams.HEIGHT + GATE_SIZE_A)) 
		{
			newGoal.x = modelDevParams.WIDTH * 0.95;
		}

		newLoc.x = correctCrossBoader(newLoc.x, width);
		newLoc.y = correctCrossBoader(newLoc.y, height);

		SocialForceAgentData *dataCopyLocalPtr = (SocialForceAgentData*)this->dataCopy;
		dataCopyLocalPtr->loc = newLoc;
		dataCopyLocalPtr->velocity = newVelo;
		dataCopyLocalPtr->goal = newGoal;
	}

	__device__ void fillSharedMem(void *dataPtr){
		SocialForceAgentData *dataSmem = (SocialForceAgentData*)dataPtr;
		SocialForceAgentData *dataAgent = (SocialForceAgentData*)this->data;
		*dataSmem = *dataAgent;
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
		SocialForceAgent *ag = &sfModel->agentsA->agentArray[dataSlot];
		ag->init(sfModel, dataSlot);
		sfModel->agentsA->add(ag, dataSlot);

	}
}

__global__ void cloneKernel(SocialForceModel *sfModel, int numAgentLocal) {
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAgentLocal){ // user init step
		SocialForceAgent *ag = &sfModel->agentsA->agentArray[idx];
		AgentPool<SocialForceAgent, SocialForceAgentData> *agentsB = sfModel->agentsB;
		if( ag->cloning == true) {
			ag->cloned = true;
			ag->cloning = false;

			int agentSlot = agentsB->agentSlot();
			int dataSlot = agentsB->dataSlot(agentSlot);

			SocialForceAgent *ag2 = &agentsB->agentArray[agentSlot];
			ag2->initNewClone(ag, dataSlot);
			agentsB->add(ag2, agentSlot);
		}
	}
}

__global__ void doubleCheckKernel(SocialForceModel *sfModel, int numAgentLocal) {
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAgentLocal) {
		SocialForceAgent *ag = &sfModel->agentsB->agentArray[idx];
		ag->step(sfModel);
	}
}

/*
__global__ void cloneBoundary(SocialForceModel *sfModel) {
	float2 clone2GateDown, clone2GateUp;
	clone2GateDown = make_float2(0.25 * modelDevParams.WIDTH, 0.5 * modelDevParams.HEIGHT - 20);
	clone2GateUp = make_float2(0.25 * modelDevParams.WIDTH, 0.5 * modelDevParams.HEIGHT + 20);
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	AgentPool<SocialForceAgent, SocialForceAgentData> *clone1pool = sfModel->clone1->agentPool;
	AgentPool<SocialForceAgent, SocialForceAgentData> *clone2pool = sfModel->clone2->agentPool;
	if (idx < clone1pool->numElem) {
		SocialForceAgent *ag = clone1pool->agentPtrArray[idx];
		float2 loc = ag->data->loc;
		if ((length(loc - clone2GateDown) < 20 || length(loc - clone2GateUp) < 20) && ag->cloned == false) {
			ag->cloned = true;
			int agentSlot = clone2pool->agentSlot();
			int dataSlot = clone2pool->dataSlot(agentSlot);

			SocialForceAgent *ag2  = &clone2pool->agentArray[agentSlot];
			ag2->init(ag, sfModel->clone2, dataSlot, colorConfigs.red);
			clone2pool->add(ag2, agentSlot);

			;// clone ag and put in clone2
		}
	}
}

__global__ void cloneNeighbor(SocialForceModel *sfModel) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	AgentPool<SocialForceAgent, SocialForceAgentData> *clone1pool = sfModel->clone1->agentPool;
	AgentPool<SocialForceAgent, SocialForceAgentData> *clone2pool = sfModel->clone2->agentPool;
	if (idx < clone1pool->numElem) {
		SocialForceAgent *ag = clone1pool->agentPtrArray[idx];
		float2 loc = ag->data->loc;
		if (ag->cloning == true && ag->cloned == false) {
			ag->cloned = true;
			ag->cloning = false;

			int agentSlot = clone2pool->agentSlot();
			int dataSlot = clone2pool->dataSlot(agentSlot);

			SocialForceAgent *ag2  = &clone2pool->agentArray[agentSlot];
			ag2->init(ag, sfModel->clone2, dataSlot, colorConfigs.red);
			clone2pool->add(ag2, agentSlot);

			;// clone ag and put in clone2
		}
	}

}

*/
#endif