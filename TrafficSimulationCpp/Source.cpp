struct link {
	int idxStart;
	int idxEnd;
	int *segments;
	int id;
};

struct segment {
	int *lanes;
	int id;
};

struct lane {
	int *connectedLanes;
	int *vehicles;
	int *bufferedVehicles;
	int inputCapacity;
	int outputCapacity;
	int id;
};

struct node {
	int nodePos;
	int upLinks;
	int downLinks;
	int id;
};

struct vehicle {
	int onLane;
	int id;
};
