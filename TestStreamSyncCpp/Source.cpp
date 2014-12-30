#include <stdio.h>
#include <stdlib.h>
#include <string>

#define NUM_GATES 4

void decode(int cloneidCode, int *cloneid, int *numChoicesPerGate) 
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
int encode(int *cloneidArrayVal, int *numChoicesPerGate)
{
	int ret = 0;
	int factor = 1;
	for (int i = 0; i < NUM_GATES; i++) {
		ret += factor * cloneidArrayVal[i];
		factor *= numChoicesPerGate[i];
	}
	return ret;
}

void addChild(int **cloneidAll, int *queue, int *choicesPerGate, int &tailIdx) {
	int headIdx = 0;
	while(headIdx < tailIdx) {
		int *cloneid = cloneidAll[queue[headIdx]];
		int childCloneId[NUM_GATES];

		int nonZeroPos = -1;
		for (int i = NUM_GATES - 1; i >= 0; i--) {
			if (cloneid[i] > 0) {
				nonZeroPos = i;
				break;
			}
		}

		nonZeroPos++;
		while (nonZeroPos < NUM_GATES) {
			for (int i = 1; i < choicesPerGate[nonZeroPos]; i++) {
				memcpy(childCloneId, cloneid, NUM_GATES * sizeof(int));
				childCloneId[nonZeroPos] = i;
				int code = encode(childCloneId, choicesPerGate);
				queue[tailIdx++] = code;
			}
			nonZeroPos++;
		}
		headIdx++;
	}
}

int main() {
	// === try another way ===
	int *choicesPerGate = new int[NUM_GATES];
	int totalClone = 1;
	for (int i = 0; i < NUM_GATES; i++) {
		choicesPerGate[i] = i+3;
		totalClone *= choicesPerGate[i];
	}
	int *queue = new int[totalClone];
	int tailIdx = 1;
	int headIdx = 0;
	queue[0] = 0;

	int **cloneid = new int*[totalClone];

	for (int i = 0; i < totalClone; i++) {
		cloneid[i] = new int[NUM_GATES];
		decode(i, cloneid[i], choicesPerGate);
	}

	addChild(cloneid, queue, choicesPerGate, tailIdx);

	int level = 0;
	int *execLevel = new int[NUM_GATES + 1];
	for(int i = 0; i < tailIdx; i++) {
		int zeroCounter = 0;
		int idArray[NUM_GATES];
		decode(queue[i], idArray, choicesPerGate);
		for (int j = 0; j < NUM_GATES; j++) {
			if (idArray[j] == 0)
				zeroCounter++;
		}
		if (level == NUM_GATES - zeroCounter) {
			printf("\nlevel: %d, at:%d\n", level, i);
			execLevel[level] = i;
			level++;
		}
		printf("%d ", queue[i]);
	}

	printf("\n");
	for (int i = 0; i < NUM_GATES + 1; i++)
		printf("%d ", execLevel[i]);
	return 0;
}