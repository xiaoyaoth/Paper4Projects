#ifndef GSIMVISUAL_H
#define GSIMVISUAL_H

#include "gsimlib_header.cuh"
#include "book.h"
#include "gl_helper.h"
#include "cuda_gl_interop.h"
#include "gsimcore.cuh"

class GSimVisual;

namespace visUtil{
	__global__ void paint(uchar4 *devPtr, const GWorld *world, int width, int height, int scale);
};

__constant__ int clicksDev;

class GSimVisual{
public:
	static int clicks;
	GLuint bufferObj;
	cudaGraphicsResource *resource;
	GWorld *world;
	int width;
	int height;
	int dotScale; // actual width = width * dotScale

	PFNGLBINDBUFFERARBPROC    glBindBuffer;
	PFNGLDELETEBUFFERSARBPROC glDeleteBuffers;
	PFNGLGENBUFFERSARBPROC    glGenBuffers;
	PFNGLBUFFERDATAARBPROC    glBufferData;

	GSimVisual(){
		if (VISUALIZE == true) {
			this->width = 256;
			this->height = 256;
			this->dotScale = 2;
			glBindBuffer     = NULL;
			glDeleteBuffers  = NULL;
			glGenBuffers     = NULL;
			glBufferData     = NULL;

			int c = 1;
			char *dummy = " ";
			glutInit( &c, &dummy );
			glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
			glutInitWindowSize( this->width * this->dotScale, this->height * this->dotScale );
			glutCreateWindow( "bitmap" );

			glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
			glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
			glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
			glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");
			glGenBuffers( 1, &bufferObj );
			glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
			glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, dotScale * dotScale * width * height * sizeof(uchar4),
				NULL, GL_DYNAMIC_DRAW_ARB );
			cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
			getLastCudaError("cudaGraphicsGLRegisterBuffer");

			glutDisplayFunc(drawFunc);
			glutIdleFunc(idleFunc);
			glutMouseFunc(mouseFunc);
			glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
		}
	}

	static void mouseFunc(int button, int state, int x, int y) {
		GSimVisual vis = GSimVisual::getInstance();
		if (state == GLUT_DOWN) 
		{
			if (button == GLUT_LEFT_BUTTON)// GLUT_MIDDLE_BUTTON or GLUT_RIGHT_BUTTON
			{
				clicks++;
			}
		}
	}

	static void idleFunc(){
		GSimVisual vis = GSimVisual::getInstance();
		uchar4* devPtr;
		size_t size;
		cudaGraphicsMapResources(1, &vis.resource, NULL);
		getLastCudaError("cudaGraphicsMapResources");
		cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, vis.resource);
		getLastCudaError("cudaGraphicsResourceGetMappedPointer");
		//paint kernel here...
		cudaGraphicsUnmapResources(1, &vis.resource, NULL);
		getLastCudaError("cudaGraphicsUnmapResources");

		glutPostRedisplay();
	}

	static void drawFunc(){
		glClearColor( 1.0, 1.0, 1.0, 1.0 );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

		GSimVisual vis = GSimVisual::getInstance();

		uchar4* devPtr;
		size_t size;
		cudaGraphicsMapResources(1, &vis.resource, NULL);
		getLastCudaError("cudaGraphicsMapResources");
		cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, vis.resource);
		getLastCudaError("cudaGraphicsResourceGetMappedPointer");
		cudaMemset(devPtr, 0, size);
		getLastCudaError("cudaMemset");

		glEnable(GL_TEXTURE_2D);
		int gSize = GRID_SIZE(modelHostParams.AGENT_NO);
		cudaMemcpyToSymbol(clicksDev, &clicks, sizeof(int));

		visUtil::paint<<<gSize, BLOCK_SIZE>>>(devPtr, vis.world, vis.width, vis.height, vis.dotScale);

		cudaGraphicsUnmapResources(1, &vis.resource, NULL);
		getLastCudaError("cudaGraphicsUnmapResources");

		glDrawPixels(vis.width * vis.dotScale, vis.height * vis.dotScale, GL_RGBA, GL_UNSIGNED_BYTE, 0 );


		glutSwapBuffers();
		glutPostRedisplay();
	}

public:
	static GSimVisual& getInstance(){
		static GSimVisual instance;
		return instance;
	}

	void setWorld(GWorld *world){
#ifdef _WIN32
		if (VISUALIZE == true) {
			GSimVisual::getInstance().world = world;
		}
#endif
	}

	void animate(){
#ifdef _WIN32
		if (VISUALIZE == true)
			glutMainLoopEvent();
#endif
	}

	void stop(){
#ifdef _WIN32
		if (VISUALIZE == true)
			glutLeaveMainLoop();
#endif
	}
};

int GSimVisual::clicks = 0;

__global__ void visUtil::paint(uchar4 *devPtr, const GWorld *world, int width, int height, int dotScale)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < modelDevParams.AGENT_NO){
		GAgent *ag = world->allAgents[idx];
		FLOATn myLoc = ag->data->loc;
		int canvasX = (int)(myLoc.x) * width * dotScale; 
		canvasX /= (int)world->width;
		int canvasY = (int)(myLoc.y) * height * dotScale;
		canvasY /= (int)world->height;

		int dotDimX = width * dotScale / world->width;
		int dotDimY = height * dotScale / world->height;

		if (dotDimX < 2) dotDimX = 2;
		if (dotDimY < 2) dotDimY = 2;
		
#ifdef CLONE
		//bool isPaint = !((ag->color.x == colorConfigs.green.x) && (ag->color.y == colorConfigs.green.y) && (ag->color.z == colorConfigs.green.z));
		bool isPaint = true;
		if (clicksDev % 3 == 0)
			isPaint = true;
		else if (clicksDev % 3 == 1)
			isPaint = !((ag->color.x == colorConfigs.green.x) && (ag->color.y == colorConfigs.green.y) && (ag->color.z == colorConfigs.green.z));
		else
			isPaint = !((ag->color.x == colorConfigs.red.x) && (ag->color.y == colorConfigs.red.y) && (ag->color.z == colorConfigs.red.z));
			
#else
		bool isPaint = true;
#endif
		if (isPaint)
			for (int j = 0; j < dotDimY-1; j++)
				for (int i = 0; i < dotDimX-1; i++) 
				{
					int canvasXNew = canvasX + i;
					int canvasYNew = canvasY + j;
					int canvasIdx = canvasYNew * width * dotScale + canvasXNew;
					devPtr[canvasIdx] = ag->color;
				}
	}
}


#endif