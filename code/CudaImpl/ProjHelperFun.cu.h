#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Constants.h"

using namespace std;


struct PrivGlobs {

    //	grid
    REAL*      myX;        // [outer][numX]
    REAL*      myY;        // [outer][numY]
    REAL*      myTimeline; // [outer][numT]
    unsigned*  myXindex;   // [outer]
    unsigned*  myYindex;   // [outer]

    //	variable
    REAL*      myResult; // [outer][numX][numY]

    //	coeffs
    REAL*      myVarX; // [outer][numX][numY]
    REAL*      myVarY; // [outer][numX][numY]

    //	operators
    REAL***   myDxx;  // [outer][numX][4]
    REAL***   myDyy;  // [outer][numY][4]

    unsigned sizeX;
    unsigned sizeY;
    unsigned sizeT;
    unsigned sizeO;

    PrivGlobs( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    PrivGlobs(  const unsigned int& numX,
                const unsigned int& numY,
                const unsigned int& numT,
                const unsigned int& outer
             ) {
        this->myXindex = new unsigned[outer];
        this->myYindex = new unsigned[outer];

        this->myX = (REAL*) malloc(outer*numX*sizeof(REAL));
        this->myY = (REAL*) malloc(outer*numY*sizeof(REAL));

        this->myDxx = new REAL**[outer];
        for(int k=0; k<outer; k++) {
            this->myDxx[k] = new REAL*[numX];
            for(int l=0; l<numX; l++) {
                this->myDxx[k][l] = new REAL[4];
            }
        }

        this->myDyy = new REAL**[outer];
        for(int k=0; k<outer; k++) {
            this->myDyy[k] = new REAL*[numY];
            for(int l=0; l<numY; l++) {
                this->myDyy[k][l] = new REAL[4];
            }
        }

	this->myTimeline = (REAL*) malloc(outer*numT*sizeof(REAL));
	/*
        //this->myTimeline = new REAL[numT];
        this->myTimeline = new REAL*[outer];
        for(int k=0; k<outer; k++) {
            this->myTimeline[k] = new REAL[numT];
        }
	*/

	this->myVarX = (REAL*) malloc(outer*numX*numY*sizeof(REAL));
	this->myVarY = (REAL*) malloc(outer*numX*numY*sizeof(REAL));
        //this->myVarX = new REAL**[outer];
        //this->myVarY = new REAL**[outer];
        this->myResult = (REAL*) malloc(outer*numX*numY*sizeof(REAL));

        /*
        for(int k=0; k<outer; k++) {
            this->  myVarX[k] = new REAL*[numX];
            this->  myVarY[k] = new REAL*[numX];
            for(unsigned i=0;i<numX;++i) {
                this->  myVarX[k][i] = new REAL[numY];
                this->  myVarY[k][i] = new REAL[numY];
            }
        }
	*/
        this->sizeX = numX;
        this->sizeY = numY;
        this->sizeT = numT;
        this->sizeO = outer;
    }
} __attribute__ ((aligned (128)));


void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const unsigned numX, const unsigned numY, const unsigned numT, const unsigned outer, PrivGlobs& globs   
            );

void initOperator(  REAL* x, 
                    REAL*** Dxx,
                    unsigned xsize,
                    unsigned k,
                    int row_s
                 );

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs);

void setPayoff(const REAL strike, PrivGlobs& globs );

void tridag(
    const vector<REAL>&   a,   // size [n]
    const vector<REAL>&   b,   // size [n]
    const vector<REAL>&   c,   // size [n]
    const vector<REAL>&   r,   // size [n]
    const int             n,
          vector<REAL>&   u,   // size [n]
          vector<REAL>&   uu   // size [n] temporary
);

void rollback( const unsigned g, PrivGlobs& globs );

REAL   value(   PrivGlobs    globs,
                const REAL s0,
                const REAL strike, 
                const REAL t, 
                const REAL alpha, 
                const REAL nu, 
                const REAL beta,
                const unsigned int numX,
                const unsigned int numY,
                const unsigned int numT
            );

void run_OrigCPU(  
                const unsigned int&   outer,
                const unsigned int&   numX,
                const unsigned int&   numY,
                const unsigned int&   numT,
                const REAL&           s0,
                const REAL&           t, 
                const REAL&           alpha, 
                const REAL&           nu, 
                const REAL&           beta,
                      REAL*           res   // [outer] RESULT
            );

#endif // PROJ_HELPER_FUNS
