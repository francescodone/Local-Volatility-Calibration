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
    REAL* myResult; // [outer][numX][numY]

    //	coeffs
    REAL*     myVarX; // [outer][numX][numY]
    REAL*     myVarY; // [outer][numX][numY]

    //	operators
    REAL*     myDxx;  // [outer][numX][4]
    REAL*     myDyy;  // [outer][numY][4]

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

        this->myDxx = (REAL*) malloc(outer*numX*4*sizeof(REAL));
	this->myDyy = (REAL*) malloc(outer*numY*4*sizeof(REAL));

        this->myTimeline = (REAL*) malloc(outer*numT*sizeof(REAL));
        
        this->myVarX = (REAL*) malloc(outer*numX*numY*sizeof(REAL));
        this->myVarY = (REAL*) malloc(outer*numX*numY*sizeof(REAL));
        this->myResult = (REAL*) malloc(outer*numX*numY*sizeof(REAL));

	this->sizeX = numX;
        this->sizeY = numY;
        this->sizeT = numT;
        this->sizeO = outer;
    }
} __attribute__ ((aligned (128)));

struct PrivGlobsCuda {

    //	grid
    REAL*      myX;        // [outer][numX]
    REAL*      myY;        // [outer][numY]
    REAL*      myTimeline; // [outer][numT]
    unsigned*  myXindex;   // [outer]
    unsigned*  myYindex;   // [outer]

    //	variable
    REAL* myResult; // [outer][numX][numY]

    //	coeffs
    REAL*     myVarX; // [outer][numX][numY]
    REAL*     myVarY; // [outer][numX][numY]

    //	operators
    REAL*     myDxx;  // [outer][numX][4]
    REAL*     myDyy;  // [outer][numY][4]

    unsigned sizeX;
    unsigned sizeY;
    unsigned sizeT;
    unsigned sizeO;

    PrivGlobsCuda( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    PrivGlobsCuda(  const unsigned int& numX,
                const unsigned int& numY,
                const unsigned int& numT,
                const unsigned int& outer
             ) {

        cudaMalloc((void**) &this->myXindex, outer * sizeof(unsigned));
        cudaMalloc((void**) &this->myYindex, outer * sizeof(unsigned));

        cudaMalloc((void**) &this->myX, outer*numX*sizeof(REAL));        
        cudaMalloc((void**) &this->myY, outer*numY*sizeof(REAL));        

        cudaMalloc((void**) &this->myDxx, outer*numX*4*sizeof(REAL));  
        cudaMalloc((void**) &this->myDyy, outer*numY*4*sizeof(REAL));  

        cudaMalloc((void**) &this->myTimeline, outer*numT*sizeof(REAL));  

        cudaMalloc((void**) &this->myVarX, outer*numX*numY*sizeof(REAL));  
        cudaMalloc((void**) &this->myVarY, outer*numX*numY*sizeof(REAL));  
        cudaMalloc((void**) &this->myResult, outer*numX*numY*sizeof(REAL));  


	    this->sizeX = numX;
        this->sizeY = numY;
        this->sizeT = numT;
        this->sizeO = outer;
    }
};


void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const unsigned numX, const unsigned numY, const unsigned numT, const unsigned outer, PrivGlobs& globs   
            );

void initOperator(  REAL* x, 
                    REAL* Dxx, // [outer][numX][4]
                    unsigned numX,
                    unsigned k,
                    int row_s
                 );

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs);

void setPayoff(const REAL strike, PrivGlobs& globs );

void tridag(
    const REAL*  a,   // size [n]
    const REAL*   b,   // size [n]
    const REAL*   c,   // size [n]
    const REAL*   r,   // size [n]
    const int             n,
          REAL*   u,   // size [n]
          REAL*   uu   // size [n] temporary
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
