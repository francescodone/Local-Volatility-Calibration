#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "TridagPar.cu.h"

/*
void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{
    for(unsigned i=0;i<globs.myX.size();++i)
        for(unsigned j=0;j<globs.myY.size();++j) {
            globs.myVarX[i][j] = exp(2.0*(  beta*log(globs.myX[i])   
                                          + globs.myY[j]             
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    );
            globs.myVarY[i][j] = exp(2.0*(  alpha*log(globs.myX[i])   
                                          + globs.myY[j]             
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    ); // nu*nu
        }
}

void setPayoff(const REAL strike, PrivGlobs& globs )
{
	for(unsigned i=0;i<globs.myX.size();++i)
	{
		REAL payoff = max(globs.myX[i]-strike, (REAL)0.0);
		for(unsigned j=0;j<globs.myY.size();++j)
			globs.myResult[i][j] = payoff;
	}
}
*/
/*
inline void tridag(
    const vector<REAL>&   a,   // size [n]
    const vector<REAL>&   b,   // size [n]
    const vector<REAL>&   c,   // size [n]
    const vector<REAL>&   r,   // size [n]
    const int             n,
          vector<REAL>&   u,   // size [n]
          vector<REAL>&   uu   // size [n] temporary
) {
    int    i, offset;
    REAL   beta;

    u[0]  = r[0];
    uu[0] = b[0];

    for(i=1; i<n; i++) {
        beta  = a[i] / uu[i-1];

        uu[i] = b[i] - beta*c[i-1];
        u[i]  = r[i] - beta*u[i-1];
    }

#if 1
    // X) this is a backward recurrence
    u[n-1] = u[n-1] / uu[n-1];
    for(i=n-2; i>=0; i--) {
        u[i] = (u[i] - c[i]*u[i+1]) / uu[i];
    }
#else
    // Hint: X) can be written smth like (once you make a non-constant)
    for(i=0; i<n; i++) a[i] =  u[n-1-i];
    a[0] = a[0] / uu[n-1];
    for(i=1; i<n; i++) a[i] = (a[i] - c[n-1-i]*a[i-1]) / uu[n-1-i];
    for(i=0; i<n; i++) u[i] = a[n-1-i];
#endif
}
*/

/*
void
rollback( const unsigned g, const unsigned k, PrivGlobs& globs ) {
    unsigned numX = globs.sizeX,
             numY = globs.sizeY;

    unsigned numZ = max(numX,numY);

    unsigned i, j;

    REAL dtInv = 1.0/(globs.myTimeline[k][g+1]-globs.myTimeline[k][g]);

    REAL** u = new REAL*[numY];
    for (int idx = 0; idx<numY; idx++) {
        u[idx] = new REAL[numX];
    }
    //(numY, vector<REAL>(numX));   // [numY][numX]

    REAL** v = new REAL*[numX];
    for (int idx = 0; idx<numX; idx++) {
        v[idx] = new REAL[numY];
    }
    //vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
    REAL* a = new REAL[numZ];
    REAL* b = new REAL[numZ];
    REAL* c = new REAL[numZ];
    REAL* y = new REAL[numZ];
    REAL* yy = new REAL[numZ];
    //vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
    //vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

    //	explicit x
    for(i=0;i<numX;i++) {
        for(j=0;j<numY;j++) {
            u[j][i] = dtInv*globs.myResult[k][i][j];

            if(i > 0) { 
              u[j][i] += 0.5*( 0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][0] ) 
                            * globs.myResult[k][i-1][j];
            }
            u[j][i]  +=  0.5*( 0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][1] )
                            * globs.myResult[k][i][j];
            if(i < numX-1) {
              u[j][i] += 0.5*( 0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][2] )
                            * globs.myResult[k][i+1][j];
            }
        }
    }

    //	explicit y
    for(j=0;j<numY;j++)
    {
        for(i=0;i<numX;i++) {
            v[i][j] = 0.0;

            if(j > 0) {
              v[i][j] +=  ( 0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][0] )
                         *  globs.myResult[k][i][j-1];
            }
            v[i][j]  +=   ( 0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][1] )
                         *  globs.myResult[k][i][j];
            if(j < numY-1) {
              v[i][j] +=  ( 0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][2] )
                         *  globs.myResult[k][i][j+1];
            }
            u[j][i] += v[i][j]; 
        }
    }

    //	implicit x
    for(j=0;j<numY;j++) {
        for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
            a[i] =		 - 0.5*(0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][0]);
            b[i] = dtInv - 0.5*(0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][1]);
            c[i] =		 - 0.5*(0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][2]);
        }
        // here yy should have size [numX]
        tridagPar(a,b,c,u[j],numX,u[j],yy);
    }

    //	implicit y
    for(i=0;i<numX;i++) { 
        for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
            a[j] =		 - 0.5*(0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][0]);
            b[j] = dtInv - 0.5*(0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][1]);
            c[j] =		 - 0.5*(0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][2]);
        }

        for(j=0;j<numY;j++)
            y[j] = dtInv*u[j][i] - 0.5*v[i][j];

        // here yy should have size [numY]
        tridagPar(a,b,c,y,numY,globs.myResult[k][i],yy);
    }
}
*/

/*
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
) {	
    initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
    initOperator(globs.myX,globs.myDxx);
    initOperator(globs.myY,globs.myDyy);

    setPayoff(strike, globs);
    for(int i = globs.myTimeline.size()-2;i>=0;--i)
    {
        updateParams(i,alpha,beta,nu,globs);
        rollback(i, globs);
    }

    return globs.myResult[globs.myXindex][globs.myYindex];
}
*/


// --- setPayoff -----

__global__ void initPayoff(int outer, int numX, REAL* payoff_cuda, REAL* myX, REAL* strike) {
    int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    int gidy = blockIdx.y*blockDim.y + threadIdx.y;

    if (gidy < outer && gidx < numX) {
        payoff_cuda[gidy*numX+gidx] = max(myX[gidy*numX+gidx]-strike[gidy], (REAL)0.0);
    }
}

__global__ void updateGlobsMyResult(int outer, int numX, int numY, REAL* d_payoff, REAL* d_my_result) {
    int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    int gidy = blockIdx.y*blockDim.y + threadIdx.y;
    int gidz = blockIdx.z*blockDim.z + threadIdx.z;

    if (gidz <= outer && gidy <= numX && gidx <= numY) {
        d_my_result[gidz*numX*numY + gidy*numY + gidx] = d_payoff[gidz*numX+gidy];
    }
}

__global__ void updateParams(const int outer,
			     const int numX,
			     const int numY,
			     const int numT,
			     const int g,
			     const REAL alpha,
			     const REAL beta,
			     const REAL nu,
			     const REAL* d_myX,
			     const REAL* d_myY,
			     const REAL* d_myTimeline,
			     REAL* d_myVarX,
			     REAL* d_myVarY) {
  int gidk = blockIdx.x*blockDim.x + threadIdx.x;
  int gidi = blockIdx.y*blockDim.y + threadIdx.y;
  int gidj = blockIdx.z*blockDim.z + threadIdx.z;

  if (gidk < outer && gidi < numX && gidj < numY) {
    d_myVarX[gidk*numX*numY + gidi*numY + gidj] = 
      exp(2.0*(  beta*log(d_myX[gidk*numX+gidi])
		 + d_myY[gidk*numY+gidj]
		 - 0.5*nu*nu*d_myTimeline[gidk*numT+g] )
	  );
    d_myVarY[gidk*numX*numY + gidi*numY + gidj] = 
      exp(2.0*(  alpha*log(d_myX[gidk*numX+gidi])
		 + d_myY[gidk*numY+gidj]
		 - 0.5*nu*nu*d_myTimeline[gidk*numT+g] )
	  );
  }
}

__global__ void rollback(const int outer,
			 const int numT,
			 const int g,
			 const REAL* d_myTimeline,
			 REAL* d_dtInv) {
  int gidk = blockIdx.x*blockDim.x + threadIdx.x;

  if (gidk < outer) {
    d_dtInv[gidk] = 1.0/(d_myTimeline[gidk*numT+g+1]-d_myTimeline[gidk*numT+g]);
  }
}

__global__ void explicitX(const int outer,
			  const int numX,
			  const int numY,
			  const REAL* d_dtInv,
			  const REAL* d_myResult,
			  const REAL* d_myVarX,
			  const REAL* d_myDxx,
			  REAL* d_u) {
  int gidk = blockIdx.x*blockDim.x + threadIdx.x;
  int gidi = blockIdx.y*blockDim.y + threadIdx.y;
  int gidj = blockIdx.z*blockDim.z + threadIdx.z;

  if (gidk < outer && gidi < numX && gidj < numY) {
    d_u[gidk*numX*numY+gidj*numX+gidi] = 
      d_dtInv[gidk]*d_myResult[gidk*numX*numY + gidi*numY + gidj];
  
    if(gidi > 0) { 
      d_u[gidk*numX*numY+gidj*numX+gidi] += 
	0.5*( 0.5*d_myVarX[gidk*numX*numY+gidi*numY+gidj]*d_myDxx[gidk*numX*4+gidi*4+0] ) 
	* d_myResult[gidk*numX*numY + (gidi-1)*numY + gidj];
    }
    d_u[gidk*numX*numY+gidj*numX+gidi] += 
      0.5*( 0.5*d_myVarX[gidk*numX*numY+gidi*numY+gidj]*d_myDxx[gidk*numX*4+gidi*4+1] )
      * d_myResult[gidk*numX*numY + gidi*numY + gidj];
    if(gidi < numX-1) {
      d_u[gidk*numX*numY+gidj*numX+gidi] += 
	0.5*( 0.5*d_myVarX[gidk*numX*numY+gidi*numY+gidj]*d_myDxx[gidk*numX*4+gidi*4+2] )
	* d_myResult[gidk*numX*numY + (gidi+1)*numY + gidj];
    }
  }
}


void   run_OrigCPU(  
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
) {

     // printf("START");

    // calculating cuda dim
    
    int block_size = 16;

    dim3 block_2(block_size, block_size, 1);

    int  dim_outer = ceil( ((float) outer)/block_size); 
    int  dim_x = ceil( ((float) numX)/block_size );
    int  dim_y = ceil( ((float) numY)/block_size );

    // ----- ARRAY EXPNASION ------

    PrivGlobs    globs(numX, numY, numT, outer);
    unsigned numZ = max(numX,numY);

    REAL* strike = new REAL[outer];
    REAL* dtInv = (REAL*) malloc(outer*sizeof(REAL));

    REAL* u = (REAL*) malloc(outer * numY * numX * sizeof(REAL)); // [outer][numY][numX]
    REAL*** v = new REAL**[outer];
    
    for(int k=0; k<outer; k++) {
      //u[k] = new REAL*[numY];
      //for (int idx = 0; idx<numY; idx++) {
      //    u[k][idx] = new REAL[numX];
      //}
        v[k] = new REAL*[numX];
        for (int idx = 0; idx<numX; idx++) {
            v[k][idx] = new REAL[numY];
        }
    }

    REAL** a = new REAL*[outer];
    REAL** b = new REAL*[outer];
    REAL** c = new REAL*[outer];
    REAL** y = new REAL*[outer];
    REAL** yy = new REAL*[outer];

    for(int k=0; k<outer; k++) {
        a[k] = new REAL[numZ];
        b[k] = new REAL[numZ];
        c[k] = new REAL[numZ];
        y[k] = new REAL[numZ];
        yy[k] = new REAL[numZ];
    }

    
    // ----- MAIN LOOP ------


    for( unsigned k = 0; k < outer; ++ k ) {
      strike[k] = 0.001*k;
      // value
      initGrid(s0,alpha,nu,t, numX, numY, numT, k, globs);
      initOperator(globs.myX,globs.myDxx, globs.sizeX, k, numX);
      initOperator(globs.myY,globs.myDyy, globs.sizeY, k, numY);
    }


    // --- beginning of setPayoff - cuda ----

    REAL *d_payoff, *d_strike, *d_myX, *d_my_result;
    cudaMalloc((void**) &d_payoff, outer * numX * sizeof(REAL));
    cudaMalloc((void**) &d_strike, outer * sizeof(REAL));
    cudaMalloc((void**) &d_myX, outer * numX * sizeof(REAL));
    cudaMalloc((void**) &d_my_result, outer * numX * numY * sizeof(REAL));

    cudaMemcpy(d_strike, strike, outer * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myX, globs.myX, outer * numX * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_my_result, globs.myResult, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);

    dim3 grid_2 (dim_x, dim_outer, 1);
    initPayoff<<<grid_2, block_2>>>(outer, numX, d_payoff, d_myX, d_strike);
    cudaDeviceSynchronize();

    dim3 grid_3 (dim_y, dim_x, outer);
    updateGlobsMyResult<<<grid_3, block_2>>>(outer, numX, numY, d_payoff, d_my_result);
    cudaDeviceSynchronize();

    cudaMemcpy(globs.myResult, d_my_result, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);


    cudaFree(d_payoff);
    cudaFree(d_strike);
    cudaFree(d_myX);
    cudaFree(d_my_result);
    // free(h_payoff);

    // --- end of setPayoff - cuda ----



    for(int g = globs.sizeT-2;g>=0;--g) { // seq


      REAL *d_myX, *d_myY, *d_myTimeline, *d_myVarX, *d_myVarY;
      cudaMalloc((void**) &d_myX, outer * numX * sizeof(REAL));
      cudaMalloc((void**) &d_myY, outer * numY * sizeof(REAL));
      cudaMalloc((void**) &d_myTimeline, outer * numT * sizeof(REAL));
      cudaMalloc((void**) &d_myVarX, outer * numX * numY * sizeof(REAL));
      cudaMalloc((void**) &d_myVarY, outer * numX * numY * sizeof(REAL));

      cudaMemcpy(d_myX, globs.myX, outer * numX * sizeof(REAL), cudaMemcpyHostToDevice);
      cudaMemcpy(d_myY, globs.myY, outer * numY * sizeof(REAL), cudaMemcpyHostToDevice);
      cudaMemcpy(d_myTimeline, globs.myTimeline, outer * numT * sizeof(REAL), cudaMemcpyHostToDevice);
      cudaMemcpy(d_myVarX, globs.myVarX, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
      cudaMemcpy(d_myVarY, globs.myVarY, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);

      
      // --- updateParams ---      
      dim3 grid_4 (outer, numX, numY);
      updateParams<<<grid_4, block_2>>>(outer,
                                        numX,
                                        numY,
					numT,
                                        g,
                                        alpha,
                                        beta,
                                        nu,
                                        d_myX,
                                        d_myY,
                                        d_myTimeline,
                                        d_myVarX,
                                        d_myVarY);
      
      cudaMemcpy(globs.myX, d_myX, outer * numX * sizeof(REAL), cudaMemcpyDeviceToHost);
      cudaMemcpy(globs.myY, d_myY, outer * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
      //cudaMemcpy(globs.myTimeline, d_myTimeline, outer * numT * sizeof(REAL), cudaMemcpyDeviceToHost);
      cudaMemcpy(globs.myVarX, d_myVarX, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
      cudaMemcpy(globs.myVarY, d_myVarY, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
      
      cudaFree(d_myX);
      cudaFree(d_myY);
      //cudaFree(d_myTimeline);
      cudaFree(d_myVarX);
      cudaFree(d_myVarY);

      /*
      // --- updateParams ---    
      for( unsigned k = 0; k < outer; ++ k ) {
	for(unsigned i=0;i<globs.sizeX;++i) {
	  for(unsigned j=0;j<globs.sizeY;++j) {
	    globs.myVarX[k*numX*numY+i*numY+j] =
	      exp(2.0*(  beta*log(globs.myX[k*numX+i])
			 + globs.myY[k*numY+j]
			 - 0.5*nu*nu*globs.myTimeline[k*numT+g] )
		  );
	    globs.myVarY[k*numX*numY+i*numY+j] =
	      exp(2.0*(  alpha*log(globs.myX[k*numX+i])
			 + globs.myY[k*numY+j]
			 - 0.5*nu*nu*globs.myTimeline[k*numT+g] )
		  ); // nu*nu
	  }
	}
      }
      */

        // --- rollback ---

      REAL* d_dtInv;
      cudaMalloc((void**) &d_dtInv, outer * sizeof(REAL));
      cudaMemcpy(d_dtInv, dtInv, outer * sizeof(REAL), cudaMemcpyHostToDevice);

      rollback<<<outer, block_2>>>(outer, numT, g, d_myTimeline, d_dtInv);

      /*
      for( unsigned k = 0; k < outer; ++ k ) {
	dtInv[k] = 1.0/(globs.myTimeline[k*numT+g+1]-globs.myTimeline[k*numT+g]);
      }
      */

      //cudaMemcpy(dtInv, d_dtInv, outer * sizeof(REAL), cudaMemcpyDeviceToHost);
      //cudaFree(d_dtInv);
      
      cudaMemcpy(globs.myTimeline, d_myTimeline, outer * numT * sizeof(REAL), cudaMemcpyDeviceToHost);
      cudaFree(d_myTimeline);

      REAL *d_myResult, *d_myDxx, *d_u;
      cudaMalloc((void**) &d_myResult, outer * numX * numY * sizeof(REAL));
      cudaMalloc((void**) &d_myDxx,    outer * numX *    4 * sizeof(REAL));
      cudaMalloc((void**) &d_u,        outer * numX * numY * sizeof(REAL));

      cudaMemcpy(d_myResult, globs.myResult, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
      cudaMemcpy(d_myDxx,    globs.myDxx,    outer * numX *    4 * sizeof(REAL), cudaMemcpyHostToDevice);
      cudaMemcpy(d_u,        u,              outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
      
      /*
      explicitX<<<grid_4, block_2>>>(outer,
				     numX,
				     numY,
				     d_dtInv,
				     d_myResult,
				     d_myVarX,
				     d_myDxx,
				     d_u);
      */

      cudaMemcpy(globs.myResult, d_myResult, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
      cudaMemcpy(globs.myVarX,   d_myVarX,   outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
      cudaMemcpy(globs.myDxx,    d_myDxx,    outer * numX *    4 * sizeof(REAL), cudaMemcpyDeviceToHost);
      cudaMemcpy(u,              d_u,        outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
      cudaMemcpy(dtInv,          d_dtInv,    outer               * sizeof(REAL), cudaMemcpyDeviceToHost);

      cudaFree(d_myResult);
      cudaFree(d_myVarX);
      cudaFree(d_myDxx);
      cudaFree(d_u);
      cudaFree(d_dtInv);

      
      //	explicit x
      // do matrix transposition for u (after kernel is executed)
      for( unsigned k = 0; k < outer; ++ k ) {
	for(unsigned i=0;i<numX;i++) {
	  for(unsigned j=0;j<numY;j++) {
	    u[k*numX*numY+j*numX+i] = dtInv[k]*globs.myResult[k*numX*numY + i*numY + j];

	    if(i > 0) { 
	      u[k*numX*numY+j*numX+i] += 
		0.5*( 0.5*globs.myVarX[k*numX*numY+i*numY+j]*globs.myDxx[k*numX*4+i*4+0] ) 
		* globs.myResult[k*numX*numY + (i-1)*numY + j];
	    }
	    u[k*numX*numY+j*numX+i] +=
	      0.5*( 0.5*globs.myVarX[k*numX*numY+i*numY+j]*globs.myDxx[k*numX*4+i*4+1] )
	      * globs.myResult[k*numX*numY + i*numY + j];
	    if(i < numX-1) {
	      u[k*numX*numY+j*numX+i] +=
		0.5*( 0.5*globs.myVarX[k*numX*numY+i*numY+j]*globs.myDxx[k*numX*4+i*4+2] )
		* globs.myResult[k*numX*numY + (i+1)*numY + j];
	    }
	  }
	}
      }


        //	explicit y
        // matrix transposition and/or loop interchange?
        for( unsigned k = 0; k < outer; ++ k ) {
            for(unsigned j=0;j<numY;j++) {
                for(unsigned i=0;i<numX;i++) {
                    v[k][i][j] = 0.0;

                    if(j > 0) {
                    v[k][i][j] +=  ( 0.5*globs.myVarY[k*numX*numY+i*numY+j]*globs.myDyy[k*numY*4+j*4+0] )
                                *  globs.myResult[k*numX*numY + i*numY + j-1];
                    }
                    v[k][i][j]  +=   ( 0.5*globs.myVarY[k*numX*numY+i*numY+j]*globs.myDyy[k*numY*4+j*4+1] )
                                *  globs.myResult[k*numX*numY + i*numY + j];
                    if(j < numY-1) {
                    v[k][i][j] +=  ( 0.5*globs.myVarY[k*numX*numY+i*numY+j]*globs.myDyy[k*numY*4+j*4+2] )
                                *  globs.myResult[k*numX*numY + i*numY + j+1];
                    }
                    u[k*numX*numY+j*numX+i] += v[k][i][j]; 
                }
            }
        }

        //	implicit x
        for( unsigned k = 0; k < outer; ++ k ) {
            for(unsigned j=0;j<numY;j++) {
                for(unsigned i=0;i<numX;i++) {  // here a, b,c should have size [numX]
                    a[k][i] =		 - 0.5*(0.5*globs.myVarX[k*numX*numY+i*numY+j]*globs.myDxx[k*numX*4+i*4+0]);
                    b[k][i] = dtInv[k] - 0.5*(0.5*globs.myVarX[k*numX*numY+i*numY+j]*globs.myDxx[k*numX*4+i*4+1]);
                    c[k][i] =		 - 0.5*(0.5*globs.myVarX[k*numX*numY+i*numY+j]*globs.myDxx[k*numX*4+i*4+2]);
                }
                // here yy should have size [numX]
                tridagPar(a[k],b[k],c[k],&u[k*numX*numY+j*numX],numX,&u[k*numX*numY+j*numX],yy[k]);
            }
        }

        //	implicit y
        for( unsigned k = 0; k < outer; ++ k ) {
            for(unsigned i=0;i<numX;i++) { 
                for(unsigned j=0;j<numY;j++) {  // here a, b, c should have size [numY]
                    a[k][j] =		 - 0.5*(0.5*globs.myVarY[k*numX*numY+i*numY+j]*globs.myDyy[k*numY*4+j*4+0]);
                    b[k][j] = dtInv[k] - 0.5*(0.5*globs.myVarY[k*numX*numY+i*numY+j]*globs.myDyy[k*numY*4+j*4+1]);
                    c[k][j] =		 - 0.5*(0.5*globs.myVarY[k*numX*numY+i*numY+j]*globs.myDyy[k*numY*4+j*4+2]);
                }
                for(unsigned j=0;j<numY;j++)
                    y[k][j] = dtInv[k]*u[k*numX*numY+j*numX+i] - 0.5*v[k][i][j];

                // here yy should have size [numY]
                tridagPar(a[k],b[k],c[k],y[k],numY,&globs.myResult[k*numX*numY + i*numY],yy[k]);
            }
        }
        
    }

    for( unsigned k = 0; k < outer; ++ k ) { 
      res[k] = globs.myResult[k*numX*numY + globs.myXindex[k]*numY + globs.myYindex[k]];
    }

}

//#endif // PROJ_CORE_ORIG
