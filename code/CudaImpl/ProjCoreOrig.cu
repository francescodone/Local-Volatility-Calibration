#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "TridagPar.cu.h"

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
			     REAL* d_myVarY) 
{
    int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    int gidy = blockIdx.y*blockDim.y + threadIdx.y;
    int gidz = blockIdx.z*blockDim.z + threadIdx.z;

    if (gidz < outer && gidy < numX && gidx < numY) {
        REAL tmp_1 = log(d_myX[gidz*numX+gidy]);
        REAL tmp_2 = d_myY[gidz*numY+gidx] - 0.5*nu*nu*d_myTimeline[gidz*numT+g];
        d_myVarX[gidz*numX*numY + gidy*numY + gidx] = exp(2.0*(beta*tmp_1 + tmp_2));
        d_myVarY[gidz*numX*numY + gidy*numY + gidx] = exp(2.0*(alpha*tmp_1 + tmp_2));
    }
}


// TODO: transpose d_dtInv
__global__ void rollback(const int outer,
			 const int numT,
			 const int g,
			 const REAL* d_myTimeline,
			 REAL* d_dtInv) 
{
    int gidx = blockIdx.x*blockDim.x + threadIdx.x;

    if (gidx < outer) {
        d_dtInv[gidx] = 1.0/(d_myTimeline[gidx*numT+g+1]-d_myTimeline[gidx*numT+g]);
    }
}


__global__ void explicitX(const int outer,
	                      const int numX,
			              const int numY,
			              const REAL* d_dtInv,
			              const REAL* d_myResult,
			              const REAL* d_myVarX,
			              const REAL* d_myDxx,
			              REAL* d_u) 
{
    int gidk = blockIdx.z*blockDim.z + threadIdx.z;
    int gidi = blockIdx.y*blockDim.y + threadIdx.y;
    int gidj = blockIdx.x*blockDim.x + threadIdx.x;

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

    int full_block_size = 256;
    int block_size = 32;

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
    cudaFree(d_my_result);

     // --- end of setPayoff - cuda ----


    REAL *d_myY, *d_myTimeline, *d_myVarX, *d_myVarY, *d_myResult, *d_myDxx, *d_u, *d_dtInv;
    cudaMalloc((void**) &d_myY, outer * numY * sizeof(REAL));
    cudaMalloc((void**) &d_myTimeline, outer * numT * sizeof(REAL));
    cudaMalloc((void**) &d_myVarX, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_myVarY, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_myResult, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_myDxx,    outer * numX *    4 * sizeof(REAL));
    cudaMalloc((void**) &d_u,        outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_dtInv, outer * sizeof(REAL));


    for(int g = globs.sizeT-2;g>=0;--g) { // seq

        cudaMemcpy(d_myX, globs.myX, outer * numX * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_myY, globs.myY, outer * numY * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_myTimeline, globs.myTimeline, outer * numT * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_myVarX, globs.myVarX, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_myVarY, globs.myVarY, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);


    // --- updateParams ---      
        updateParams<<<grid_3, block_2>>>(outer,
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

        cudaDeviceSynchronize();
        cudaMemcpy(globs.myVarX, d_myVarX, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(globs.myVarY, d_myVarY, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);


	 // --- rollback ---
        cudaMemcpy(d_dtInv, dtInv, outer * sizeof(REAL), cudaMemcpyHostToDevice);
        unsigned int num_blocks_outer = ((outer + (full_block_size - 1)) / full_block_size);
        rollback<<<num_blocks_outer, full_block_size>>>(outer, numT, g, d_myTimeline, d_dtInv);
        cudaDeviceSynchronize();   
        cudaMemcpy(dtInv, d_dtInv, outer * sizeof(REAL), cudaMemcpyDeviceToHost);


    // ---- explicit x

        cudaMemcpy(globs.myTimeline, d_myTimeline, outer * numT * sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_myResult, globs.myResult, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_myDxx,    globs.myDxx,    outer * numX *    4 * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_u,        u,              outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
       
       
       //dim3 grid_5 (outer, numX, numY);      
       explicitX<<<grid_3, block_2>>>(outer,
				     numX,
				     numY,
				     d_dtInv,
				     d_myResult,
				     d_myVarX,
				     d_myDxx,
				     d_u);
       

        // cudaDeviceSynchronize();

        // cudaMemcpy(globs.myResult, d_myResult, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost); 
        // cudaMemcpy(globs.myVarX,   d_myVarX,   outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
        // cudaMemcpy(globs.myDxx,    d_myDxx,    outer * numX *    4 * sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(u,              d_u,        outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);

        // cudaDeviceSynchronize();
      
      
        //	explicit x
        // do matrix transposition for u (after kernel is executed)
        // for( unsigned k = 0; k < outer; ++ k ) {
        //     for(unsigned i=0;i<numX;i++) {
        //         for(unsigned j=0;j<numY;j++) {
        //             u[k*numX*numY+j*numX+i] = 
        //                 dtInv[k]*globs.myResult[k*numX*numY + i*numY + j];

        //             if(i > 0) { 
        //                 u[k*numX*numY+j*numX+i] += 
        //                     0.5*( 0.5*globs.myVarX[k*numX*numY+i*numY+j]*globs.myDxx[k*numX*4+i*4+0] ) 
        //                     * globs.myResult[k*numX*numY + (i-1)*numY + j];
        //             }

        //             u[k*numX*numY+j*numX+i] +=
        //                 0.5*( 0.5*globs.myVarX[k*numX*numY+i*numY+j]*globs.myDxx[k*numX*4+i*4+1] )
        //                 * globs.myResult[k*numX*numY + i*numY + j];

        //             if(i < numX-1) {
        //                 u[k*numX*numY+j*numX+i] +=
        //                     0.5*( 0.5*globs.myVarX[k*numX*numY+i*numY+j]*globs.myDxx[k*numX*4+i*4+2] )
        //                     * globs.myResult[k*numX*numY + (i+1)*numY + j];
        //             }
        //         }
        //     }
        // }
      

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

    cudaFree(d_myX);
    cudaFree(d_myY);
    cudaFree(d_myTimeline);
    cudaFree(d_myVarX);
    cudaFree(d_myVarY);
    cudaFree(d_dtInv);

}

//#endif // PROJ_CORE_ORIG
