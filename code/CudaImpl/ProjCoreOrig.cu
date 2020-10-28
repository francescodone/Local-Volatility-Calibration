#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "TridagPar.cu.h"

/***********************************************************************************/
/* // Example of usage of the matTransposeTiled function			   */
/* // d_u   : [outer][numY][numX]						   */
/* // d_u_T : [outer][numX][numY]						   */
/* 										   */
/* REAL *d_u_T;									   */
/* cudaMalloc((void**) &d_u_T,        outer * numX * numY * sizeof(REAL));	   */
/* 										   */
/* matTransposeTiled<<<numX, numY, outer>>>(d_u, d_u_T, numX, numY, outer);	   */
/* cudaDeviceSynchronize();							   */
/* 										   */
/* matTransposeTiled<<<numY, numX, outer>>>(d_u_T, d_u, numY, numX, outer);	   */
/* cudaDeviceSynchronize();							   */
/* cudaFree(d_u_T);								   */
/***********************************************************************************/
__global__ void matTransposeTiled(REAL* A, REAL* B, int rowsA, int colsA, int outer) {
    const int TILE = 16;

    __shared__ REAL shtileTR[TILE][TILE][TILE /* +1 */];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z * threadIdx.z;

    if( x < rowsA && y < colsA )
	shtileTR[threadIdx.z][threadIdx.y][threadIdx.x] = A[z*rowsA*colsA + y*colsA + x];

    __syncthreads();

    if( x < colsA && y < rowsA && z < outer )
	B[z*rowsA*colsA + y*rowsA + x] = shtileTR[threadIdx.z][threadIdx.x][threadIdx.y];
}

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

// outer,numX, numY, d_myResult, d_myVarY, d_myDyy, d_v, d_u)

__global__ void explicitY(const int outer,
	                      const int numX,
			              const int numY,
			              const REAL* d_myResult,
			              const REAL* d_myVarY,
			              const REAL* d_myDyy,
			              REAL* d_v,
                          REAL* d_u) 
{
    int gidk = blockIdx.z*blockDim.z + threadIdx.z;
    int gidj = blockIdx.y*blockDim.y + threadIdx.y;
    int gidi = blockIdx.x*blockDim.x + threadIdx.x;

    if (gidk < outer && gidi < numX && gidj < numY) {

        int v_ind = gidk*numX*numY + gidi*numY + gidj;
        int my_vary_ind = gidk*numX*numY + gidi*numY + gidj;
        int my_dyy_ind = gidk*numY*4 + gidj*4;
        int my_result_ind = gidk*numX*numY + gidi*numY + gidj;

        d_v[v_ind] = 0.0;

        if (gidj > 0) {
            d_v[v_ind] += (0.5 * d_myVarY[my_vary_ind] * d_myDyy[my_dyy_ind])
                * d_myResult[my_result_ind-1];
        }

        d_v[v_ind] += (0.5 * d_myVarY[my_vary_ind] * d_myDyy[my_dyy_ind+1] )
            * d_myResult[my_result_ind];
        
        if(gidj < numY-1) {
            d_v[v_ind] +=  (0.5 * d_myVarY[my_vary_ind] * d_myDyy[my_dyy_ind+2] )
            *  d_myResult[my_result_ind+1];
        }

        d_u[gidk*numX*numY + gidj*numX + gidi] += d_v[v_ind]; 

    }
}


__global__ void implicitX(const int outer,
	                      const int numX,
			              const int numY,
                          const REAL* d_dtInv,
			              const REAL* d_myVarX,
			              const REAL* d_myDxx,
			              REAL* a,
                          REAL* b,
                          REAL* c)
{
    int gidi = blockIdx.x*blockDim.x + threadIdx.x;
    int gidj = blockIdx.y*blockDim.y + threadIdx.y;
    int gidk = blockIdx.z*blockDim.z + threadIdx.z;

    if (gidk < outer && gidi < numX && gidj < numY) { 
        const int ind = gidk*numX*numY+gidj*numX+gidi;
        const int mydxx_ind = gidk*numX*4+gidi*4;
        const int varx_ind = gidk*numX*numY+gidi*numY+gidj;

        a[ind] = - 0.5*(0.5*d_myVarX[varx_ind]*d_myDxx[mydxx_ind]);
        b[ind] = d_dtInv[gidk] - 0.5*(0.5*d_myVarX[varx_ind]*d_myDxx[mydxx_ind+1]);
        c[ind] = - 0.5*(0.5*d_myVarX[varx_ind]*d_myDxx[mydxx_ind+2]);
    }
}

// __global__ void implicitX_tridag(const int outer,
//                           const int numX,
// 			              const int numY,
// 			              REAL* a,
//                           REAL* b,
//                           REAL* c,
//                           REAL* u,
//                           REAL* yy)
// {
//     int gidj = blockIdx.x*blockDim.x + threadIdx.x;
//     int gidk = blockIdx.y*blockDim.y + threadIdx.y;

//     if (gidk < outer && gidj < numY) { 
//         const int ind = gidk*numX*numY + gidj*numX;
//         tridagPar(&a[ind], &b[ind], &c[ind],&u[ind],
//             numX, &u[ind], &yy[gidk*numY+gidj]);
//     }
// }


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
    REAL* v = (REAL*) malloc(outer * numY * numX * sizeof(REAL)); // [outer][numY][numX]

    REAL* a = (REAL*) malloc(outer * numZ * numZ * sizeof(REAL));
    REAL* b = (REAL*) malloc(outer * numZ * numZ * sizeof(REAL));
    REAL* c = (REAL*) malloc(outer * numZ * numZ * sizeof(REAL));
    REAL* y = (REAL*) malloc(outer * numZ * numZ * sizeof(REAL));
    REAL* yy = (REAL*) malloc(outer * numZ * numZ * sizeof(REAL));

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


    REAL *d_myY, *d_myTimeline, *d_myVarX, *d_myVarY, *d_myResult, *d_myDxx, *d_myDyy, *d_u, *d_v, *d_dtInv;
    REAL *d_a, *d_b, *d_c, *d_yy;
    cudaMalloc((void**) &d_myY, outer * numY * sizeof(REAL));
    cudaMalloc((void**) &d_myTimeline, outer * numT * sizeof(REAL));
    cudaMalloc((void**) &d_myVarX, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_myVarY, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_myResult, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_myDxx,    outer * numX *    4 * sizeof(REAL));
    cudaMalloc((void**) &d_myDyy,    outer * numY *    4 * sizeof(REAL));
    cudaMalloc((void**) &d_u,        outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_v,        outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_dtInv, outer * sizeof(REAL));

    cudaMalloc((void**) &d_a, outer * numZ * numZ * sizeof(REAL));
    cudaMalloc((void**) &d_b, outer * numZ * numZ * sizeof(REAL));
    cudaMalloc((void**) &d_c, outer * numZ * numZ * sizeof(REAL));
    cudaMalloc((void**) &d_yy, outer * numZ * numZ * sizeof(REAL));



    for(int g = globs.sizeT-2;g>=0;--g) { // seq

        cudaMemcpy(d_myX, globs.myX, outer * numX * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_myY, globs.myY, outer * numY * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_myTimeline, globs.myTimeline, outer * numT * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_myVarX, globs.myVarX, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_myVarY, globs.myVarY, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_myDyy, globs.myDyy, outer * numY * 4 * sizeof(REAL), cudaMemcpyHostToDevice);


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
           
       explicitX<<<grid_3, block_2>>>(outer,
				     numX,
				     numY,
				     d_dtInv,
				     d_myResult,
				     d_myVarX,
				     d_myDxx,
				     d_u);
       

        cudaDeviceSynchronize();

    // ------ explicit y

        dim3 grid_3_2 (dim_x, dim_y, outer);
        explicitY<<<grid_3_2, block_2>>>(outer,
	        numX, numY, d_myResult, d_myVarY, d_myDyy, d_v, d_u);
        cudaDeviceSynchronize();
        
        cudaMemcpy(u, d_u, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(v, d_v, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);

    // ------- implicit x

        implicitX<<<grid_3_2, block_2>>>(outer,
	        numX, numY, d_dtInv, d_myVarX, d_myDxx, d_a, d_b, d_c);
        cudaDeviceSynchronize();

        // dim3 grid_2_2 (dim_y, dim_outer, 1);
        // implicitX_tridag<<<grid_2_2, block_2>>>(outer, numX, numY, d_a, d_b, d_c, d_u, d_yy);


        cudaMemcpy(a, d_a, outer * numZ * numZ * sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(b, d_b, outer * numZ * numZ * sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(c, d_c, outer * numZ * numZ * sizeof(REAL), cudaMemcpyDeviceToHost);
        //cudaMemcpy(u, d_u, outer * numZ * numZ * sizeof(REAL), cudaMemcpyDeviceToHost);


        for( unsigned k = 0; k < outer; ++ k ) {
            for(unsigned j=0;j<numY;j++) {
                // here yy should have size [numX]
                tridagPar(&a[k*numX*numY+j*numX],
                    &b[k*numX*numY+j*numX],
                    &c[k*numX*numY+j*numX],
                    &u[k*numX*numY+j*numX],
                    numX,
                    &u[k*numX*numY+j*numX],
                    &yy[k*numY+j]);
            }
        }

        // ---- end of implicit x


        //	implicit y
        for( unsigned k = 0; k < outer; ++ k ) {
            for(unsigned i=0;i<numX;i++) { 
                for(unsigned j=0;j<numY;j++) {  // here a, b, c should have size [numY]
                    a[k*numX*numY+i*numY+j] = 
                        - 0.5*(0.5*globs.myVarY[k*numX*numY+i*numY+j]*globs.myDyy[k*numY*4+j*4+0]);
                    b[k*numX*numY+i*numY+j] = 
                        dtInv[k] - 0.5*(0.5*globs.myVarY[k*numX*numY+i*numY+j]*globs.myDyy[k*numY*4+j*4+1]);
                    c[k*numX*numY+i*numY+j] = 
                        - 0.5*(0.5*globs.myVarY[k*numX*numY+i*numY+j]*globs.myDyy[k*numY*4+j*4+2]);

                    y[k*numX*numY+i*numY+j] = dtInv[k]*u[k*numX*numY+j*numX+i] - 0.5*v[k*numX*numY+i*numY+j];
                }
            }
        }

        for( unsigned k = 0; k < outer; ++ k ) {
            for(unsigned i=0;i<numX;i++) { 
                // here yy should have size [numY]
                tridagPar(&a[k*numX*numY+i*numY],
                    &b[k*numX*numY+i*numY],
                    &c[k*numX*numY+i*numY],
                    &y[k*numX*numY+i*numY],
                    numY,
                    &globs.myResult[k*numX*numY + i*numY],
                    &yy[k*numX+i]);
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
