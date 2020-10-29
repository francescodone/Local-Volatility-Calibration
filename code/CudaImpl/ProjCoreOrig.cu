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

__global__ void initPayoff(int outer, int numX, REAL* payoff_cuda, REAL* myX) {
    int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    int gidy = blockIdx.y*blockDim.y + threadIdx.y;

    if (gidy < outer && gidx < numX) {
        payoff_cuda[gidy*numX+gidx] = max(myX[gidy*numX+gidx]-gidy*0.001, (REAL)0.0);
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

__global__ void implicitX_tridag(const int outer,
                          const int numX,
			              const int numY,
			              REAL* a,
                          REAL* b,
                          REAL* c,
                          REAL* u,
                          REAL* yy)
{
    int gidj = blockIdx.x*blockDim.x + threadIdx.x;
    int gidk = blockIdx.y*blockDim.y + threadIdx.y;

    if (gidk < outer && gidj < numY) { 
        const int ind = gidk*numX*numY + gidj*numX;
        tridag(&a[ind], &b[ind], &c[ind],&u[ind],
            numX, &u[ind], &yy[gidk*numY*numX+gidj*numX]);
    }
}


__global__ void implicitY_1(const int outer,
	                      const int numX,
			              const int numY,
                          const REAL* d_dtInv,
			              const REAL* d_myVarY,
			              const REAL* d_myDyy,
			              REAL* a,
                          REAL* b,
                          REAL* c)
{
    int gidj = blockIdx.x*blockDim.x + threadIdx.x;
    int gidi = blockIdx.y*blockDim.y + threadIdx.y;
    int gidk = blockIdx.z*blockDim.z + threadIdx.z;

    if (gidk < outer && gidj < numY && gidi < numX) { 
        const int ind = gidk*numX*numY + gidi*numY + gidj;
        const int vary_ind = gidk*numX*numY + gidi*numY + gidj;
        const int mydyy_ind = gidk*numY*4+gidj*4;

        a[ind] = - 0.5*(0.5 * d_myVarY[vary_ind] * d_myDyy[mydyy_ind]);
        b[ind] = d_dtInv[gidk] - 0.5*(0.5 * d_myVarY[vary_ind] * d_myDyy[mydyy_ind+1]);
        c[ind] = - 0.5*(0.5 * d_myVarY[vary_ind] * d_myDyy[mydyy_ind+2]);
    }
}

__global__ void implicitY_2(const int outer,
	                      const int numX,
			              const int numY,
                          const REAL* d_dtInv,
			              const REAL* d_u,
			              const REAL* d_v,
			              REAL* d_y)
{
    int gidj = blockIdx.x*blockDim.x + threadIdx.x;
    int gidi = blockIdx.y*blockDim.y + threadIdx.y;
    int gidk = blockIdx.z*blockDim.z + threadIdx.z;

    if (gidk < outer && gidj < numY && gidi < numX) { 
        d_y[gidk*numX*numY + gidi*numY + gidj] = 
            d_dtInv[gidk] * d_u[gidk*numX*numY+gidj*numX+gidi] - 0.5 * d_v[gidk*numX*numY + gidi*numY + gidj];
    }
}

__global__ void implicitY_tridag(const int outer,
                          const int numX,
			              const int numY,
			              REAL* d_a,
                          REAL* d_b,
                          REAL* d_c,
                          REAL* d_y,
                          REAL* d_yy,
                          REAL* d_my_result)
{
    int gidi = blockIdx.x*blockDim.x + threadIdx.x;
    int gidk = blockIdx.y*blockDim.y + threadIdx.y;

    if (gidk < outer && gidi < numX) { 
        const int ind = gidk*numX*numY + gidi*numY;
        tridag(&d_a[ind], &d_b[ind], &d_c[ind],&d_y[ind],
            numY, &d_my_result[ind], &d_yy[ind]);
    }
}


__device__ void tridag(
    const REAL*   a,   // size [n]
    const REAL*   b,   // size [n]
    const REAL*  c,   // size [n]
    const REAL*   r,   // size [n]
    const int             n,
          REAL*   u,   // size [n]
          REAL*   uu   // size [n] temporary
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

    // X) this is a backward recurrence
    u[n-1] = u[n-1] / uu[n-1];
    for(i=n-2; i>=0; i--) {
        u[i] = (u[i] - c[i]*u[i+1]) / uu[i];
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

    // ----- MAIN LOOP ------


    for( unsigned k = 0; k < outer; ++ k ) {
        initGrid(s0,alpha,nu,t, numX, numY, numT, k, globs);
        initOperator(globs.myX,globs.myDxx, globs.sizeX, k, numX);
        initOperator(globs.myY,globs.myDyy, globs.sizeY, k, numY);
    }


  // --- beginning of setPayoff - cuda ----

    REAL *d_payoff, *d_myX, *d_my_result;
    cudaMalloc((void**) &d_payoff, outer * numX * sizeof(REAL));
    cudaMalloc((void**) &d_myX, outer * numX * sizeof(REAL));
    cudaMalloc((void**) &d_my_result, outer * numX * numY * sizeof(REAL));

    cudaMemcpy(d_myX, globs.myX, outer * numX * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_my_result, globs.myResult, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);

    dim3 grid_2 (dim_x, dim_outer, 1);
    initPayoff<<<grid_2, block_2>>>(outer, numX, d_payoff, d_myX);
    cudaDeviceSynchronize();

    dim3 grid_3 (dim_y, dim_x, outer);
    updateGlobsMyResult<<<grid_3, block_2>>>(outer, numX, numY, d_payoff, d_my_result);
    cudaDeviceSynchronize();

    cudaFree(d_payoff);

     // --- end of setPayoff - cuda ----


    REAL *d_myY, *d_myTimeline, *d_myVarX, *d_myVarY, *d_myDxx, *d_myDyy, *d_u, *d_v, *d_dtInv;
    REAL *d_a, *d_b, *d_c, *d_yy, *d_y;
    cudaMalloc((void**) &d_myY, outer * numY * sizeof(REAL));
    cudaMalloc((void**) &d_myTimeline, outer * numT * sizeof(REAL));
    cudaMalloc((void**) &d_myVarX, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_myVarY, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_myDxx,    outer * numX *    4 * sizeof(REAL));
    cudaMalloc((void**) &d_myDyy,    outer * numY *    4 * sizeof(REAL));
    cudaMalloc((void**) &d_u,        outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_v,        outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_dtInv, outer * sizeof(REAL));

    cudaMalloc((void**) &d_a, outer * numZ * numZ * sizeof(REAL));
    cudaMalloc((void**) &d_b, outer * numZ * numZ * sizeof(REAL));
    cudaMalloc((void**) &d_c, outer * numZ * numZ * sizeof(REAL));
    cudaMalloc((void**) &d_yy, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_y, outer * numX * numY * sizeof(REAL));

    cudaMemcpy(d_myX, globs.myX, outer * numX * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myY, globs.myY, outer * numY * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myTimeline, globs.myTimeline, outer * numT * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myVarX, globs.myVarX, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myVarY, globs.myVarY, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myDyy, globs.myDyy, outer * numY * 4 * sizeof(REAL), cudaMemcpyHostToDevice);

    cudaMemcpy(d_myDxx,    globs.myDxx,    outer * numX *    4 * sizeof(REAL), cudaMemcpyHostToDevice);

    for(int g = globs.sizeT-2;g>=0;--g) { // seq


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


	    // --- rollback ---
        unsigned int num_blocks_outer = ((outer + (full_block_size - 1)) / full_block_size);
        rollback<<<num_blocks_outer, full_block_size>>>(outer, numT, g, d_myTimeline, d_dtInv);
        cudaDeviceSynchronize();   


        // ---- explicit x
        explicitX<<<grid_3, block_2>>>(outer,
				     numX,
				     numY,
				     d_dtInv,
				     d_my_result,
				     d_myVarX,
				     d_myDxx,
				     d_u);
       

        cudaDeviceSynchronize();

        // ------ explicit y
        dim3 grid_3_2 (dim_x, dim_y, outer);
        explicitY<<<grid_3_2, block_2>>>(outer,
	        numX, numY, d_my_result, d_myVarY, d_myDyy, d_v, d_u);
        cudaDeviceSynchronize();

        // ------- implicit x
        implicitX<<<grid_3_2, block_2>>>(outer, numX, numY, d_dtInv, d_myVarX, d_myDxx, d_a, d_b, d_c);
        cudaDeviceSynchronize();

        dim3 grid_2_2 (dim_y, dim_outer, 1);
        implicitX_tridag<<<grid_2_2, block_2>>>(outer, numX, numY, d_a, d_b, d_c, d_u, d_yy);
        cudaDeviceSynchronize();


        //	------- implicit y
        implicitY_1<<<grid_3, block_2>>>(outer,numX, numY, d_dtInv, d_myVarY, d_myDyy, d_a, d_b, d_c);
        cudaDeviceSynchronize();

        implicitY_2<<<grid_3, block_2>>>(outer,numX, numY, d_dtInv, d_u, d_v, d_y);
        cudaDeviceSynchronize();

        implicitY_tridag<<<grid_2, block_2>>>(outer,numX, numY, d_a, d_b, 
            d_c, d_y, d_yy, d_my_result);
        cudaDeviceSynchronize();
        
    }

    cudaMemcpy(globs.myResult, d_my_result, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
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
