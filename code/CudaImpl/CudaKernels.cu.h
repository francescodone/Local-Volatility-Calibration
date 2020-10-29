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
        //const int d_ind = gidk*numX*numY+gidi*numY+gidj;
        const int d_ind = gidk*numX*numY+gidj*numX+gidi;
        d_u[d_ind] = 
            d_dtInv[gidk]*d_myResult[gidk*numX*numY + gidi*numY + gidj];

        if(gidi > 0) { 
            d_u[d_ind] += 
                0.5*( 0.5*d_myVarX[gidk*numX*numY+gidi*numY+gidj]*d_myDxx[gidk*numX*4+gidi*4+0] ) 
                * d_myResult[gidk*numX*numY + (gidi-1)*numY + gidj];
        }

        d_u[d_ind] += 
            0.5*( 0.5*d_myVarX[gidk*numX*numY+gidi*numY+gidj]*d_myDxx[gidk*numX*4+gidi*4+1] )
            * d_myResult[gidk*numX*numY + gidi*numY + gidj];

        if(gidi < numX-1) {
            d_u[d_ind] += 
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


__global__ void updateRes(const int outer,
                          const int numX,
			              const int numY,
                          const unsigned* d_myXindex,
                          const unsigned* d_myYindex,
                          const REAL* d_my_result,
			              REAL* d_res)                     
{
    int gidk = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidk < outer) { 
        int ind = gidk*numX*numY + d_myXindex[gidk]*numY + d_myYindex[gidk];
        d_res[gidk] = d_my_result[ind];
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