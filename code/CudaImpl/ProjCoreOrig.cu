#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "TridagPar.cu.h"
#include "CudaKernels.cu.h"


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
    int block_size = 16;

    dim3 block_2(block_size, block_size, 1);

    int  dim_outer = ceil( ((float) outer)/block_size); 
    int  dim_x = ceil( ((float) numX)/block_size );
    int  dim_y = ceil( ((float) numY)/block_size );

    dim3 grid_2 (dim_x, dim_outer, 1);
    dim3 grid_2_2 (dim_y, dim_outer, 1);
    dim3 grid_3 (dim_y, dim_x, outer);
    dim3 grid_3_2 (dim_x, dim_y, outer);

    unsigned int num_blocks_outer = ((outer + (full_block_size - 1)) / full_block_size);

    // ----- ARRAY EXPNASION ------

    PrivGlobsCuda    globsCuda(numX, numY, numT, outer);
    unsigned numZ = max(numX,numY);

    // ----- MAIN LOOP ------

    initGridKernel<<<num_blocks_outer, full_block_size>>>(s0,alpha,nu,t, numX, numY, numT, outer, globsCuda);
    cudaDeviceSynchronize();

    initOperatorKernel<<<num_blocks_outer, full_block_size>>>(globsCuda.myX,globsCuda.myDxx, globsCuda.sizeX, outer, numX);
    initOperatorKernel<<<num_blocks_outer, full_block_size>>>(globsCuda.myY,globsCuda.myDyy, globsCuda.sizeY, outer, numY);
    cudaDeviceSynchronize();

    REAL *d_payoff;
    cudaMalloc((void**) &d_payoff, outer * numX * sizeof(REAL));

    // ---- setPayoff ----
    
    initPayoff<<<grid_2, block_2>>>(outer, numX, d_payoff, globsCuda.myX);
    cudaDeviceSynchronize();

    updateGlobsMyResult<<<grid_3, block_2>>>(outer, numX, numY, d_payoff, globsCuda.myResult);
    cudaDeviceSynchronize();

    cudaFree(d_payoff);

     // --- end of setPayoff ----


    REAL *d_u, *d_v, *d_dtInv, *d_u_T, *d_a, *d_b, *d_c, *d_yy, *d_y;
    cudaMalloc((void**) &d_u,        outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_v,        outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_dtInv, outer * sizeof(REAL));
    cudaMalloc((void**) &d_a, outer * numZ * numZ * sizeof(REAL));
    cudaMalloc((void**) &d_b, outer * numZ * numZ * sizeof(REAL));
    cudaMalloc((void**) &d_c, outer * numZ * numZ * sizeof(REAL));
    cudaMalloc((void**) &d_yy, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**) &d_y, outer * numX * numY * sizeof(REAL));
    
    cudaMalloc((void**) &d_u_T, outer * numX * numY * sizeof(REAL));


    for(int g = globsCuda.sizeT-2;g>=0;--g) { // seq

        // --- updateParams ---      
        updateParams<<<grid_3, block_2>>>(outer, numX, numY, numT, g, alpha,
            beta, nu, globsCuda.myX, globsCuda.myY, globsCuda.myTimeline, globsCuda.myVarX, globsCuda.myVarY);
        cudaDeviceSynchronize();

	    // --- rollback ---
        rollback<<<num_blocks_outer, full_block_size>>>(outer, numT, g, globsCuda.myTimeline, d_dtInv);
        cudaDeviceSynchronize();   

        // ---- explicit x

        // matrix transposition 
        
        // matTransposeTiled<<<numX, numY, outer>>>(d_u, d_u_T, numX, numY, outer);
        // cudaDeviceSynchronize();


        explicitX<<<grid_3, block_2>>>(outer, numX, numY, d_dtInv, globsCuda.myResult,
			globsCuda.myVarX, globsCuda.myDxx, d_u);
        cudaDeviceSynchronize();

        // matTransposeTiled<<<numY, numX, outer>>>(d_u_T, d_u, numY, numX, outer);
        // cudaDeviceSynchronize();


        // ------ explicit y
        explicitY<<<grid_3_2, block_2>>>(outer,
	        numX, numY, globsCuda.myResult, globsCuda.myVarY, globsCuda.myDyy, d_v, d_u);
        // cudaDeviceSynchronize();

        // ------- implicit x
        implicitX<<<grid_3_2, block_2>>>(outer, numX, numY, d_dtInv, globsCuda.myVarX, globsCuda.myDxx, d_a, d_b, d_c);
        cudaDeviceSynchronize();

        implicitX_tridag<<<grid_2_2, block_2>>>(outer, numX, numY, d_a, d_b, d_c, d_u, d_yy);
        cudaDeviceSynchronize();

        //	------- implicit y
        implicitY_1<<<grid_3, block_2>>>(outer,numX, numY, d_dtInv, globsCuda.myVarY, globsCuda.myDyy, d_a, d_b, d_c);
        // cudaDeviceSynchronize();

        implicitY_2<<<grid_3, block_2>>>(outer,numX, numY, d_dtInv, d_u, d_v, d_y);
        cudaDeviceSynchronize();

        implicitY_tridag<<<grid_2, block_2>>>(outer,numX, numY, d_a, d_b, 
            d_c, d_y, d_yy, globsCuda.myResult);
        cudaDeviceSynchronize();
        
    }

    REAL *d_res;
    cudaMalloc((void**) &d_res, outer * sizeof(REAL));

    updateRes<<<num_blocks_outer, full_block_size>>>(outer, numX, numY, globsCuda.myXindex, globsCuda.myYindex, globsCuda.myResult, d_res);
    cudaDeviceSynchronize();

    cudaMemcpy(res, d_res, outer * sizeof(REAL), cudaMemcpyDeviceToHost);

    cudaFree(globsCuda.myX);
    cudaFree(globsCuda.myY);
    cudaFree(globsCuda.myTimeline);
    cudaFree(globsCuda.myVarX);
    cudaFree(globsCuda.myVarY);
    cudaFree(globsCuda.myDxx);
    cudaFree(globsCuda.myDyy);
    cudaFree(globsCuda.myResult);
    cudaFree(globsCuda.myXindex);
    cudaFree(globsCuda.myYindex);
    
    cudaFree(d_dtInv);
    cudaFree(d_res);
    
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_yy);
    cudaFree(d_y);
}

//#endif // PROJ_CORE_ORIG
