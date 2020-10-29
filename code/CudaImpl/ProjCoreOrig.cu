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
