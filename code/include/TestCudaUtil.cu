#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>
#include <time.h> 

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}


#include "CudaUtilProj.cu.h" 


__global__ void 
trivial_map(int* inp_d, MyInt4* inp_lift, int inp_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < inp_size) {
        int el = inp_d[gid];
        MyInt4 res(el,el,el,el);
        if(el < 0) { res.x = 0;  res.y = 0;  res.z = 0; }
        inp_lift[gid] = res;
    }
}


int MsspProblem(int block_size, int inp_size) {
    int  mem_size = inp_size*sizeof(MyInt4);
    int    *inp_h = (int*)malloc(inp_size*sizeof(int));
    int    *inp_d;    cudaMalloc((void**)&inp_d   , inp_size*sizeof(int));
    MyInt4 *inp_lift; cudaMalloc((void**)&inp_lift, mem_size);
    MyInt4 *res_d;    cudaMalloc((void**)&res_d,    mem_size);

    for(int i = 0; i < inp_size; i+=9) {
        inp_h[i   ] = -15;
        inp_h[i+1 ] = 2;
        inp_h[i+2 ] = -1;
        inp_h[i+3 ] = 3;
        inp_h[i+4 ] = -2;
        inp_h[i+5 ] = 4;
        inp_h[i+6 ] = 3;
        inp_h[i+7 ] = 1;
        inp_h[i+8 ] = -4;
    }
    inp_h[(inp_size/9)*9 - 18 + 7] = 2;

    int num_blocks = ( (inp_size % block_size) == 0) ?
                        inp_size / block_size     :
                        inp_size / block_size + 1 ;

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    cudaMemcpy(inp_d, inp_h, inp_size*sizeof(int), cudaMemcpyHostToDevice);
    { // KERNELS
        // 1. apply map, i.e., lift each element x to 
        //    (max(x,0),max(x,0), max(x,0), x)
        trivial_map<<< num_blocks, block_size >>>(inp_d, inp_lift, inp_size);
        cudaThreadSynchronize();
    
        // 2. apply scan with the given operator, i.e.,
        //    write the apply operator in class MsspOP in 
        //    ScanKernels.cu.h and call scanInc from ScanHost.cu.h  
        scanInc< MsspOp,MyInt4 > ( block_size, inp_size, inp_lift, res_d );
        cudaThreadSynchronize();
    }
    MyInt4 res_h(0,0,0,0);
    // 3. copy back only the last element of the res_d array (of size sizeof(MyInt4))
    cudaMemcpy(&res_h, res_d+inp_size-1, sizeof(MyInt4), cudaMemcpyDeviceToHost);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("MSSP version runs in: %lu microsecs\n", elapsed);

    printf("RESULT is: %d %d %d %d\n", res_h.x, res_h.y, res_h.z, res_h.w); 

    if(res_h.x == 11) {
        printf("MSSP VALID EXECUTION!\n");
    } else {
        printf("MSSP INVALID EXECUTION!\n");
    }

    free(inp_h);
    cudaFree(inp_d);
    cudaFree(inp_lift);
    cudaFree(res_d);

    return 1;
}





int scanIncTest() {
    const unsigned int num_threads = 8353455;
    const unsigned int block_size  = 512;
    unsigned int mem_size = num_threads * sizeof(int);

    int* h_in    = (int*) malloc(mem_size);
    int* h_out   = (int*) malloc(mem_size);
    int* flags_h = (int*) malloc(num_threads*sizeof(int));

    int sgm_size = 123;
    { // init segments and flags
        for(unsigned int i=0; i<num_threads; i++) {
            h_in   [i] = 1; 
            flags_h[i] = (i % sgm_size == 0) ? 1 : 0;
        }
    }

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 


    { // calling exclusive (segmented) scan
        int* d_in;
        int* d_out;
        int* flags_d;
        cudaMalloc((void**)&d_in ,   mem_size);
        cudaMalloc((void**)&d_out,   mem_size);
        cudaMalloc((void**)&flags_d, num_threads*sizeof(int));

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(flags_d, flags_h, num_threads*sizeof(int), cudaMemcpyHostToDevice);

        // execute kernel
        sgmScanInc< Add<int>,int > ( block_size, num_threads, d_in, flags_d, d_out );

        // copy host memory to device
        cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_in );
        cudaFree(d_out);
        cudaFree(flags_d);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("Scan Inclusive on GPU runs in: %lu microsecs\n", elapsed);


    { // validation
        bool  success = true;
        int   accum   = 0;
        for(int i=0; i<num_threads; i++) {
            // for segmented scan exclusive test
            if (i % sgm_size == 0) accum  = 1;
            else                   accum += 1;
            
            if ( accum != h_out[i] ) { 
                success = false;
                printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
            // for scan exclusive test
//            accum += 1;
        }
        if(success) printf("\nScan Exclusive +   VALID RESULT!\n");
        else        printf("\nScan Exclusive + INVALID RESULT!\n");
    }

    // cleanup memory
    free(h_in );
    free(h_out);
    free(flags_h);

    return 0;
}


void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}


template<class T>
bool validateTranspose(float* A,float* trA, unsigned int rowsA, unsigned int colsA){
  bool valid = true;
  for(unsigned int i = 0; i < rowsA; i++) {
    for(unsigned int j = 0; j < colsA; j++) {
      if(trA[j*rowsA + i] != A[i*colsA + j]) {
        printf("row: %d, col: %d, A: %.4f, trA: %.4f\n", 
                i, j, A[i*colsA + j], trA[j*rowsA + i] );
        valid = false;
        break;
      }
    }
    if(!valid) break;
  }
  if (valid) printf("GPU TRANSPOSITION   VALID!\n");
  else       printf("GPU TRANSPOSITION INVALID!\n");
  return valid;
}


void testTranspose() {
   // set seed for rand()
   srand(2006); 
 
    const unsigned int HEIGHT_A = 1024*8;
    const unsigned int WIDTH_A  = 1024*8;

   // 1. allocate host memory for the two matrices
   size_t size_A = WIDTH_A * HEIGHT_A;
   size_t mem_size_A = sizeof(float) * size_A;
   float* h_A = (float*) malloc(mem_size_A);
   float* h_B = (float*) malloc(mem_size_A);
 
   // 2. initialize host memory
   randomInit(h_A, size_A);
    
   // 3. allocate device memory
   float* d_A;
   float* d_B;
   cudaMalloc((void**) &d_A, mem_size_A);
   cudaMalloc((void**) &d_B, mem_size_A);
 
   // 4. copy host memory to device
   cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

   {
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        //transpose<float, TILE>( d_A, d_B, HEIGHT_A, WIDTH_A );
        transpose<float, 32>( d_A, d_B, HEIGHT_A, WIDTH_A );

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("Transpose on GPU runs in: %lu microsecs\n", elapsed);

        // copy result from device to host
        cudaMemcpy(h_B, d_B, mem_size_A, cudaMemcpyDeviceToHost);
  
        // 12. validate
        //validateTranspose<float>( h_A, h_B, HEIGHT_A, WIDTH_A );
        validateTranspose<float>( h_A, h_B, HEIGHT_A, WIDTH_A );
   }


   // clean up memory
   free(h_A);
   free(h_B);
   cudaFree(d_A);
   cudaFree(d_B);
}



int main(int argc, char** argv) { 
    const unsigned int mssp_list_size = 8353455; 
    const unsigned int block_size     = 256;

    scanIncTest();
    MsspProblem(block_size, mssp_list_size);
    testTranspose();
}
