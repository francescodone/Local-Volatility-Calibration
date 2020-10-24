#ifndef SCAN_KERS
#define SCAN_KERS

#include <cuda_runtime.h>

// typedef float REAL;

class MyReal2 {
  public:
    REAL x; REAL y;

    __device__ __host__ inline MyReal2() {
        x = 0.0; y = 0.0; 
    }
    __device__ __host__ inline MyReal2(const REAL& a, const REAL& b) {
        x = a; y = b;
    }
    __device__ __host__ inline MyReal2(const MyReal2& i4) { 
        x = i4.x; y = i4.y;
    }
    volatile __device__ __host__ inline MyReal2& operator=(const MyReal2& i4) volatile {
        x = i4.x; y = i4.y;
        return *this;
    }
    __device__ __host__ inline MyReal2& operator=(const MyReal2& i4) {
        x = i4.x; y = i4.y;
        return *this;
    }
};

class MyReal4 {
  public:
    REAL x; REAL y; REAL z; REAL w;

    __device__ __host__ inline MyReal4() {
        x = 0.0; y = 0.0; z = 0.0; w = 0.0; 
    }
    __device__ __host__ inline MyReal4(const REAL& a, const REAL& b, const REAL& c, const REAL& d) {
        x = a; y = b; z = c; w = d; 
    }
    __device__ __host__ inline MyReal4(const MyReal4& i4) { 
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
    }
    volatile __device__ __host__ inline MyReal4& operator=(const MyReal4& i4) volatile {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
        return *this;
    }
    __device__ __host__ inline MyReal4& operator=(const MyReal4& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
        return *this;
    }
};

class LinFunComp {
  public:
    typedef MyReal2 BaseType;

    static __device__ __host__ inline
    MyReal2 apply(volatile MyReal2& a, volatile MyReal2& b) {
      return MyReal2( b.x + b.y*a.x, a.y*b.y );
    }

    static __device__ __host__ inline 
    MyReal2 identity() { 
      return MyReal2(0.0, 1.0);
    }
};

class MatMult2b2 {
  public:
    typedef MyReal4 BaseType;

    static __device__ __host__ inline
    MyReal4 apply(volatile MyReal4& a, volatile MyReal4& b) {
      REAL val = 1.0/(a.x*b.x);
      return MyReal4( (b.x*a.x + b.y*a.z)*val,
                      (b.x*a.y + b.y*a.w)*val,
                      (b.z*a.x + b.w*a.z)*val,
                      (b.z*a.y + b.w*a.w)*val );
    }

    static __device__ __host__ inline 
    MyReal4 identity() { 
      return MyReal4(1.0,  0.0, 0.0, 1.0);
    }
};

/***************************************/
/*** Scan Inclusive Helpers & Kernel ***/
/***************************************/
template<class OP, class T>
__device__ inline
T scanIncWarp( volatile T* ptr, const unsigned int idx ) {
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)  ptr[idx] = OP::apply(ptr[idx-1],  ptr[idx]); 
    if (lane >= 2)  ptr[idx] = OP::apply(ptr[idx-2],  ptr[idx]);
    if (lane >= 4)  ptr[idx] = OP::apply(ptr[idx-4],  ptr[idx]);
    if (lane >= 8)  ptr[idx] = OP::apply(ptr[idx-8],  ptr[idx]);
    if (lane >= 16) ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]);

    return const_cast<T&>(ptr[idx]);
}

template<class OP, class T>
__device__ inline
T scanIncBlock(volatile T* ptr, const unsigned int idx) {
    const unsigned int lane   = idx &  31;
    const unsigned int warpid = idx >> 5;

    T val = scanIncWarp<OP,T>(ptr,idx);
    __syncthreads();

    // place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and 
    //   max block size = 32^2 = 1024
    if (lane == 31) { ptr[warpid] = const_cast<T&>(ptr[idx]); } 
    __syncthreads();

    //
    if (warpid == 0) scanIncWarp<OP,T>(ptr, idx);
    __syncthreads();

    if (warpid > 0) {
        val = OP::apply(ptr[warpid-1], val);
    }

    return val;
}


/*************************************************/
/*************************************************/
/*** Segmented Inclusive Scan Helpers & Kernel ***/
/*************************************************/
/*************************************************/
template<class OP, class T, class F>
__device__ inline
T sgmScanIncWarp(volatile T* ptr, volatile F* flg, const unsigned int idx) {
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-1], ptr[idx]); }
        flg[idx] = flg[idx-1] | flg[idx];
    }
    if (lane >= 2)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-2], ptr[idx]); }
        flg[idx] = flg[idx-2] | flg[idx];
    }
    if (lane >= 4)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-4], ptr[idx]); }
        flg[idx] = flg[idx-4] | flg[idx];
    }
    if (lane >= 8)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-8], ptr[idx]); }
        flg[idx] = flg[idx-8] | flg[idx];
    }
    if (lane >= 16)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]); }
        flg[idx] = flg[idx-16] | flg[idx];
    }

    return const_cast<T&>(ptr[idx]);
}

template<class OP, class T, class F>
__device__ inline
T sgmScanIncBlock(volatile T* ptr, volatile F* flg, const unsigned int idx) {
    const unsigned int lane   = idx &  31;
    const unsigned int warpid = idx >> 5;
    const unsigned int warplst= (warpid<<5) + 31;

    // 1a: record whether this warp begins with an ``open'' segment.
    bool warp_is_open = (flg[(warpid << 5)] == 0);
    __syncthreads();

    // 1b: intra-warp segmented scan for each warp
    T val = sgmScanIncWarp<OP,T>(ptr,flg,idx);

    // 2a: the last value is the correct partial result
    T warp_total = const_cast<T&>(ptr[warplst]);
    
    // 2b: warp_flag is the OR-reduction of the flags 
    //     in a warp, and is computed indirectly from
    //     the mindex in hd[]
    bool warp_flag = flg[warplst]!=0 || !warp_is_open;
    bool will_accum= warp_is_open && (flg[idx] == 0);

    __syncthreads();

    // 2c: the last thread in a warp writes partial results
    //     in the first warp. Note that all fit in the first
    //     warp because warp = 32 and max block size is 32^2
    if (lane == 31) {
        ptr[warpid] = warp_total; //ptr[idx]; 
        flg[warpid] = warp_flag;
    }
    __syncthreads();

    // 
    if (warpid == 0) sgmScanIncWarp<OP,T>(ptr, flg, idx);
    __syncthreads();

    if (warpid > 0 && will_accum) {
        val = OP::apply(ptr[warpid-1], val);
    }
    return val;
}

/*********************/
/*** Tridag Kernel ***/
/*********************/
// Try to optimize it: for example,
//    (The allocated shared memory is enough for 8 floats / thread): 
//    1. the shared memory space for "mat_sh" can be reused for "lin_sh"
//    2. with 1., now you have space to hold "u" and "uu" in shared memory.
//    3. you may hold "a[gid]" in a register, since it is accessed twice, etc.
__global__ void 
TRIDAG_SOLVER(  REAL* a,
                REAL* b,
                REAL* c,
                REAL* r,
                const unsigned int n,
                const unsigned int sgm_sz,
                REAL* u,
                REAL* uu
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + tid;

    // total shared memory (declared outside)
    extern __shared__ char sh_mem[];
    // shared memory space for the 2x2 matrix multiplication SCAN
    volatile MyReal4* mat_sh = (volatile MyReal4*)sh_mem;
    // shared memory space for the linear-function composition SCAN
    volatile MyReal2* lin_sh = (volatile MyReal2*) (mat_sh + blockDim.x);
    // shared memory space for the flag array
    volatile int*     flg_sh = (volatile int*    ) (lin_sh + blockDim.x);
    
    // make the flag array
    flg_sh[tid] = (tid % sgm_sz == 0) ? 1 : 0;
    __syncthreads();
    
    //--------------------------------------------------
    // Recurrence 1: b[i] = b[i] - a[i]*c[i-1]/b[i-1] --
    //   solved by scan with 2x2 matrix mult operator --
    //--------------------------------------------------
    // 1.a) first map
    const unsigned int beg_seg_ind = (gid / sgm_sz) * sgm_sz;
    REAL b0 = (gid < n) ? b[beg_seg_ind] : 1.0;
    mat_sh[tid] = (gid!=beg_seg_ind && gid < n) ?
                    MyReal4(b[gid], -a[gid]*c[gid-1], 1.0, 0.0) :
                    MyReal4(1.0,                 0.0, 0.0, 1.0) ;
    // 1.b) inplaceScanInc<MatMult2b2>(n,mats);
    __syncthreads();
    MyReal4 res4 = sgmScanIncBlock <MatMult2b2, MyReal4, int>(mat_sh, flg_sh, tid);
    // 1.c) second map
    if(gid < n) {
        uu[gid] = (res4.x*b0 + res4.y) / (res4.z*b0 + res4.w) ;
    }
    __syncthreads();

    // make the flag array
    flg_sh[tid] = (tid % sgm_sz == 0) ? 1 : 0;
    __syncthreads();

    //----------------------------------------------------
    // Recurrence 2: y[i] = y[i] - (a[i]/b[i-1])*y[i-1] --
    //   solved by scan with linear func comp operator  --
    //----------------------------------------------------
    // 2.a) first map
    REAL y0 = (gid < n) ? r[beg_seg_ind] : 1.0;
    lin_sh[tid] = (gid!=beg_seg_ind && gid < n) ?
                    MyReal2(r[gid], -a[gid]/uu[gid-1]) :
                    MyReal2(0.0,    1.0              ) ;
    // 2.b) inplaceScanInc<LinFunComp>(n,lfuns);
    __syncthreads();
    MyReal2 res2 = sgmScanIncBlock <LinFunComp, MyReal2, int>(lin_sh, flg_sh, tid);
    // 2.c) second map
    if(gid < n) {
        u[gid] = res2.x + y0*res2.y;
    }
    __syncthreads();

    // make the flag array
    flg_sh[tid] = (tid % sgm_sz == 0) ? 1 : 0;
    __syncthreads();
#if 1
    //----------------------------------------------------
    // Recurrence 3: backward recurrence solved via     --
    //             scan with linear func comp operator  --
    //----------------------------------------------------
    // 3.a) first map
    const unsigned int end_seg_ind = (beg_seg_ind + sgm_sz) - 1;
    const unsigned int k = (end_seg_ind - gid) + beg_seg_ind ;  
    REAL yn = u[end_seg_ind] / uu[end_seg_ind];
    lin_sh[tid] = (gid!=beg_seg_ind && gid < n) ?
                    MyReal2( u[k]/uu[k], -c[k]/uu[k] ) :
                    MyReal2( 0.0,        1.0         ) ;
    // 3.b) inplaceScanInc<LinFunComp>(n,lfuns);
    __syncthreads();
    MyReal2 res3 = sgmScanIncBlock <LinFunComp, MyReal2, int>(lin_sh, flg_sh, tid);
    __syncthreads();
    // 3.c) second map
    if(gid < n) {
        u[k] = res3.x + yn*res3.y;
    }
#endif
}

#endif //SCAN_KERS

