#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "TridagPar.cu.h"
#include <sys/time.h>

int timeval_subtract_2(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}



__global__ void updateParamsKernel(REAL* d_myVarX, REAL* d_myVarY, 
    REAL* d_myX, REAL* d_myY, REAL mult, REAL alpha, REAL beta, const int numX, const int numY)
{
    int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    int gidy = blockIdx.y*blockDim.y + threadIdx.y;

    if (gidx < numX && gidy < numY) {
        d_myVarX[gidx][gidy] = exp(2.0*(beta*log(d_myX[gidx]) + d_myY[gidy]-mult));
        d_myVarY[gidx][gidy] = exp(2.0*(alpha*log(d_myX[gidx]) + d_myY[gidy] - mult)); 
    } 
}

__global__ void updateParamsCuda(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{

    const int numX = globs.myX.size();
    const int numY = globs.myY.size();

    dim3 block(numX, numY, 1);
    dim3 grid (1, 1, 1);
    
    vector<REAL> d_myX;
    vector<REAL> d_myY;
    REAL* d_myVarX;
    REAL* d_myVarY;

    unsigned int mem_size_my_x = numX * sizeof(REAL);
    unsigned int mem_size_my_y = numY * sizeof(REAL);
    unsigned int mem_size_my_var = numX * numY * sizeof(REAL);


    cudaMalloc(&d_myX,  mem_size_my_x);
    cudaMalloc(&d_myY,  mem_size_my_y);
    cudaMalloc((void**)&d_myVarX,  mem_size_my_var);
    cudaMalloc((void**)&d_myVarY,  mem_size_my_var);

    cudaMemcpy(d_myX, globs.myX, mem_size_my_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_myY, globs.myY, mem_size_my_y, cudaMemcpyHostToDevice);

    const int mult = 0.5*nu*nu*globs.myTimeline[g];

    updateParamsKernel<<< grid, block >>>(d_myVarX, d_myVarY, d_myX, d_myY, mult, alpha, beta, numX, numY);
    cudaDeviceSynchronize();

    cudaMemcpy(globs.myVarX, d_myVarX, mem_size_my_var, cudaMemcpyDeviceToHost);
    cudaMemcpy(globs.myVarY, d_myVarY, mem_size_my_var, cudaMemcpyDeviceToHost);

}

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{
    unsigned long int elapsed = 0;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    
    #pragma omp parallel for collapse(2)
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

    gettimeofday(&t_end, NULL);
    timeval_subtract_2(&t_diff, &t_end, &t_start);
    elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;
    fprintf(stdout, "%ld\t\t// updateParams Runtime in microseconds,\n", elapsed);
}

void setPayoff(const REAL strike, PrivGlobs& globs )
{
	
    REAL payoff[globs.myX.size()];

    #pragma omp parallel for default(shared) schedule(static)
    for(unsigned i=0;i<globs.myX.size();++i) 
    {
        payoff[i] = max(globs.myX[i]-strike, (REAL)0.0);
    }


    #pragma omp parallel for collapse(2)
    for(unsigned i=0;i<globs.myX.size();++i)
	{
		for(unsigned j=0;j<globs.myY.size();++j)
			globs.myResult[i][j] = payoff[i];
	}
}

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


void
rollback( const unsigned g, PrivGlobs& globs ) {
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();

    unsigned numZ = max(numX,numY);

    unsigned i, j;

    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
    vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
    vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
    vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

    //	explicit x
    #pragma omp parallel for collapse(2)
    for(i=0;i<numX;i++) {
        for(j=0;j<numY;j++) {
            u[j][i] = dtInv*globs.myResult[i][j];

            if(i > 0) { 
              u[j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][0] ) 
                            * globs.myResult[i-1][j];
            }
            u[j][i]  +=  0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][1] )
                            * globs.myResult[i][j];
            if(i < numX-1) {
              u[j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][2] )
                            * globs.myResult[i+1][j];
            }
        }
    }

    //	explicit y
    #pragma omp parallel for collapse(2)
    for(j=0;j<numY;j++)
    {
        for(i=0;i<numX;i++) {
            v[i][j] = 0.0;

            if(j > 0) {
              v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][0] )
                         *  globs.myResult[i][j-1];
            }
            v[i][j]  +=   ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][1] )
                         *  globs.myResult[i][j];
            if(j < numY-1) {
              v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][2] )
                         *  globs.myResult[i][j+1];
            }
            u[j][i] += v[i][j]; 
        }
    }


    vector<vector<REAL> > a_cp(numY, vector<REAL>(numX));
    vector<vector<REAL> > b_cp(numY, vector<REAL>(numX));
    vector<vector<REAL> > c_cp(numY, vector<REAL>(numX));

    //	implicit x
    #pragma omp parallel for collapse(2)
    for(j=0;j<numY;j++) {
        for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
            a_cp[j][i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][0]);
            b_cp[j][i] = dtInv - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][1]);
            c_cp[j][i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][2]);
        }
    }

    #pragma omp parallel for default(shared) schedule(static)
    for(j=0;j<numY;j++) {
        // here yy should have size [numX]
        vector<REAL> yy(numZ);
        tridagPar(a_cp[j],b_cp[j],c_cp[j],u[j],numX,u[j],yy);
    }



    //	implicit y
    for(i=0;i<numX;i++) { 
        for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
            a[j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][0]);
            b[j] = dtInv - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][1]);
            c[j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][2]);
        }

        for(j=0;j<numY;j++)
            y[j] = dtInv*u[j][i] - 0.5*v[i][j];

        // here yy should have size [numY]
        tridagPar(a,b,c,y,numY,globs.myResult[i],yy);
    }
}

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
    #pragma omp parallel for default(shared) schedule(static)
    for( unsigned i = 0; i < outer; ++ i ) {
        REAL strike;
        PrivGlobs    globs(numX, numY, numT);

        strike = 0.001*i;
        res[i] = value( globs, s0, strike, t,
                        alpha, nu,    beta,
                        numX,  numY,  numT );
    }
}

//#endif // PROJ_CORE_ORIG
