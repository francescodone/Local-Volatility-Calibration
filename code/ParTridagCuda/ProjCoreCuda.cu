#include <cuda_runtime.h>
#include "TridagKernel.cu.h"
#include "CoreKernel.cu.h"
#include "../OrigImpl/ProjHelperFun.h"
#include "Constants.h"
#include "TridagPar.h"

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{
#pragma omp parallel for collapse(2) default(shared) schedule(static)
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
    vector<REAL> payoffs(globs.myX.size());
    #pragma omp parallel for default(shared) schedule(static)
    for(unsigned i=0;i<globs.myX.size();++i) {
        payoffs[i] = max(globs.myX[i]-strike, (REAL)0.0);
    }

    #pragma omp parallel for collapse(2) default(shared) schedule(static)
    for(unsigned i=0;i<globs.myX.size();++i)
    {
        for(unsigned j=0;j<globs.myY.size();++j)
            globs.myResult[i][j] = payoffs[i];
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

    //	explicit x
    #pragma omp parallel for collapse(2) default(shared) schedule(static)
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
    #pragma omp parallel for collapse(2) default(shared) schedule(static)
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

    vector< vector<REAL> > as1(numY, vector<REAL>(numX));
    vector< vector<REAL> > bs1(numY, vector<REAL>(numX));
    vector< vector<REAL> > cs1(numY, vector<REAL>(numX));
    #pragma omp parallel for collapse(2) default(shared) schedule(static)
    for(j=0;j<numY;j++) {
        for(i=0;i<numX;i++) {
            as1[j][i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][0]);
            bs1[j][i] = dtInv - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][1]);
            cs1[j][i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][2]);            
        }
    }

    //	implicit x
    #pragma omp parallel for default(shared) schedule(static)
    for(j=0;j<numY;j++) {
        // here yy should have size [numX]
        vector<REAL> yy(numX);
        tridagPar(as1[j],bs1[j],cs1[j],u[j],numX,u[j],yy);
    }

    vector< vector<REAL> > as2(numX, vector<REAL>(numY));
    vector< vector<REAL> > bs2(numX, vector<REAL>(numY));
    vector< vector<REAL> > cs2(numX, vector<REAL>(numY));
    vector< vector<REAL> > ys2(numX, vector<REAL>(numY));
    #pragma omp parallel for collapse(2) default(shared) schedule(static)
    for(i=0;i<numX;i++) { 
        for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
            as2[i][j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][0]);
            bs2[i][j] = dtInv - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][1]);
            cs2[i][j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][2]);
            ys2[i][j] = dtInv*u[j][i] - 0.5*v[i][j];
        }
    }
    //	implicit y
    #pragma omp parallel for collapse(1) default(shared) schedule(static)
    for(i=0;i<numX;i++) { 
        // here yy should have size [numY]
        vector<REAL> yy(numY);
        tridagPar(as2[i],bs2[i],cs2[i],ys2[i],numY,globs.myResult[i],yy);
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
    #pragma omp parallel for default(shared) schedule(static) if(outer>8)
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
