#include "ProjHelperFun.h"
#include "Constants.h"
#include "TridagPar.h"
/*
void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{
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
	for(unsigned i=0;i<globs.myX.size();++i)
	{
		REAL payoff = max(globs.myX[i]-strike, (REAL)0.0);
		for(unsigned j=0;j<globs.myY.size();++j)
			globs.myResult[i][j] = payoff;
	}
}
*/
/*
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
*/

/*
void
rollback( const unsigned g, const unsigned k, PrivGlobs& globs ) {
    unsigned numX = globs.sizeX,
             numY = globs.sizeY;

    unsigned numZ = max(numX,numY);

    unsigned i, j;

    REAL dtInv = 1.0/(globs.myTimeline[k][g+1]-globs.myTimeline[k][g]);

    REAL** u = new REAL*[numY];
    for (int idx = 0; idx<numY; idx++) {
        u[idx] = new REAL[numX];
    }
    //(numY, vector<REAL>(numX));   // [numY][numX]

    REAL** v = new REAL*[numX];
    for (int idx = 0; idx<numX; idx++) {
        v[idx] = new REAL[numY];
    }
    //vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
    REAL* a = new REAL[numZ];
    REAL* b = new REAL[numZ];
    REAL* c = new REAL[numZ];
    REAL* y = new REAL[numZ];
    REAL* yy = new REAL[numZ];
    //vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
    //vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

    //	explicit x
    for(i=0;i<numX;i++) {
        for(j=0;j<numY;j++) {
            u[j][i] = dtInv*globs.myResult[k][i][j];

            if(i > 0) { 
              u[j][i] += 0.5*( 0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][0] ) 
                            * globs.myResult[k][i-1][j];
            }
            u[j][i]  +=  0.5*( 0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][1] )
                            * globs.myResult[k][i][j];
            if(i < numX-1) {
              u[j][i] += 0.5*( 0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][2] )
                            * globs.myResult[k][i+1][j];
            }
        }
    }

    //	explicit y
    for(j=0;j<numY;j++)
    {
        for(i=0;i<numX;i++) {
            v[i][j] = 0.0;

            if(j > 0) {
              v[i][j] +=  ( 0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][0] )
                         *  globs.myResult[k][i][j-1];
            }
            v[i][j]  +=   ( 0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][1] )
                         *  globs.myResult[k][i][j];
            if(j < numY-1) {
              v[i][j] +=  ( 0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][2] )
                         *  globs.myResult[k][i][j+1];
            }
            u[j][i] += v[i][j]; 
        }
    }

    //	implicit x
    for(j=0;j<numY;j++) {
        for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
            a[i] =		 - 0.5*(0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][0]);
            b[i] = dtInv - 0.5*(0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][1]);
            c[i] =		 - 0.5*(0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][2]);
        }
        // here yy should have size [numX]
        tridagPar(a,b,c,u[j],numX,u[j],yy);
    }

    //	implicit y
    for(i=0;i<numX;i++) { 
        for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
            a[j] =		 - 0.5*(0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][0]);
            b[j] = dtInv - 0.5*(0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][1]);
            c[j] =		 - 0.5*(0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][2]);
        }

        for(j=0;j<numY;j++)
            y[j] = dtInv*u[j][i] - 0.5*v[i][j];

        // here yy should have size [numY]
        tridagPar(a,b,c,y,numY,globs.myResult[k][i],yy);
    }
}
*/

/*
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
*/

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

    // ----- ARRAY EXPNASION ------

    PrivGlobs    globs(numX, numY, numT, outer);
    unsigned numZ = max(numX,numY);

    REAL* strike = new REAL[outer];
    REAL* dtInv = new REAL[outer];

    REAL*** u = new REAL**[outer];
    REAL*** v = new REAL**[outer];
    
    for(int k=0; k<outer; k++) {
        u[k] = new REAL*[numY];
        for (int idx = 0; idx<numY; idx++) {
            u[k][idx] = new REAL[numX];
        }
        v[k] = new REAL*[numX];
        for (int idx = 0; idx<numX; idx++) {
            v[k][idx] = new REAL[numY];
        }
    }

    REAL** payoff = new REAL*[outer];
    REAL** a = new REAL*[outer];
    REAL** b = new REAL*[outer];
    REAL** c = new REAL*[outer];
    REAL** y = new REAL*[outer];
    REAL** yy = new REAL*[outer];

    for(int k=0; k<outer; k++) {
        payoff[k] = new REAL[numX];
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
      initOperator(globs.myX,globs.myDxx, globs.sizeX, k);
      initOperator(globs.myY,globs.myDyy, globs.sizeY, k);
    }

    // --- setPayoff ----

    for( unsigned k = 0; k < outer; ++ k ) {
        for(unsigned i=0;i<globs.sizeX;++i) {
            payoff[k][i] = max(globs.myX[k][i]-strike[k], (REAL)0.0);
        }
    }

    for( unsigned k = 0; k < outer; ++ k ) {
        for(unsigned i=0;i<globs.sizeX;++i) {
            for(unsigned j=0;j<globs.sizeY;++j)
                globs.myResult[k][i][j] = payoff[k][i];
        }
    }


    for(int g = globs.sizeT-2;g>=0;--g) { // seq


        // --- updateParams ---    
        for( unsigned k = 0; k < outer; ++ k ) {
            for(unsigned i=0;i<globs.sizeX;++i) {
                for(unsigned j=0;j<globs.sizeY;++j) {
                    globs.myVarX[k][i][j] = exp(2.0*(  beta*log(globs.myX[k][i])
                                                     + globs.myY[k][j]
                                                     - 0.5*nu*nu*globs.myTimeline[k][g] )
                                          );
                    globs.myVarY[k][i][j] = exp(2.0*(  alpha*log(globs.myX[k][i])
                                                     + globs.myY[k][j]
                                                     - 0.5*nu*nu*globs.myTimeline[k][g] )
                                          ); // nu*nu
                }
            }
        }

        // --- rollback ---

        for( unsigned k = 0; k < outer; ++ k ) {
            dtInv[k] = 1.0/(globs.myTimeline[k][g+1]-globs.myTimeline[k][g]);
        }

        //	explicit x
        // do matrix transposition for u (after kernel is executed)
        for( unsigned k = 0; k < outer; ++ k ) {
            for(unsigned i=0;i<numX;i++) {
                for(unsigned j=0;j<numY;j++) {
                    u[k][j][i] = dtInv[k]*globs.myResult[k][i][j];

                    if(i > 0) { 
                    u[k][j][i] += 0.5*( 0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][0] ) 
                                    * globs.myResult[k][i-1][j];
                    }
                    u[k][j][i]  +=  0.5*( 0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][1] )
                                    * globs.myResult[k][i][j];
                    if(i < numX-1) {
                        u[k][j][i] += 0.5*( 0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][2] )
                                        * globs.myResult[k][i+1][j];
                    }
                }
            }
        }

        //	explicit y
        // matrix transposition and/or loop interchange?
        for( unsigned k = 0; k < outer; ++ k ) {
            for(unsigned j=0;j<numY;j++) {
                for(unsigned i=0;i<numX;i++) {
                    v[k][i][j] = 0.0;

                    if(j > 0) {
                    v[k][i][j] +=  ( 0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][0] )
                                *  globs.myResult[k][i][j-1];
                    }
                    v[k][i][j]  +=   ( 0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][1] )
                                *  globs.myResult[k][i][j];
                    if(j < numY-1) {
                    v[k][i][j] +=  ( 0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][2] )
                                *  globs.myResult[k][i][j+1];
                    }
                    u[k][j][i] += v[k][i][j]; 
                }
            }
        }

        //	implicit x
        for( unsigned k = 0; k < outer; ++ k ) {
            for(unsigned j=0;j<numY;j++) {
                for(unsigned i=0;i<numX;i++) {  // here a, b,c should have size [numX]
                    a[k][i] =		 - 0.5*(0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][0]);
                    b[k][i] = dtInv[k] - 0.5*(0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][1]);
                    c[k][i] =		 - 0.5*(0.5*globs.myVarX[k][i][j]*globs.myDxx[k][i][2]);
                }
                // here yy should have size [numX]
                tridagPar(a[k],b[k],c[k],u[k][j],numX,u[k][j],yy[k]);
            }
        }

        //	implicit y
        for( unsigned k = 0; k < outer; ++ k ) {
            for(unsigned i=0;i<numX;i++) { 
                for(unsigned j=0;j<numY;j++) {  // here a, b, c should have size [numY]
                    a[k][j] =		 - 0.5*(0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][0]);
                    b[k][j] = dtInv[k] - 0.5*(0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][1]);
                    c[k][j] =		 - 0.5*(0.5*globs.myVarY[k][i][j]*globs.myDyy[k][j][2]);
                }
                for(unsigned j=0;j<numY;j++)
                    y[k][j] = dtInv[k]*u[k][j][i] - 0.5*v[k][i][j];

                // here yy should have size [numY]
                tridagPar(a[k],b[k],c[k],y[k],numY,globs.myResult[k][i],yy[k]);
            }
        }
        
    }

    for( unsigned k = 0; k < outer; ++ k ) { 
      res[k] = globs.myResult[k][globs.myXindex[k]][globs.myYindex[k]];
    }

}

//#endif // PROJ_CORE_ORIG
