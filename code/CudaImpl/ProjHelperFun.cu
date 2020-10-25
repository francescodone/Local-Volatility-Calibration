#include "ProjHelperFun.cu.h"

/**************************/
/**** HELPER FUNCTIONS ****/
/**************************/

/**
 * Fills in:
 *   globs.myTimeline  of size [0..numT-1]
 *   globs.myX         of size [0..numX-1]
 *   globs.myY         of size [0..numY-1]
 * and also sets
 *   globs.myXindex and globs.myYindex (both scalars)
 */
void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const unsigned numX, const unsigned numY, const unsigned numT, const unsigned ido, PrivGlobs& globs   
) {
    for(unsigned i=0;i<numT;++i)
        globs.myTimeline[ido][i] = t*i/(numT-1);

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    globs.myXindex[ido] = static_cast<unsigned>(s0/dx) % numX;

    for(unsigned i=0;i<numX;++i)
        globs.myX[ido][i] = i*dx - globs.myXindex[ido]*dx + s0;

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    globs.myYindex[ido] = static_cast<unsigned>(numY/2.0);

    for(unsigned i=0;i<numY;++i)
        globs.myY[ido][i] = i*dy - globs.myYindex[ido]*dy + logAlpha;
}

/**
 * Fills in:
 *    Dx  [0..n-1][0..3] and 
 *    Dxx [0..n-1][0..3] 
 * Based on the values of x, 
 * Where x's size is n.
 */
void initOperator(  REAL** x,
                    REAL*** Dxx,
                    unsigned xsize,
                    unsigned k
) {
	const unsigned n = xsize;

	REAL dxl, dxu;

	//	lower boundary
	dxl		 =  0.0;
	dxu		 =  x[k][1] - x[k][0];
	
	Dxx[k][0][0] =  0.0;
	Dxx[k][0][1] =  0.0;
	Dxx[k][0][2] =  0.0;
        Dxx[k][0][3] =  0.0;
	
	//	standard case
	for(unsigned i=1;i<n-1;i++)
	{
		dxl      = x[k][i]   - x[k][i-1];
		dxu      = x[k][i+1] - x[k][i];

		Dxx[k][i][0] =  2.0/dxl/(dxl+dxu);
		Dxx[k][i][1] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
		Dxx[k][i][2] =  2.0/dxu/(dxl+dxu);
        Dxx[k][i][3] =  0.0;
	}

	//	upper boundary
	dxl		   =  x[k][n-1] - x[k][n-2];
	dxu		   =  0.0;

	Dxx[k][n-1][0] = 0.0;
	Dxx[k][n-1][1] = 0.0;
	Dxx[k][n-1][2] = 0.0;
        Dxx[k][n-1][3] = 0.0;
}




