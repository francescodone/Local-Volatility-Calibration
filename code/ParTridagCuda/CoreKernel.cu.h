#ifndef CORE_KERS
#define CORE_KERS

#include <cuda_runtime.h>
#include "ProjHelperFun.h"
#include "Constants.h"

__global__
void setPayoff1(const float strike, REAL* myX, REAL* myY, REAL* myResult, int xmax, int ymax)
{
  int idx = threadIdx.x;
  if (idx > xmax)
    return;
  float payoff = max(myX[idx]-strike, (float)0.0);
  for(unsigned j=0;j<ymax;++j)
    myResult[idx*xmax+j] = payoff;
}

__global__
void setPayoff2(const float strike, float* myX, float* myY, float* myResult, int xmax, int ymax)
{
  int idx = threadIdx.x;
  int idy = threadIdx.y;

  //if (idx > globs.myX.size() || idy > globs.myY.size())
  if (idx > xmax || idy > ymax)
    return;

  float payoff = max(myX[idx]-strike, (float)0.0);
  myResult[idx*xmax+idy] = payoff;
}

 

#endif //CORE_KERS
