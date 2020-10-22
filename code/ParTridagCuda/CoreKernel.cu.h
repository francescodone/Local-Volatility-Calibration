#ifndef CORE_KERS
#define CORE_KERS

#inclide <cuda_runtime.h>
#include "../OrigImpl/ProjHelperFun.h"
#include "Constants.h"

__global__
void setPayoff(const float strike, PrivGlobs& globs)
{
  auto idx = threadIdx.x;
  if (idx > globs.myX.size())
    return;
  float payoff = max(globs.myX[idx]-strike, (float)0.0);
  for(unsigned j=0;j<globs.myY.size();++j)
    globs.myResult[idx][j] = payoff;
}

__global__
void setPayoff(const float strike, PrivGlobs& globs)
{
  auto idx = threadIdx.x;
  auto idy = threadIdx.y;
  
  if (idx > globs.myX.size() || idy > globs.myY.size())
    return;
  
  float payoff = max(globs.myX[idx]-strike, (float)0.0);
  globs.myResult[idx][idy] = payoff;
}

#endif //CORE_KERS
