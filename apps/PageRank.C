// This code is part of the project "Ligra: A Lightweight Graph Processing
// Framework for Shared Memory", presented at Principles and Practice of 
// Parallel Programming, 2013.
// Copyright (c) 2013 Julian Shun and Guy Blelloch
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "ligra.h"
#include "math.h"
#include "pcm/cpucounters.h"
#include "immintrin.h"

template <class vertex>
struct PR_F {
  double* p_curr, *p_next;
  vertex* V;
  PR_F(double* _p_curr, double* _p_next, vertex* _V) : 
    p_curr(_p_curr), p_next(_p_next), V(_V) {}
  inline bool update(uintE s, uintE d){ //update function applies PageRank equation
    p_next[d] += p_curr[s]/V[s].getOutDegree();
    return 1;
  }
  inline bool updateAtomic (uintE s, uintE d) { //atomic Update
    writeAdd(&p_next[d],p_curr[s]/V[s].getOutDegree());
    return 1;
  }
  inline bool cond (intT d) { return cond_true(d); }};

//vertex map function to update its p value according to PageRank equation
struct PR_Vertex_F {
  double damping;
  double addedConstant;
  double* p_curr;
  double* p_next;
  PR_Vertex_F(double* _p_curr, double* _p_next, double _damping, intE n) :
    p_curr(_p_curr), p_next(_p_next), 
    damping(_damping), addedConstant((1-_damping)*(1/(double)n)){}
  inline bool operator () (uintE i) {
    p_next[i] = damping*p_next[i] + addedConstant;
    return 1;
  }
};

//resets p
struct PR_Vertex_Reset {
  double* p_curr;
  PR_Vertex_Reset(double* _p_curr) :
    p_curr(_p_curr) {}
  inline bool operator () (uintE i) {
    p_curr[i] = 0.0;
    return 1;
  }
};

template <class vertex>
void Compute(graph<vertex>& GA, commandLine P) {
  long maxIters = P.getOptionLongValue("-maxiters",100);
  const int detail = (int)P.getOptionLongValue("-detail", 0);
  const intE n = GA.n;
  const double damping = 0.85, epsilon = 0.0000001;

  double one_over_n = 1/(double)n;
  double* p_curr = newA(double,n);
  {parallel_for(long i=0;i<n;i++) p_curr[i] = one_over_n;}
  double* p_next = newA(double,n);
  {parallel_for(long i=0;i<n;i++) p_next[i] = 0;} //0 if unchanged
  bool* frontier = newA(bool,n);
  {parallel_for(long i=0;i<n;i++) frontier[i] = 1;}

  const auto pcm = PCM::getInstance();
  pcm->program(PCM::DEFAULT_EVENTS, nullptr);

  vertexSubset Frontier(n,n,frontier);
  const auto start = getSystemCounterState();
  long iter = 0;
  using std::vector;
  vector<SystemCounterState> p;
  p.reserve(maxIters);
  while(iter++ < maxIters) {
    if(detail & 0b001) {
      p.emplace_back(getSystemCounterState());
    }
    vertexSubset output = edgeMap(GA,Frontier,PR_F<vertex>(p_curr,p_next,GA.V),0);
    if(detail & 0b011) {
      p.emplace_back(getSystemCounterState());
    }
    vertexMap(Frontier,PR_Vertex_F(p_curr,p_next,damping,n));
    if(detail & 0b110) {
      p.emplace_back(getSystemCounterState());
    }
    //compute L1-norm between p_curr and p_next
    {parallel_for(long i=0;i<n;i++) {
      p_curr[i] = fabs(p_curr[i]-p_next[i]);
    }}
    //double L1_norm = sequence::plusReduce(p_curr,n);
    //if(L1_norm < epsilon) break;
    //reset p_curr
    vertexMap(Frontier,PR_Vertex_Reset(p_curr));
    if(detail & 0b100) {
      p.emplace_back(getSystemCounterState());
    }
    swap(p_curr,p_next);
    Frontier.del();
    Frontier = output;
  }
  Frontier.del(); free(p_curr); free(p_next);
  const auto end = getSystemCounterState();
  std::cout << "-----overall-----"
    << "\nMemREAD[GB] " << getBytesReadFromMC(start, end) / 1024.0 / 1024.0 / 1024.0
    << "\nMemWrite[GB] " << getBytesWrittenToMC(start, end) / 1024.0 / 1024.0 / 1024.0
    << "\n#L2Miss " << getL2CacheMisses(start, end)
    << "\n#L3Miss " << getL3CacheMisses(start, end)
    << "\nL2Hit " << getL2CacheHitRatio(start, end)
    << "\nL3Hit " << getL3CacheHitRatio(start, end)
    << "\n#Retired " << getInstructionsRetired(start, end)
    << "\nIPC " << getIPC(start, end) << std::endl;

  const int N = _popcnt32(detail);
  const int M = N+1;
  vector<double> read(M,0);
  vector<double> write(M,0);
  vector<long> l2miss(M,0);
  vector<long> l3miss(M,0);
  vector<double> l2hit(M,0);
  vector<double> l3hit(M,0);
  vector<int> retired(M,0);
  vector<double> ipc(M,0);
  for(int i=0;i<maxIters;++i){
    for(int j=0,end=N;j<end;++j){
      read[j] += (getBytesReadFromMC(p[M*i+j],p[M*i+j+1])/1024.0/1024.0);
      write[j] += (getBytesWrittenToMC(p[M*i+j],p[M*i+j+1])/1024.0/1024.0);
      l2miss[j] += (getL2CacheMisses(p[M*i+j],p[M*i+j+1])/1024.0);
      l3miss[j] += (getL3CacheMisses(p[M*i+j],p[M*i+j+1])/1024.0);
      l2hit[j] += getL2CacheHitRatio(p[M*i+j],p[M*i+j+1]);
      l3hit[j] += getL3CacheHitRatio(p[M*i+j],p[M*i+j+1]);
      retired[j] += (getInstructionsRetired(p[M*i+j],p[M*i+j+1])/1024.0/1024.0);
      ipc[j] += getIPC(p[M*i+j],p[M*i+j+1]);
    }
  }
  for(int j=0;j<N;++j){
    read[j] /= maxIters;
    write[j] /= maxIters;
    l2miss[j] /= maxIters;
    l3miss[j] /= maxIters;
    l2hit[j] /= maxIters;
    l3hit[j] /= maxIters;
    retired[j] /= maxIters;
    ipc[j] /= maxIters;
  }
  if(detail & 0b001) {
    std::cout << "-----edgeMap-----"
      << "\nMemREAD[MB] " << read[0]
      << "\nMemWrite[MB] " << write[0]
      << "\n#L2Miss(K) " << l2miss[0]
      << "\n#L3Miss(K) " << l3miss[0]
      << "\nL2Hit " << l2hit[0]
      << "\nL3Hit " << l3hit[0]
      << "\n#Retired(M) " << retired[0]
      << "\nIPC " << ipc[0] << std::endl;
  }
  if(detail & 0b010) {
    //const auto i = N - _tzcnt_u32((uint32_t)detail);
    const auto i = detail & 0b001;
    std::cout << "-----vertexMap pr-----"
      << "\nMemREAD[MB] " << read[i]
      << "\nMemWrite[MB] " << write[i]
      << "\n#L2Miss(K) " << l2miss[i]
      << "\n#L3Miss(K) " << l3miss[i]
      << "\nL2Hit " << l2hit[i]
      << "\nL3Hit " << l3hit[i]
      << "\n#Retired(M) " << retired[i]
      << "\nIPC " << ipc[i] << std::endl;
  }
  if(detail & 0b100) {
    //const auto i = N - _tzcnt_u32((uint32_t)detail);
    const auto i = (detail & 0b001) + ((detail & 0b010)>0);
    std::cout << "-----vertexMap reset-----"
      << "\nMemREAD[MB] " << read[i]
      << "\nMemWrite[MB] " << write[i]
      << "\n#L2Miss(K) " << l2miss[i]
      << "\n#L3Miss(K) " << l3miss[i]
      << "\nL2Hit " << l2hit[i]
      << "\nL3Hit " << l3hit[i]
      << "\n#Retired(M) " << retired[i]
      << "\nIPC " << ipc[i] << std::endl;
  }
}
