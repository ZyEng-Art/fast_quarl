// Unweighted shortest paths and mean edge encodings over all shortest paths.
#include <vector>
#include <deque>
#include <algorithm>
#include <cstdint>
#include <cmath>
struct Edge { int to; int feature[3]; };
extern "C" int graphormer_paths(int n, int m, const int64_t* src, const int64_t* dst, const int64_t* sp, const int64_t* dp, const int64_t* rev, int32_t* distance, float* encoded) {
  if (n < 1 || m < 0) return -1;
  std::vector<std::vector<Edge>> adj(n);
  for (int e=0;e<m;++e) {
    if (src[e]<0 || src[e]>=n || dst[e]<0 || dst[e]>=n) return -2;
    Edge x; x.to=dst[e]; x.feature[0]=std::clamp<int64_t>(sp[e],0,7); x.feature[1]=8+std::clamp<int64_t>(dp[e],0,7); x.feature[2]=16+(rev[e]!=0); adj[src[e]].push_back(x);
  }
  std::fill(distance,distance+n*n,-1);
  std::fill(encoded,encoded+n*n*18,0.f);
  for (int root=0;root<n;++root) {
    int32_t* d=distance+root*n; float* h=encoded+root*n*18;
    std::vector<double> count(n,0.); std::deque<int> q;
    count[root]=1.; d[root]=0; q.push_back(root);
    while (!q.empty()) {
      int u=q.front();q.pop_front();
      for (const auto& e:adj[u]) {
        int v=e.to;
        if (d[v]<0) {d[v]=d[u]+1;q.push_back(v);}
        if (d[v]!=d[u]+1) continue;
        double total=count[v]+count[u]; float a=count[v]/total, b=count[u]/total;
        for (int k=0;k<18;++k) h[v*18+k]=a*h[v*18+k]+b*h[u*18+k];
        for (int k=0;k<3;++k) h[v*18+e.feature[k]]+=b;
        count[v]=total;
      }
    }
    for (int v=0;v<n;++v) if (d[v]>0) for (int k=0;k<18;++k) h[v*18+k]/=d[v];
  }
  return 0;
}
