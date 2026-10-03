"""Circuit Graphormer: global attention with centrality, distance and path bias.

Path edge encodings average all shortest paths to avoid node-ID tie breaking.
Only unweighted structural arrays are cached; learned embeddings are recomputed.
"""
import collections
import ctypes
import math
import pathlib
import time
import numpy as np
import torch
from torch import nn

LIB = pathlib.Path(__file__).resolve().parents[3]/'.graphormer-native/libgraphormer.so'
_NATIVE = None
if LIB.exists():
    _NATIVE = ctypes.CDLL(str(LIB)).graphormer_paths
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    _NATIVE.argtypes = [ctypes.c_int, ctypes.c_int, i64, i64, i64, i64, i64, i32, f32]
    _NATIVE.restype = ctypes.c_int

def path_structure(n, src, dst, sp, dp, rev, use_native=True):
    arrays = [np.ascontiguousarray(x, dtype=np.int64) for x in (src,dst,sp,dp,rev)]
    distance = np.full((n,n), -1, dtype=np.int32)
    encoded = np.zeros((n,n,18), dtype=np.float32)
    if use_native and _NATIVE is not None:
        ret = _NATIVE(n, len(src), *arrays, distance, encoded)
        if ret != 0: raise ValueError('Invalid circuit edges: native preprocessing code %d' % ret)
        return distance, encoded
    adj = [[] for _ in range(n)]
    for u,v,a,b,r in zip(*arrays):
        adj[u].append((v, (min(max(a,0),7), 8+min(max(b,0),7), 16+int(r!=0))))
    for root in range(n):
        d=distance[root]; h=encoded[root]; count=np.zeros(n); count[root]=1; d[root]=0
        q=collections.deque([root])
        while q:
            u=q.popleft()
            for v,features in adj[u]:
                if d[v]<0: d[v]=d[u]+1; q.append(v)
                if d[v]!=d[u]+1: continue
                total=count[v]+count[u]; a=count[v]/total; b=count[u]/total
                h[v]=a*h[v]+b*h[u]
                for f in features: h[v,f]+=b
                count[v]=total
        valid=d>0; h[valid]/=d[valid,None]
    return distance, encoded

class GraphormerLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__(); self.heads=heads; self.head_dim=width//heads
        self.norm1=nn.LayerNorm(width); self.norm2=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width); self.out=nn.Linear(width,width)
        self.ffn=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
    def forward(self, x, bias):
        batch,length,width=x.shape
        qkv=self.qkv(self.norm1(x)).reshape(batch,length,3,self.heads,self.head_dim)
        q,k,v=qkv.permute(2,0,3,1,4).unbind(0)
        scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)+bias
        attention=torch.softmax(scores,dim=-1)
        value=torch.matmul(attention,v).transpose(1,2).reshape(batch,length,width)
        x=x+self.out(value)
        return x+self.ffn(self.norm2(x))

class QGraphormer(nn.Module):
    def __init__(self,num_layers,num_gate_types,gate_type_embed_dim,h_feats,inter_dim,heads=4):
        super().__init__()
        if h_feats % heads: raise ValueError('Graphormer width must be divisible by attention heads')
        self.num_gate_types=num_gate_types; self.heads=heads; self.max_distance=32
        self.gate_embedding=nn.Embedding(num_gate_types+1,h_feats)
        self.in_degree_embedding=nn.Embedding(33,h_feats); self.out_degree_embedding=nn.Embedding(33,h_feats)
        self.spatial_embedding=nn.Embedding(34,heads); self.path_edge_encoder=nn.Linear(18,heads,bias=False)
        self.graph_token=nn.Parameter(torch.empty(1,1,h_feats)); self.virtual_distance=nn.Parameter(torch.zeros(heads))
        self.layers=nn.ModuleList([GraphormerLayer(h_feats,heads) for _ in range(num_layers)])
        self.final_norm=nn.LayerNorm(h_feats)
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.normal_(module.weight,std=0.02/math.sqrt(num_layers))
                if module.bias is not None: nn.init.zeros_(module.bias)
            elif isinstance(module,nn.Embedding): nn.init.normal_(module.weight,std=0.02)
        nn.init.normal_(self.graph_token,std=0.02)
        self._structure_cache=collections.OrderedDict(); self._cache_bytes=0
        self.cache_limit_bytes=64*1024*1024; self.cache_limit_entries=1024
        self.stats={'native_preprocessing':_NATIVE is not None,'forward_calls':0,'cache_hits':0,'cache_misses':0,'prepare_seconds':0.0}

    def _structure(self,n,src,dst,sp,dp,rev):
        key=(n,src.tobytes(),dst.tobytes(),sp.tobytes(),dp.tobytes(),rev.tobytes())
        if key in self._structure_cache:
            self.stats['cache_hits']+=1; self._structure_cache.move_to_end(key); return self._structure_cache[key][0]
        self.stats['cache_misses']+=1
        distance,paths=path_structure(n,src,dst,sp,dp,rev)
        original=rev==0
        indeg=np.bincount(dst[original],minlength=n).clip(0,32)
        outdeg=np.bincount(src[original],minlength=n).clip(0,32)
        spatial=np.where(distance<0,33,np.minimum(distance,32)).astype(np.int64)
        value=(indeg,outdeg,spatial,paths); size=sum(v.nbytes for v in value)+sum(len(k) for k in key[1:])
        if size<=self.cache_limit_bytes:
            while self._structure_cache and (self._cache_bytes+size>self.cache_limit_bytes or len(self._structure_cache)>=self.cache_limit_entries):
                _,(_,oldsize)=self._structure_cache.popitem(last=False); self._cache_bytes-=oldsize
            self._structure_cache[key]=(value,size); self._cache_bytes+=size
        return value

    def _prepare(self,g):
        started=time.perf_counter(); counts=g.batch_num_nodes().numpy().tolist(); edge_counts=g.batch_num_edges().numpy().tolist()
        if not counts or min(counts)<1: raise ValueError('Graphormer requires nonempty circuits')
        batch=len(counts); length=max(counts)+1
        gate=np.full((batch,length),self.num_gate_types,dtype=np.int64)
        indeg=np.zeros_like(gate); outdeg=np.zeros_like(gate); spatial=np.full((batch,length,length),33,dtype=np.int64)
        paths=np.zeros((batch,length,length,18),dtype=np.float32); valid=np.zeros((batch,length),dtype=np.bool_)
        src,dst=(x.numpy() for x in g.edges(order='eid'))
        sp=g.edata['src_idx'].numpy();dp=g.edata['dst_idx'].numpy();rev=g.edata['reversed'].numpy()
        gates=g.ndata['gate_type'].numpy();flat=[];node_start=edge_start=0
        for i,(n,m) in enumerate(zip(counts,edge_counts)):
            sl=slice(edge_start,edge_start+m)
            si,so,sd,se=self._structure(n,src[sl]-node_start,dst[sl]-node_start,sp[sl],dp[sl],rev[sl])
            gate[i,1:n+1]=gates[node_start:node_start+n];indeg[i,1:n+1]=si;outdeg[i,1:n+1]=so
            spatial[i,1:n+1,1:n+1]=sd;paths[i,1:n+1,1:n+1]=se;valid[i,:n+1]=True
            flat.extend(range(i*length+1,i*length+n+1));node_start+=n;edge_start+=m
        self.stats['forward_calls']+=1;self.stats['prepare_seconds']+=time.perf_counter()-started
        return gate,indeg,outdeg,spatial,paths,valid,np.array(flat,dtype=np.int64)

    def forward(self,g):
        gate,indeg,outdeg,spatial,paths,valid,flat=self._prepare(g)
        device=self.graph_token.device
        gate,indeg,outdeg,spatial=[torch.from_numpy(a).to(device) for a in (gate,indeg,outdeg,spatial)]
        path_tensor=torch.from_numpy(paths).to(device=device,dtype=self.graph_token.dtype)
        valid=torch.from_numpy(valid).to(device);flat=torch.from_numpy(flat).to(device)
        x=self.gate_embedding(gate)+self.in_degree_embedding(indeg)+self.out_degree_embedding(outdeg)
        token=self.graph_token.expand(x.shape[0],-1,-1)
        x=torch.cat([token,x[:,1:]],dim=1)
        bias=(self.spatial_embedding(spatial)+self.path_edge_encoder(path_tensor)).permute(0,3,1,2)
        # Virtual token is at distance independent of circuit topology.
        token_mask=torch.zeros((x.shape[1],x.shape[1]),device=device,dtype=torch.bool)
        token_mask[0,:]=True;token_mask[:,0]=True
        bias=torch.where(token_mask[None,None,:,:],self.virtual_distance[None,:,None,None],bias)
        bias=bias.masked_fill(~valid[:,None,None,:],-1e9)
        for layer in self.layers: x=layer(x,bias)
        x=self.final_norm(x)
        return x.reshape(-1,x.shape[-1])[flat]
