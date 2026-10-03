"""Validate circuit Graphormer structural encodings and CPU/P800 autograd."""
import os, sys, pathlib, argparse, json
ROOT=pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault('QUARL_GRAPH_BACKEND','torch');os.environ.setdefault('DGLBACKEND','pytorch')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
sys.path[:0]=[str(ROOT/'experiment/ppo-new'),str(ROOT/'tools/fast_quarl')]
import torch,dgl,numpy as np
from runtime import configure_runtime,select_device
from model.qgraphormer import QGraphormer,path_structure,_NATIVE
from model.actor_critic import ActorCritic
from check_global_context import graph

def structural_check():
    src=np.array([0,0,1,2,1,2,3,3],dtype=np.int64)
    dst=np.array([1,2,3,3,0,0,1,2],dtype=np.int64)
    sp=np.array([0,1,0,1,0,0,0,0],dtype=np.int64);dp=np.array([0,0,0,0,0,1,0,1],dtype=np.int64)
    rev=np.array([0,0,0,0,1,1,1,1],dtype=np.int64)
    a=path_structure(5,src,dst,sp,dp,rev,use_native=False)
    b=path_structure(5,src,dst,sp,dp,rev,use_native=True)
    assert _NATIVE is not None,'Build native preprocessor first'
    assert np.array_equal(a[0],b[0]);error=float(np.max(np.abs(a[1]-b[1])));assert error<1e-6
    assert b[0][0,3]==2 and b[0][0,4]==-1
    assert abs(float(b[1][0,3,0])-0.5)<1e-6
    assert abs(float(b[1][0,3,1])-0.5)<1e-6
    assert abs(float(b[1][0,3,8])-1.0)<1e-6
    return dict(native_reference_max_error=error,disconnected_distance=-1,all_shortest_path_average=True)

def check(device):
    m=QGraphormer(2,8,8,32,32).to(device).eval();a=graph(16);b=graph(7)
    with torch.no_grad():
        ref=m(a);batched=m(dgl.batch([a,b]))[:16]
        batch_error=float((ref-batched).abs().max());assert torch.allclose(ref,batched,atol=2e-4,rtol=2e-4)
        changed=graph(16);changed.ndata['gate_type'][-1]=7
        distant=float((ref[0]-m(changed)[0]).abs().max());assert distant>1e-5,distant
        reverse=torch.arange(15,-1,-1);permuted=dgl.node_subgraph(a,reverse)
        pe=float((ref[reverse.to(device)]-m(permuted)).abs().max());assert pe<2e-4,pe
        one=graph(1);assert bool(torch.isfinite(m(dgl.batch([one,a]))).all())
        altered=graph(16);altered.edata['src_idx'][0]=1;misses=m.stats['cache_misses'];m(altered);assert m.stats['cache_misses']>misses
    m.train();m.zero_grad();before=m(a).detach();out=m(a);(out[:,0].square().mean()+out[:,2].mean()).backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
    assert float(m.path_edge_encoder.weight.grad.abs().sum())>0
    assert float(m.spatial_embedding.weight.grad.abs().sum())>0
    torch.optim.Adam(m.parameters(),lr=3e-4).step();after=m(a).detach();delta=float((before-after).abs().max());assert delta>1e-5
    clone=QGraphormer(2,8,8,32,32).to(device).eval();clone.load_state_dict(m.state_dict());reload_error=float((clone(a)-after).detach().abs().max());assert reload_error<2e-4
    ac=ActorCritic(gnn_type='QGraphormer',num_gate_types=8,gate_type_embed_dim=8,gnn_num_layers=2,gnn_hidden_dim=32,gnn_output_dim=32,action_dim=20,device=device).to(device).eval()
    h=ac.gnn(a);assert ac.actor(h).shape==(16,20);assert ac.critic(h).shape==(16,1)
    return dict(device=str(device),batch_isolation_error=batch_error,permutation_error=pe,distant_change=distant,update_output_change=delta,reload_error=reload_error,finite_gradients=True,stats=m.stats)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=6);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    torch.manual_seed(98766);torch.cuda.manual_seed_all(98766);configure_runtime()
    results=dict(structure=structural_check(),devices=[check(torch.device('cpu')),check(select_device([a.gpu]))])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(results,indent=2));print('GRAPHORMER_CHECK_PASS',json.dumps(results),flush=True)
