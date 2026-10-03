"""Check global reach, batch isolation, equivariance and gradients on CPU/XPU."""
import os, sys, pathlib, argparse, json
ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault('QUARL_GRAPH_BACKEND', 'torch')
os.environ.setdefault('DGLBACKEND', 'pytorch')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
sys.path.insert(0, str(ROOT/'experiment/ppo-new'))
import torch, dgl
from runtime import configure_runtime, select_device
from model.qgnn import QGNN
from model.qgnn_global import QGNNGlobal
from model.actor_critic import ActorCritic

def graph(n):
    a=torch.arange(n-1); b=a+1
    g=dgl.graph((torch.cat([a,b]),torch.cat([b,a])),num_nodes=n)
    g.ndata['gate_type']=torch.arange(n)%4
    g.edata['src_idx']=torch.zeros(2*(n-1),dtype=torch.int32)
    g.edata['dst_idx']=torch.zeros(2*(n-1),dtype=torch.int32)
    g.edata['reversed']=torch.cat([torch.zeros(n-1),torch.ones(n-1)]).int()
    return g

def check(device):
    kw=dict(num_layers=6,num_gate_types=8,gate_type_embed_dim=8,h_feats=32,inter_dim=32)
    local=QGNN(**kw).to(device).eval(); model=QGNNGlobal(**kw).to(device).eval()
    model.load_state_dict(local.state_dict(),strict=False)
    a=graph(16);b=graph(9)
    with torch.no_grad():
        separate=model(a);joined=model(dgl.batch([a,b]))[:16]
        batch_error=float((separate-joined).abs().max())
        assert torch.allclose(separate,joined,atol=2e-4,rtol=2e-4),batch_error
        changed=graph(16);changed.ndata['gate_type'][-1]=7
        local_diff=float((local(a)[0]-local(changed)[0]).abs().max())
        global_diff=float((model(a)[0]-model(changed)[0]).abs().max())
        assert local_diff<1e-5,local_diff
        assert global_diff>1e-5,global_diff
        reverse=torch.arange(15,-1,-1);permuted=dgl.node_subgraph(a,reverse)
        permutation_error=float((model(a)[reverse.to(device)]-model(permuted)).abs().max())
        assert permutation_error<2e-4,permutation_error
    model.train();model.zero_grad();model(a)[0].square().mean().backward()
    assert model.context_fusion[0].weight.grad is not None
    assert float(model.context_fusion[0].weight.grad.abs().sum())>0
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    before=model.context_fusion[0].weight.detach().clone();torch.optim.Adam(model.parameters(),lr=3e-4).step()
    delta=float((before-model.context_fusion[0].weight.detach()).abs().max());assert delta>0
    ac=ActorCritic(gnn_type='QGNNGlobal',num_gate_types=8,gate_type_embed_dim=8,gnn_hidden_dim=32,gnn_output_dim=32,action_dim=20,device=device).to(device).eval()
    embeds=ac.gnn(a);assert ac.actor(embeds).shape==(16,20);assert ac.critic(embeds).shape==(16,1)
    return dict(device=str(device),batch_isolation_max_error=batch_error,local_distant_change=local_diff,global_distant_change=global_diff,permutation_max_error=permutation_error,adam_parameter_delta=delta,finite_gradients=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=4);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    torch.manual_seed(98766);torch.cuda.manual_seed_all(98766);configure_runtime()
    results=[check(torch.device('cpu')),check(select_device([a.gpu]))]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(results,indent=2));print('GLOBAL_CONTEXT_CHECK_PASS',json.dumps(results),flush=True)
