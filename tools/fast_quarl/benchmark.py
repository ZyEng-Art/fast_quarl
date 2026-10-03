# coding: utf-8
import os,sys,pathlib,time,json,importlib.util,statistics
import argparse,types
parser=argparse.ArgumentParser(description='Compare baseline and shared-topology GNN forward/backward')
parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--output',type=pathlib.Path);args=parser.parse_args()
ROOT=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parent
os.environ['QUARL_GRAPH_BACKEND']='torch';os.environ['DGLBACKEND']='pytorch';sys.path[:0]=[str(ROOT/'experiment/ppo-new'),str(ROOT/'python')]
try:
 import torch,dgl,qtz
 from runtime import select_device
 from omegaconf import OmegaConf
 from model.actor_critic import ActorCritic
 os.chdir(ROOT/'experiment/ppo-new');print('STAGE imports',flush=True)
 from utils import seed_all
 seed_all(98766)
 device=select_device([args.gpu])
 cfg=types.SimpleNamespace(gnn_num_layers=6,num_gate_types=40,gate_type_embed_dim=16,gnn_hidden_dim=128,gate_set=['h','cx','x','rz','add','neg'])
 qtz.init_quartz_context(cfg.gate_set,str(ROOT/'experiment/ecc_set/nam_325_ecc.json'),False,True)
 print('STAGE context',flush=True)
 circ=qtz.qasm_to_graph((ROOT/'experiment/circs/nam_circs/barenco_tof_3.qasm').read_text());circ.rotation_merging('rz');graph=circ.to_dgl_graph()
 print('STAGE graph',flush=True)
 def load(name):
  spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 base=load('qgnn_baseline');from model import qgnn as fast
 kwargs=dict(num_layers=cfg.gnn_num_layers,num_gate_types=cfg.num_gate_types,gate_type_embed_dim=cfg.gate_type_embed_dim,h_feats=cfg.gnn_hidden_dim,inter_dim=cfg.gnn_hidden_dim)
 print('STAGE kwargs',kwargs,flush=True)
 a=base.QGNN(**kwargs);print('STAGE CPU model',flush=True);a=a.to(device);print('STAGE device model',flush=True);b=fast.QGNN(**kwargs).to(device)
 print('STAGE models',flush=True)
 b.load_state_dict(a.state_dict())
 print('STAGE weights',flush=True)
 results=[]
 def sync():torch.cuda.synchronize(device)
 def step(m,g,backward):
  m.zero_grad(set_to_none=True)
  if backward:m(g).square().mean().backward()
  else:
   with torch.no_grad():m(g)
 for batch in [8,64,256]:
  g=dgl.batch([graph]*batch)
  a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True);x=a(g);y=b(g);output_error=(x-y).abs().max().item();x.square().mean().backward();y.square().mean().backward();gradient_error=max((p.grad-q.grad).abs().max().item() for p,q in zip(a.parameters(),b.parameters()));assert torch.allclose(x,y,atol=1e-6,rtol=1e-5)
  for p,q in zip(a.parameters(),b.parameters()):assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5)
  for backward in [False,True]:
   for m in [a,b]:
    for _ in range(3):step(m,g,backward)
   timings={'baseline':[],'optimized':[]}
   for repeat in range(3):
    pairs=[('baseline',a),('optimized',b)]
    if repeat%2:pairs.reverse()
    for label,m in pairs:
     sync();t=time.perf_counter()
     for _ in range(10):step(m,g,backward)
     sync();timings[label].append((time.perf_counter()-t)/10)
   med={k:statistics.median(v) for k,v in timings.items()};r={'batch':batch,'nodes':g.num_nodes(),'forward_backward':backward,'output_max_abs_error':output_error,'gradient_max_abs_error':gradient_error,'seconds':med,'speedup':med['baseline']/med['optimized'],'repetitions':timings};results.append(r);print('BENCHMARK',json.dumps(r),flush=True)
 output=args.output or ROOT/'runs/benchmark_results.json';output.parent.mkdir(parents=True,exist_ok=True)
 output.write_text(json.dumps({'device':str(device),'threads':4,'scope':'GNN forward or forward+backward only; no candidate matching, PPO actor/critic, optimizer or search-quality measurement','results':results},indent=2))
except BaseException as e:
 import traceback
 print('BENCH_ERROR',repr(e),flush=True)
 traceback.print_exc(file=sys.stdout)
 raise
