"""Matched P800 encoder benchmark, including Graphormer structural preparation."""
import os,sys,pathlib,argparse,time,json,statistics,gc
ROOT=pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault('QUARL_GRAPH_BACKEND','torch');os.environ.setdefault('DGLBACKEND','pytorch')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
sys.path[:0]=[str(ROOT/'experiment/ppo-new'),os.environ.get('QUARL_NATIVE_PYTHON',str(ROOT/'python'))]
os.chdir(ROOT/'experiment/ppo-new')
import torch,dgl,qtz
from runtime import select_device
from utils import seed_all
from model.actor_critic import ActorCritic

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=1);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    seed_all(98766);device=select_device([a.gpu])
    qtz.init_quartz_context(['h','cx','x','rz','add','neg'],str(ROOT/'experiment/ecc_set/nam_325_ecc.json'),False,True)
    circuit=qtz.qasm_to_graph((ROOT/'experiment/circs/nam_circs/barenco_tof_3.qasm').read_text());circuit.rotation_merging('rz');g=circuit.to_dgl_graph()
    results=[];parameters={}
    for name in ['QGNN','QGNNGlobal','QGraphormer']:
        seed_all(98766)
        model=ActorCritic(gnn_type=name,num_gate_types=40,gate_type_embed_dim=16,gnn_num_layers=6,gnn_hidden_dim=128,gnn_output_dim=128,actor_hidden_size=256,critic_hidden_size=128,action_dim=1,device=device).to(device).gnn
        parameters[name]=sum(p.numel() for p in model.parameters())
        def run(batch,backward):
            model.zero_grad(set_to_none=True)
            if backward:
                model(batch)[:,0].square().mean().backward()
            else:
                with torch.no_grad():model(batch)
        for size in [8,64,4800]:
            batched=dgl.batch([g]*size)
            for backward in [False,True]:
                model.train(backward);run(batched,backward);torch.cuda.synchronize(device)
                repeats=3 if size<4800 else 2;times=[]
                for _ in range(repeats):
                    torch.cuda.synchronize(device);start=time.perf_counter();run(batched,backward);torch.cuda.synchronize(device);times.append(time.perf_counter()-start)
                row=dict(encoder=name,batch=size,nodes=batched.num_nodes(),forward_backward=backward,median_seconds=statistics.median(times),samples_seconds=times,structural_cache='warm; same repeated initial circuit',stats=getattr(model,'stats',{}).copy());results.append(row);print('ENCODER_BENCH',json.dumps(row),flush=True)
            del batched
        del model;gc.collect();torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'device':str(device),'parameters':parameters,'scope':'Encoder forward / forward+backward only; excludes action matching, rewrites, actor/critic and optimizer. Repeated 46-gate initial circuit gives warm structural cache; actual trajectory iteration timers include novel structures.','results':results},indent=2));print('ENCODER_BENCH_DONE',flush=True)
