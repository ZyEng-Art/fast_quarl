"""From-scratch PPO plus concurrent search, with verified target stopping."""
import argparse, hashlib, json, os, pathlib, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
BASE=ROOT/'runs'/'fast_quarl'

def atomic(path,data):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def imports():
 os.chdir(ROOT/'experiment/ppo-new');sys.path.insert(0,str(ROOT/'experiment/ppo-new'));sys.path.insert(0,os.environ.get('QUARL_NATIVE_PYTHON',str(ROOT/'python')))
 os.environ.setdefault('QUARL_GRAPH_BACKEND','torch');os.environ.setdefault('DGLBACKEND','pytorch')
 os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4');os.environ.setdefault('WANDB_MODE','disabled')

def train(a):
 imports()
 import torch,ppo,types
 from omegaconf import OmegaConf
 from config.nam2_config import Nam2FTConfig
 class PilotPPO(ppo.PPOMod):
  def train_iter(self):
   begin=time.monotonic()
   if a.mode=='uniform':
    self.agent.perpare_buf_for_next_iter()
    with torch.no_grad():exp=self.agent.collect_data_by_self(self.cfg.num_eps_per_iter,self.cfg.agent_batch_size,self.cfg.max_extra_cost,self.cfg.nop_stop,self.cfg.greedy_sample)
    self.tot_exps_collected+=len(exp);self.agent.sync_best_graph();self.agent.output_best_graph(self.cfg.best_graph_output_dir)
    initial=torch.load(run/'initial.pt',map_location='cpu')['model_state_dict'];state=self.ac_net.state_dict()
    delta=max(float((state[k].detach().cpu()-initial[k]).abs().max()) for k in initial);assert delta==0.0,delta
    atomic(run/'rollout_status.json',{'iteration':self.i_iter,'transitions':self.tot_exps_collected,'model_state_max_delta':delta,'backward_calls':0,'optimizer_steps':0,'elapsed_seconds':time.time()-self.start_time_sec})
    result=0.0
   else:result=super().train_iter()
   with (run/'iteration_timing.jsonl').open('a') as f:f.write(json.dumps({'iteration':self.i_iter,'seconds':time.monotonic()-begin,'loss':result,'transitions':self.tot_exps_collected,'train_encoder_stats':getattr(self.ac_net.gnn,'stats',{}),'rollout_encoder_stats':getattr(self.ac_net_old.gnn,'stats',{})})+'\n')
   return result
  def _make_actor_critic(self):
   model=super()._make_actor_critic()
   if a.mode=='uniform':
    with torch.no_grad():
     for module in [model.actor,model.critic]:
      for p in module.parameters():p.zero_()
   if a.mode!='update':
    def frozen_train(self, mode=True):return torch.nn.Module.train(self,False)
    model.train=types.MethodType(frozen_train,model);model.eval()
   state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   fingerprint=hashlib.sha256(b''.join(v.numpy().tobytes() for v in state.values())).hexdigest()
   torch.save({'model_state_dict':state},run/'initial.pt')
   atomic(run/'initial.json',{'hash':fingerprint,'seed':a.seed,'mode':a.mode,'gnn_type':a.gnn_type,'parameters':sum(p.numel() for p in model.parameters()),'encoder_parameters':sum(p.numel() for p in model.gnn.parameters())})
   return model
 run=BASE/f'{a.seed}_{a.mode}';run.mkdir(parents=True,exist_ok=True)
 cfg=OmegaConf.structured(Nam2FTConfig());cfg.input_graphs[0].path=str(BASE/'initial.qasm');cfg.seed=a.seed;cfg.gpus=[a.gpu];cfg.resume=False;cfg.wandb.en=False
 cfg.gnn_type=a.gnn_type;cfg.subgraph_opt=a.subgraph_opt;cfg.gnn_num_layers=6;cfg.num_eps_per_iter=64;cfg.agent_batch_size=64;cfg.max_eps_len=600;cfg.min_eps_len=20
 cfg.dyn_eps_len=False;cfg.mini_batch_size=4800;cfg.k_epochs=5;cfg.lr_scheduler='none';cfg.time_budget=''
 cfg.best_graph_output_dir=str(run/"best_graphs");cfg.ddp_port=a.ddp_port+a.gpu;cfg.omp_num_threads=4;cfg.max_iterations=100000;cfg.obs_per_agent=0
 if a.mode!='update':cfg.lr_gnn=cfg.lr_actor=cfg.lr_critic=0.0
 (run/'.hydra').mkdir(exist_ok=True);OmegaConf.save(OmegaConf.create({'c':cfg}),run/'.hydra/config.yaml')
 started=time.monotonic();runner=PilotPPO(cfg,str(run));runner.init_process(0,1,0)
 last=json.loads((run/'ckpts/latest.json').read_text());laststate=torch.load(last['path'],map_location='cpu')['model_state_dict']
 initial=torch.load(run/'initial.pt',map_location='cpu')['model_state_dict']
 delta=sum(float((laststate[k]-initial[k]).square().sum()) for k in dict(runner.ac_net.named_parameters()))**.5
 state_delta=max(float((laststate[k]-initial[k]).abs().max()) for k in initial)
 if a.mode!='update':assert state_delta==0.0,state_delta
 atomic(run/'training_result.json',{'parameter_delta_l2':delta,'all_state_max_delta':state_delta,'wall_seconds':time.monotonic()-started,'last_checkpoint':last['path']})

def search(a):
 imports()
 import gc,math,re,types
 import torch,qtz
 from omegaconf import OmegaConf
 from model.actor_critic import ActorCritic
 from runtime import select_device
 from tester import Tester
 from utils import CostType,seed_all
 from qiskit import QuantumCircuit
 from qiskit.quantum_info import Operator
 run=BASE/f'{a.seed}_{a.mode}';deadline=time.monotonic()+90
 while not (run/'initial.json').exists():
  if time.monotonic()>deadline:raise RuntimeError('trainer initialization timeout')
  time.sleep(.2)
 cfg=OmegaConf.load(run/'.hydra/config.yaml').c;seed_all(a.seed);device=select_device([a.gpu])
 qtz.init_quartz_context(cfg.gate_set,str(ROOT/'experiment/ecc_set/nam_325_ecc.json'),False,True)
 keys=('gnn_type','num_gate_types','gate_type_embed_dim','gnn_num_layers','gnn_hidden_dim','gnn_output_dim','gin_num_mlp_layers','gin_learn_eps','gin_neighbor_pooling_type','gin_graph_pooling_type','actor_hidden_size','critic_hidden_size')
 model=ActorCritic(**{k:cfg[k] for k in keys},action_dim=qtz.quartz_context.num_xfers,device=device).to(device)
 model.load_state_dict(torch.load(run/'initial.pt',map_location='cpu')['model_state_dict'])
 if a.mode=='uniform':
  # Independent continuous tie-breakers make greedy selection uniform over
  # valid transformations while retaining the same masks/restart procedure.
  def uniform_forward(self,x):return torch.rand((x.shape[0],qtz.quartz_context.num_xfers),device=x.device)
  model.actor.forward=types.MethodType(uniform_forward,model.actor)
 original=(ROOT/'experiment/circs/nam_circs/barenco_tof_3.qasm').read_text();best=qtz.qasm_to_graph((BASE/'initial.qasm').read_text())
 tester=Tester(cost_type=CostType.from_str('gate_count'),ac_net=model,device=device,output_dir=str(run),sync_tuning_dir=False,hit_rate=cfg.hit_rate,batch_size=8,max_loss_tolerance=math.inf,max_search_sec=1200,vmem_perct_limit=70)
 # Restore upstream's disabled training-to-search exchange during search.
 import inspect,textwrap
 scope={};exec(textwrap.dedent(inspect.getsource(Tester.random_search)).replace('if False:', 'if True:'),vars(sys.modules['tester']),scope)
 tester.random_search=types.MethodType(scope['random_search'],tester);tester.sync_tuning_dir=True
 original_expand=tester.random_expand;last_reload=[0.0];loaded=[None]
 def expand(*args,**kwargs):
  if (BASE/'stop').exists():raise SystemExit(0)
  if time.monotonic()-last_reload[0]>30:
   try:
    info=json.loads((run/'ckpts/latest.json').read_text())
    if info['path']!=loaded[0]:
     model.load_state_dict(torch.load(info['path'],map_location='cpu')['model_state_dict']);loaded[0]=info['path']
     print('LIVE_CHECKPOINT',info['path'],flush=True)
   except (FileNotFoundError,json.JSONDecodeError):pass
   last_reload[0]=time.monotonic()
  return original_expand(*args,**kwargs)
 tester.random_expand=expand
 trace=[];adopted=[];lastpath=None;start=time.monotonic()
 with torch.no_grad():
  while not (BASE/'stop').exists():
   try:
    info=json.loads((run/'ckpts/latest.json').read_text())
    if info['path']!=lastpath:
     model.load_state_dict(torch.load(info['path'],map_location='cpu')['model_state_dict']);lastpath=info['path'];adopted.append({'t':time.monotonic()-start,'path':lastpath})
   except (FileNotFoundError,json.JSONDecodeError):pass
   # Apply the paper's two-way best-circuit exchange in every arm. The
   # upstream tester disables this direction with an `if False` guard.
   try:
    for b in json.loads((run/'sync_dir/best_info_0.json').read_text()):
     if b['best_cost']<best.gate_count:
      best=qtz.qasm_to_graph(b['qasm']);trace.append({'t':time.monotonic()-start,'gates':best.gate_count,'source':'training'})
   except (FileNotFoundError,json.JSONDecodeError):pass
   tester.max_search_sec=1200
   candidate=tester.random_search('barenco_tof_3',best)
   if candidate is not None and candidate.gate_count<best.gate_count:
    best=candidate;trace.append({'t':time.monotonic()-start,'gates':best.gate_count,'source':'search'})
   atomic(run/'sync_dir/best_info_search.json',[{'name':'barenco_tof_3','best_cost':best.gate_count,'qasm':best.to_qasm_str()}])
 (run/'best.qasm').write_text(best.to_qasm_str())
 eq=bool(Operator(QuantumCircuit.from_qasm_str(original)).equiv(Operator(QuantumCircuit.from_qasm_str(best.to_qasm_str()))));assert eq
 atomic(run/'search_result.json',{'best_gates':best.gate_count,'seconds':time.monotonic()-start,'equivalent':eq,'trace':trace,'adopted_checkpoints':adopted})
 print('PILOT_SEARCH_DONE',a.seed,a.mode,best.gate_count,flush=True)
 del tester,model;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()

def controller():
 BASE.mkdir(parents=True,exist_ok=True)
 if (BASE/'status.json').exists():raise RuntimeError('Use a fresh --run-dir; an existing experiment will not be overwritten')
 os.environ['LD_LIBRARY_PATH']=str(ROOT/'.xpu-native/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
 imports()
 import qtz
 from qiskit import QuantumCircuit
 from qiskit.quantum_info import Operator
 qtz.init_quartz_context(['h','cx','x','rz','add','neg'],str(ROOT/'experiment/ecc_set/nam_325_ecc.json'),False,True)
 original=(ROOT/'experiment/circs/nam_circs/barenco_tof_3.qasm').read_text();reference=Operator(QuantumCircuit.from_qasm_str(original))
 initial=qtz.qasm_to_graph(original)
 if ARGS.preprocess:initial.rotation_merging('rz')
 (BASE/'initial.qasm').write_text(initial.to_qasm_str())
 start=time.time();run=BASE/f'{ARGS.seed}_{ARGS.mode}';run.mkdir(exist_ok=True)
 protocol={'start_unix':start,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(start)),'seed':ARGS.seed,'target_rm':35,'target_raw':36,'pretrained':False,'initial_raw_gates':initial.gate_count,'preprocess_rotation_merging':ARGS.preprocess,'warmup_seconds':300,'search_timeout_seconds':1200,'training':{'episodes':64,'horizon':600,'epochs':5,'minibatch':4800,'learning_rates':[.0003,.0003,.0005]},'rotation_merging':'postprocess every new best circuit; raw best shared between trainer and search','stop_rule':'stop both workers on verified <=35 RM gates; no automatic wall-time cutoff','implementation':'fast_quarl','upstream_xpu_commit':'c719a205b75af90f01ab67affb764cdfff15a738'}
 protocol['code_commit']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();protocol['native_python']=os.environ.get('QUARL_NATIVE_PYTHON',str(ROOT/'python'));protocol['gnn_type']=ARGS.gnn_type;protocol['subgraph_opt']=ARGS.subgraph_opt;protocol['devices']={'train':ARGS.train_gpu,'search':ARGS.search_gpu};protocol['mode']=ARGS.mode;protocol['backward']=ARGS.mode=='update';protocol['optimizer_updates']=ARGS.mode=='update'
 protocol['acceleration']='prepare shared graph topology once per GNN forward; identical network and PPO hyperparameters'
 atomic(BASE/'protocol.json',protocol);jobs={};logs={};best_raw=59;best_rm=59;milestones=[];verified={}
 def launch(kind,gpu):
  logs[kind]=open(run/(kind+'.log'),'w')
  jobs[kind]=subprocess.Popen([sys.executable,__file__,kind,'--seed',str(ARGS.seed),'--mode',ARGS.mode,'--gpu',str(gpu),'--run-dir',str(BASE),'--gnn-type',ARGS.gnn_type,'--ddp-port',str(ARGS.ddp_port),('--subgraph-opt' if ARGS.subgraph_opt else '--no-subgraph-opt')],stdout=logs[kind],stderr=subprocess.STDOUT)
 launch('train',ARGS.train_gpu)
 try:
  while True:
   elapsed=time.time()-start
   if elapsed>=300 and 'search' not in jobs:launch('search',ARGS.search_gpu)
   candidates=[('initial',(BASE/'initial.qasm').read_text())]
   for kind,file in [('training','best_info_0.json'),('search','best_info_search.json')]:
    try:
     candidates += [(kind,b['qasm']) for b in json.loads((run/'sync_dir'/file).read_text())]
    except (FileNotFoundError,json.JSONDecodeError):pass
   for source,qasm in candidates:
    key=hashlib.sha256(qasm.encode()).hexdigest()
    if key in verified:continue
    graph=qtz.qasm_to_graph(qasm);raw=graph.gate_count;graph.rotation_merging('rz');rm=graph.gate_count;merged=graph.to_qasm_str()
    eq=bool(reference.equiv(Operator(QuantumCircuit.from_qasm_str(merged))))
    if not eq:raise RuntimeError('Circuit equivalence failed: '+source)
    verified[key]=True
    if raw<best_raw or rm<best_rm:
     best_raw=min(best_raw,raw);best_rm=min(best_rm,rm)
     record={'seconds':time.time()-start,'source':source,'raw_gates':raw,'rm_gates':rm,'equivalent':True};milestones.append(record)
     atomic(BASE/'milestones.json',milestones)
     (BASE/f'best_raw_{raw}.qasm').write_text(qasm);(BASE/f'best_rm_{rm}.qasm').write_text(merged)
     print('IMPROVEMENT',json.dumps(record),flush=True)
   atomic(BASE/'status.json',{'elapsed_seconds':time.time()-start,'best_raw':best_raw,'best_rm':best_rm,'target_reached':best_rm<=35,'worker_pids':{k:p.pid for k,p in jobs.items()},'milestones':milestones})
   if best_rm<=35:
    atomic(BASE/'result.json',{'elapsed_seconds':time.time()-start,'first_target_seconds':next(x['seconds'] for x in milestones if x['rm_gates']<=35),'best_raw':best_raw,'best_rm':best_rm,'equivalent':True,'milestones':milestones});break
   for kind,p in jobs.items():
    if p.poll() is not None:raise RuntimeError(f'{kind} exited unexpectedly: {p.returncode}; see {run}/{kind}.log')
   # Keep the latest two complete checkpoints; avoid unbounded storage use.
   checkpoints=sorted((run/'ckpts').glob('iter_*.pt'),key=lambda p:p.stat().st_mtime)
   for old in checkpoints[:-2]:
    try:old.unlink()
    except FileNotFoundError:pass
   time.sleep(5)
 except BaseException as exc:
  atomic(BASE/'error.json',{'error':repr(exc),'elapsed_seconds':time.time()-start});raise
 finally:
  (BASE/'stop').touch()
  for p in jobs.values():
   if p.poll() is None:p.terminate()
  for p in jobs.values():
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for log in logs.values():log.close()

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['controller','train','search']);p.add_argument('--seed',type=int,default=98766);p.add_argument('--mode',choices=['update','uniform'],default='update');p.add_argument('--gpu',type=int,default=0);p.add_argument('--train-gpu',type=int,default=0);p.add_argument('--search-gpu',type=int,default=1);p.add_argument('--run-dir',type=pathlib.Path,default=BASE);p.add_argument('--preprocess',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--gnn-type',choices=['QGNN','QGNNGlobal','QGraphormer'],default='QGNN');p.add_argument('--subgraph-opt',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--ddp-port',type=int,default=24600);ARGS=p.parse_args();BASE=ARGS.run_dir.resolve()
 if ARGS.gnn_type in ('QGNNGlobal','QGraphormer') and ARGS.subgraph_opt:p.error('Global encoders require --no-subgraph-opt to use complete circuit context')
 {'controller':controller,'train':lambda:train(ARGS),'search':lambda:search(ARGS)}[ARGS.kind]()
