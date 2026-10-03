import ast, collections, csv, difflib, hashlib, json, math, pathlib, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
BASE=pathlib.Path('/Users/klx-pc/Downloads');OUT=pathlib.Path(__file__).resolve().parent
font=pathlib.Path('/System/Library/Fonts/STHeiti Medium.ttc')
if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'figure.facecolor':'#fafbfe','axes.facecolor':'#fafbfe','svg.fonttype':'none'})
def angle(text):
 def ev(n):
  if isinstance(n,ast.Expression):return ev(n.body)
  if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)):return n.value
  if isinstance(n,ast.Name) and n.id=='pi':return math.pi
  if isinstance(n,ast.UnaryOp):return (-1 if isinstance(n.op,ast.USub) else 1)*ev(n.operand)
  if isinstance(n,ast.BinOp):
   a,b=ev(n.left),ev(n.right)
   if isinstance(n.op,ast.Mult):return a*b
   if isinstance(n.op,ast.Div):return a/b
   if isinstance(n.op,ast.Add):return a+b
   if isinstance(n.op,ast.Sub):return a-b
  raise ValueError(text)
 return ev(ast.parse(text,mode='eval'))
def parse(file):
 text=file.read_text();n=int(re.search(r'qreg\s+q\[(\d+)\]',text).group(1));ops=[];counts=collections.Counter();depth=[0]*n;wire_hash=['input:'+str(i) for i in range(n)]
 for statement in text.split(';'):
  line=statement.strip()
  if not line or line.startswith(('OPENQASM','include','qreg','creg','//')):continue
  m=re.fullmatch(r'(\w+)(?:\((.*?)\))?\s+(.+)',line)
  if not m:raise ValueError((file,line))
  gate,param,targets=m.groups();qubits=list(map(int,re.findall(r'q\[(\d+)\]',targets)));counts[gate]+=1
  column=max(depth[q] for q in qubits)+1
  for q in qubits:depth[q]=column
  normalparam=str(round((angle(param)/math.pi)%2,10)) if gate=='rz' else (param or '')
  nodehash=hashlib.sha256(json.dumps([gate,normalparam,qubits,[wire_hash[q] for q in qubits]]).encode()).hexdigest()
  for q in qubits:wire_hash[q]=nodehash
  ops.append({'g':gate,'p':param or '', 'q':qubits,'col':column})
 return {'n':n,'gates':len(ops),'counts':dict(counts),'depth':max(depth),'ops':ops,'hash':hashlib.sha256('|'.join(wire_hash).encode()).hexdigest()}
allsets=[];audit={'filename_mismatches':[],'missing_steps':[],'reward_direction':'outgoing: metadata on snapshot i describes action i -> i+1','reward_mismatches':[]};csvrows=[]
for name in ['barenco_tof_3','gf2^6_mult']:
 paths=[]
 for directory in sorted((BASE/name).iterdir(),key=lambda p:tuple(map(int,p.name.split('_')))):
  if not directory.is_dir():continue
  rows=[]
  for file in directory.glob('*.qasm'):
   fields=list(map(int,file.stem.split('_')));r=parse(file);r.update({'step':fields[0],'cost':fields[1],'reward':fields[2],'node':fields[3],'xfer':fields[4],'file':file.name});rows.append(r)
   if r['gates']!=r['cost']:audit['filename_mismatches'].append(str(file))
  rows.sort(key=lambda r:r['step'])
  if [r['step'] for r in rows]!=list(range(len(rows))):audit['missing_steps'].append(directory.name)
  costs=[r['gates'] for r in rows];delta=[b-a for a,b in zip(costs,costs[1:])];freq=collections.Counter(delta);best=costs[0];barrier=0
  for c in costs:barrier=max(barrier,c-best);best=min(best,c)
  firstdown=next((i for i,c in enumerate(costs) if c<costs[0]),None)
  unique=len(set(r['hash'] for r in rows));rules=collections.Counter();ruledata={}
  for i,r in enumerate(rows[:-1]):
   change=costs[i+1]-costs[i]
   if r['reward']!=-change:audit['reward_mismatches'].append([name,directory.name,i])
   rules[r['xfer']]+=1;v=ruledata.setdefault(str(r['xfer']),{'uses':0,'net_gate_reduction':0,'neutral':0,'up':0,'down':0});v['uses']+=1;v['net_gate_reduction']-=change;v['neutral' if change==0 else 'up' if change>0 else 'down']+=1
   csvrows.append([name,directory.name,i,costs[i],costs[i+1],change,r['reward'],r['node'],r['xfer'],r['depth'],r['hash']])
  summary={'id':directory.name,'start':costs[0],'end':costs[-1],'steps':len(rows)-1,'gain':costs[0]-costs[-1],'peak':max(costs),'barrier':barrier,'neutral':freq[0],'up':sum(v for k,v in freq.items() if k>0),'down':sum(v for k,v in freq.items() if k<0),'first_improvement':firstdown,'unique_structures':unique,'revisited':len(rows)-unique,'rules':dict(rules),'ruledata':ruledata}
  paths.append({'summary':summary,'states':rows})
 totals=collections.Counter();rules=collections.Counter()
 for p in paths:
  totals.update({k:p['summary'][k] for k in ['steps','neutral','up','down','revisited']});rules.update(p['summary']['rules'])
 bestend=min(p['summary']['end'] for p in paths);bestpaths=[p for p in paths if p['summary']['end']==bestend];representative=min(bestpaths,key=lambda p:p['summary']['steps'])
 dataset={'name':name,'paths':paths,'totals':dict(totals),'rules':dict(rules),'best_end':bestend,'representative':representative['summary']['id'],'snapshots':sum(len(p['states']) for p in paths),'baseline_start_max':max(p['summary']['start'] for p in paths)}
 allsets.append(dataset)
summary={'datasets':[{k:v for k,v in d.items() if k!='paths'}|{'path_summaries':[p['summary'] for p in d['paths']]} for d in allsets],'audit':audit,'limits':['Saved successful segments only; no failed searches, times, policy probabilities, or alternative actions.','Segments have different initial circuits and are not independent trials or a single chronological trajectory.','DAG fingerprints identify identical structure modulo independent-gate ordering and Rz 2pi global phase; they do not prove general unitary equality.','Rule IDs have no semantic names without the exact ECC used to generate these paths.']}
(OUT/'data.json').write_text(json.dumps(allsets,separators=(',',':')))
with (OUT/'transitions.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['circuit','segment','step','before_gates','after_gates','delta_after_minus_before','reward_outgoing','action_node','xfer_id','depth_before','structural_hash']);w.writerows(csvrows)
# Overview: four complementary views for each circuit.
fig,axes=plt.subplots(2,4,figsize=(19,9),layout='constrained')
for row,d in enumerate(allsets):
 color=['#2563eb','#7c3aed'][row];ax=axes[row,0]
 for p in d['paths']:
  s=p['summary'];xs=np.linspace(0,1,len(p['states']));ys=[r['gates']-s['start'] for r in p['states']];ax.plot(xs,ys,color=color,alpha=.13,lw=1)
 for p in d['paths']:
  if p['summary']['id']==d['representative']:ax.plot(np.linspace(0,1,len(p['states'])),[r['gates']-p['summary']['start'] for r in p['states']],color='#f97316',lw=2,label='最短最佳终点段 '+p['summary']['id'])
 ax.axhline(0,color='#94a3b8',lw=.8);ax.set(title=d['name']+'：全部轨迹',xlabel='段内进度（归一化）',ylabel='相对起点的门数变化');ax.legend(fontsize=8)
 t=d['totals'];ax=axes[row,1];vals=[t['down'],t['neutral'],t['up']];bars=ax.bar(['减少','不变','增加'],vals,color=['#059669','#64748b','#ef4444'])
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+max(vals)*.02,f'{v}\n{v/t["steps"]:.1%}',ha='center')
 ax.set(title=f'保存段中的 {t["steps"]} 次变换',ylabel='次数');ax.set_ylim(0,max(vals)*1.25)
 ax=axes[row,2];p=next(p for p in d['paths'] if p['summary']['id']==d['representative']);s=p['summary'];cost=[r['gates'] for r in p['states']];ax.plot(cost,color=color,lw=2);ax.plot(np.minimum.accumulate(cost),ls='--',color='#059669',label='段内历史最优');ax.axhline(cost[0],color='#94a3b8',lw=.8);ax.set(title=f'最佳终点代表段 {s["id"]}：{s["steps"]} 步',xlabel='变换步数',ylabel='门数');ax.legend(fontsize=8)
 ax=axes[row,3];top=collections.Counter(d['rules']).most_common(9);ax.barh([str(k) for k,v in top][::-1],[v for k,v in top][::-1],color=color,alpha=.8);ax.set(title='最常见变换 ID（不含终态占位）',xlabel='使用次数',ylabel='xfer ID')
fig.suptitle('高质量优化轨迹：如何经过平台和暂时扩张到达更小的电路\n仅分析已保存成功段；不能据此估计成功率或强化学习收益',fontsize=17)
for ext in ['png','svg','pdf']:fig.savefig(OUT/('overview.'+ext),dpi=180)
plt.close(fig)
# Explicit NumPy simulation: full unitary for all Barenco snapshots; random-vector endpoint check for GF.
indexcache={}
def simulate(state,ops,n):
 state=state.copy()
 def indices(g,qs):
  key=(n,g,tuple(qs))
  if key not in indexcache:
   ids=np.arange(1<<n);target=qs[-1];mask=(ids & (1<<target))==0
   if g=='cx':mask &= (ids & (1<<qs[0]))!=0
   a=ids[mask];indexcache[key]=(a,a|(1<<target))
  return indexcache[key]
 for op in ops:
  g=op['g'];a,b=indices(g,op['q'])
  if g=='h':u=state[a].copy();v=state[b].copy();state[a]=(u+v)/math.sqrt(2);state[b]=(u-v)/math.sqrt(2)
  elif g in ['cx','x']:u=state[a].copy();state[a]=state[b];state[b]=u
  elif g=='rz':theta=angle(op['p']);state[a]*=np.exp(-.5j*theta);state[b]*=np.exp(.5j*theta)
  else:raise ValueError(g)
 return state
checks=[]
bar=allsets[0];reference=simulate(np.eye(32,dtype=complex),bar['paths'][0]['states'][0]['ops'],5);maxerr=0;unique={}
for p in bar['paths']:
 for r in p['states']:
  if r['hash'] in unique:continue
  u=simulate(np.eye(32,dtype=complex),r['ops'],5);overlap=np.vdot(reference,u);phase=overlap/abs(overlap);err=float(np.max(np.abs(u-phase*reference)));maxerr=max(maxerr,err);unique[r['hash']]=err
checks.append({'circuit':bar['name'],'method':'full 32x32 unitary, all distinct structural snapshots, global phase aligned','unique_snapshots':len(unique),'max_abs_error':maxerr,'passed':maxerr<1e-8})
gf=allsets[1];p=next(p for p in gf['paths'] if p['summary']['id']==gf['representative']);states=p['states'];rng=np.random.default_rng(20261002);vector=rng.normal(size=1<<18)+1j*rng.normal(size=1<<18);vector/=np.linalg.norm(vector);ref=simulate(vector,states[0]['ops'],18);errs=[]
for i in sorted(set([len(states)-1,max(range(len(states)),key=lambda j:states[j]['gates'])])):
 v=simulate(vector,states[i]['ops'],18);overlap=np.vdot(ref,v);phase=overlap/abs(overlap);err=float(np.linalg.norm(v-phase*ref));errs.append({'step':i,'gates':states[i]['gates'],'l2_error':err})
checks.append({'circuit':gf['name'],'method':'one seeded random complex statevector on 18 qubits; representative segment start vs peak and endpoint; probabilistic check only','checks':errs,'passed':max(x['l2_error'] for x in errs)<1e-8})
summary['equivalence_checks']=checks;(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'datasets':[{k:v for k,v in d.items() if k not in ['paths','rules']} for d in allsets],'checks':checks,'audit':audit},indent=2))
