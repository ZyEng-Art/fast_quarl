from pathlib import Path
import gzip,json,collections,statistics,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
OUT=Path(__file__).resolve().parent
font_manager.fontManager.addfont('/System/Library/Fonts/STHeiti Medium.ttc');plt.rcParams['font.family']=font_manager.FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc').get_name();plt.rcParams['svg.fonttype']='none'
data=json.loads(gzip.decompress((OUT/'action_traces.json.gz').read_bytes()))
def metrics(parents):
 depth=[];chain=[]
 for i,ps in enumerate(parents):
  pred=max(ps,key=lambda j:depth[j],default=None);depth.append(1+(depth[pred] if pred is not None else 0));chain.append(pred)
 last=max(range(len(depth)),key=lambda i:depth[i]);critical=[]
 while last is not None:critical.append(last);last=chain[last]
 layers=collections.Counter(depth)
 return {'critical_layers':max(depth),'max_asap_layer_size':max(layers.values()),'unit_cost_speedup_bound':len(depth)/max(depth),'layers':dict(sorted(layers.items())),'critical_action_chain':critical[::-1],'parents':[sorted(s) for s in parents]}
def footprints(p,i):
 a=p['actions'][i];before=p['states'][i];after=p['states'][i+1];src=set(a['source']);reads={('node',g) for g in src};writes={('node',g) for g in a['effective_removed']+a['effective_created']}
 oldops={o['guid']:o for o in before['ops']};newops={o['guid']:o for o in after['ops']}
 for g in oldops.keys()&newops.keys():
  if oldops[g]!=newops[g]:writes.add(('node',g))
 for g in src:
  for port in range(len(oldops[g]['q'])):reads.add(('out',g,port));reads.add(('in',g,port))
 def edges(s):return {(e['a'],e['b'],e['ap'],e['bp']) for e in s['edges']}
 oldedges=edges(before);newedges=edges(after)
 for x,y,xp,yp in oldedges:
  if x in src or y in src:reads.add(('out',x,xp));reads.add(('in',y,yp))
 for x,y,xp,yp in oldedges^newedges:writes.add(('out',x,xp));writes.add(('in',y,yp))
 # Explicit input/output boundaries, absent from gate-only native edge lists.
 def endpoints(state):
  roots={};ends={}
  for o in state['ops']:
   for q in o['q']:roots.setdefault(q,o['guid']);ends[q]=o['guid']
  return roots,ends
 r0,e0=endpoints(before);r1,e1=endpoints(after)
 for kind,old,new in [('wire_input',r0,r1),('wire_output',e0,e1)]:
  for q,g in old.items():
   if g in src:reads.add((kind,q))
   if new[q]!=g:writes.add((kind,q))
 return reads,writes
results=[]
for d in data['datasets']:
 rows=[]
 for p in d['paths']:
  prod={};producer=[]
  for a in p['actions']:
   producer.append({prod[g] for g in a['source'] if g in prod})
   for g in a['created_surviving']:prod[g]=a['step']
  lastwrite={};readers=collections.defaultdict(set);boundary=[]
  for i,a in enumerate(p['actions']):
   R,W=footprints(p,i);ps=set(producer[i])
   for k in R|W:
    if k in lastwrite:ps.add(lastwrite[k])
   for k in W:ps|=readers[k]
   boundary.append(ps)
   for k in W:lastwrite[k]=i;readers[k].clear()
   for k in R-W:readers[k].add(i)
  lastqubit={};qubit=[]
  for a in p['actions']:
   qubit.append({lastqubit[q] for q in a['qubits'] if q in lastqubit})
   for q in a['qubits']:lastqubit[q]=a['step']
  row={'path':p['id'],'actions':len(p['actions']),'producer_only':metrics(producer),'boundary_conflicts':metrics(boundary),'whole_qubit_serialization':metrics(qubit)};rows.append(row)
 results.append({'name':d['name'],'paths':rows,'summary':{'segments':len(rows),'median_producer_depth_ratio':statistics.median(r['producer_only']['critical_layers']/r['actions'] for r in rows),'median_boundary_depth_ratio':statistics.median(r['boundary_conflicts']['critical_layers']/r['actions'] for r in rows),'segments_with_producer_chain_over_two':sum(r['producer_only']['critical_layers']>2 for r in rows)}})
report={'datasets':results,'definitions':{'producer_only':'Read-after-create dependency: an action consumes a surviving new gate from an earlier action. This omits edge/boundary hazards and gives an optimistic lower bound on rounds for this recovered trace.','boundary_conflicts':'Conservative read/write hazards on matched nodes, incident edge ports, changed edge ports, and wire input/output boundaries; includes producer dependencies. Proposed scheduling model, not validated by simultaneous rewrite replay.','whole_qubit_serialization':'Serialize all source actions sharing a physical qubit, regardless of distance; conservative locality heuristic, not a precise graph-conflict test.','unit_cost_speedup_bound':'N / critical_layers, assuming all original actions retained, unit action duration, unlimited workers, and no synchronization cost. Not a measured speedup.'},'scope':'These are saved successful trajectories with one recovered binding per action; loops may be redundant. Bounds do not apply to every possible optimization algorithm, macro actions, or other trajectories.'}
(OUT/'parallelism.json').write_text(json.dumps(report,indent=2))
selected=[(results[0],'38_4'),(results[0],'38_0'),(results[0],'38_3'),(results[0],'38_5'),(results[1],'370_2'),(results[1],'371_5')]
with (OUT/'parallelism.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['circuit','segment','actions','producer_layers','boundary_layers','qubit_layers','optimistic_unit_cost_speedup_bound'])
 for d in results:
  for r in d['paths']:w.writerow([d['name'],r['path'],r['actions'],r['producer_only']['critical_layers'],r['boundary_conflicts']['critical_layers'],r['whole_qubit_serialization']['critical_layers'],r['producer_only']['unit_cost_speedup_bound']])
fig,axes=plt.subplots(1,2,figsize=(15,5.8),layout='constrained')
for ax,d,ids in [(axes[0],results[0],['38_4','38_0','38_3','38_5']),(axes[1],results[1],['370_2','371_5'])]:
 rr=[next(r for r in d['paths'] if r['path']==k) for k in ids];xx=range(len(rr));width=.24
 for shift,key,label,color in [(-1,'actions','原始动作数','#94a3b8'),(0,'producer_only','仅产物依赖：乐观最少轮数','#7c3aed'),(1,'boundary_conflicts','计入边界冲突：保守层数','#0d9488')]:
  vals=[r[key] if key=='actions' else r[key]['critical_layers'] for r in rr];bars=ax.bar([x+shift*width for x in xx],vals,width,label=label,color=color)
  for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+max(r['actions'] for r in rr)*.012,str(v),ha='center',fontsize=9)
 ax.set_xticks(list(xx),ids);ax.set(title=d['name'],xlabel='轨迹段',ylabel='动作数 / 依赖层数',ylim=(0,max(r['actions'] for r in rr)*1.2));ax.legend(fontsize=8)
fig.suptitle('有并行空间，但这些轨迹不能压成两轮：最长依赖链决定下限\n只分析保留原有动作的调度；非实测加速比，尚未并行重放边界冲突模型',fontsize=14)
for ext in ['png','svg','pdf']:fig.savefig(OUT/('parallelism.'+ext),dpi=180)
for d,key in selected:
 r=next(r for r in d['paths'] if r['path']==key);print(d['name'],key,r['actions'],r['producer_only']['critical_layers'],r['boundary_conflicts']['critical_layers'],r['whole_qubit_serialization']['critical_layers'],round(r['producer_only']['unit_cost_speedup_bound'],2))
print('dataset summaries',[(d['name'],d['summary']) for d in results])
