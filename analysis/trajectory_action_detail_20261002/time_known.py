import sys,pathlib,gzip,json,re,time,statistics
BASE=pathlib.Path('/workspace/zengyong/quarl_experiments/actiontrace');sys.path.insert(0,str(BASE/'source/python'))
import quartz
D=json.loads(gzip.decompress((BASE/'action_traces.json.gz').read_bytes()));ctx=quartz.QuartzContext(gate_set=['h','cx','x','rz','add','neg'],filename=str(BASE/'nam_ecc.json'),no_increase=False,include_nop=True)
def sigs(ops,n):
 counts=[0]*n;result={}
 for o in ops:
  q=o['q'];key=(o['g'],o['p'],tuple(q),tuple(counts[k] for k in q));result[o['guid']]=key
  for k in q:counts[k]+=1
 return result

def parse(g):
 ops=[]
 for line in g.to_qasm_with_guids().splitlines():
  m=re.fullmatch(r'(\w+)(?:\((.*?)\))?\s+(.*?); # guid = (\d+)',line)
  if m:
   name,param,targets,guid=m.groups();ops.append({'g':name,'p':param or '', 'q':list(map(int,re.findall(r'q\[(\d+)\]',targets))),'guid':int(guid)})
 return ops
results=[]
for d,name in [(D['datasets'][0],'38_0'),(D['datasets'][1],'370_2')]:
 p=next(p for p in d['paths'] if p['id']==name);keys=[sigs(s['ops'],s['n'])[a['anchor_guid']] for s,a in zip(p['states'],p['actions'])];files=sorted((BASE/'trajectories'/d['name']/name).glob('*.qasm'),key=lambda f:int(f.stem.split('_')[0]));reps=[]
 for repeat in range(3):
  g=quartz.PyGraph.from_qasm(context=ctx,filename=str(files[0]));start=time.perf_counter();apply_time=0;mapping_time=0
  for i,a in enumerate(p['actions']):
   tick=time.perf_counter();keys_actual=sigs(parse(g),p['states'][i]['n']);bykey={v:k for k,v in keys_actual.items()};guid=bykey[keys[i]];nodes={int(n.guid):n for n in g.nodes};xfer=ctx.get_xfer_from_id(id=a['xfer']);mapping_time+=time.perf_counter()-tick
   tick=time.perf_counter();g=g.apply_xfer(xfer=xfer,node=nodes[guid],eliminate_rotation=True);apply_time+=time.perf_counter()-tick
   assert g is not None and int(g.hash())==p['states'][i+1]['hash'],(d['name'],name,i)
  reps.append({'total_seconds':time.perf_counter()-start,'native_apply_seconds':apply_time,'anchor_mapping_seconds':mapping_time,'final_gates':g.gate_count})
 result={'circuit':d['name'],'path':name,'actions':len(p['actions']),'repetitions':reps,'median_total_seconds':statistics.median(r['total_seconds'] for r in reps),'median_native_apply_seconds':statistics.median(r['native_apply_seconds'] for r in reps),'scope':'known saved actions; no exploration, GNN inference, DGL construction, PPO update, or alternative candidate matching; excludes imports, ECC initialization, and initial parsing; native hashing checked at every step'};results.append(result);print('KNOWN_PATH',json.dumps(result),flush=True)
(BASE/'known_path_timing.json').write_text(json.dumps(results,indent=2))
