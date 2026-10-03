import os,sys,pathlib,json,re,gzip,time,collections
BASE=pathlib.Path('/workspace/zengyong/quarl_experiments/actiontrace');sys.path.insert(0,str(BASE/'source/python'))
import quartz
ROOT=BASE.parent/'Quarl-xpu';ctx=quartz.QuartzContext(gate_set=['h','cx','x','rz','add','neg'],filename=str(BASE/'nam_ecc.json'),no_increase=False,include_nop=True)

def snapshot(g):
 ops=[];n=None
 for line in g.to_qasm_with_guids().splitlines():
  if line.startswith('qreg'):n=int(re.search(r'\[(\d+)\]',line).group(1))
  m=re.fullmatch(r'(\w+)(?:\((.*?)\))?\s+(.*?); # guid = (\d+)',line)
  if m:
   name,param,args,guid=m.groups();ops.append({'g':name,'p':param or '', 'q':list(map(int,re.findall(r'q\[(\d+)\]',args))),'guid':int(guid)})
 assert len(ops)==g.gate_count,(len(ops),g.gate_count)
 ids=[int(node.guid) for node in g.nodes];edges=[{'a':ids[a],'b':ids[b],'ap':ap,'bp':bp} for a,b,ap,bp in g.all_edges()]
 return {'n':n,'gates':g.gate_count,'ops':ops,'edges':edges,'hash':int(g.hash())}

def classify(previous,current,intermediate):
 created=set(previous['created_surviving']);source=set(current['source']);overlap=created&source
 anchor=current['anchor_guid'];effective_old=set(previous['effective_removed']);pred=set(previous['predecessors_surviving']);qprev=set(previous['qubits']);qcur=set(current['qubits'])
 adjacency=collections.defaultdict(set)
 for e in intermediate['edges']:adjacency[e['a']].add(e['b']);adjacency[e['b']].add(e['a'])
 seen=set(created);front=set(created);distance=0;found=None
 while front:
  if front&source:found=distance;break
  front={b for a in front for b in adjacency[a]}-seen;seen|=front;distance+=1
 if overlap:label='uses_previous_output'
 elif source&pred or found==1:label='adjacent_region'
 elif qprev&qcur:label='same_qubits_elsewhere'
 else:label='disjoint_qubits'
 return {'category':label,'consumed_previous_created':sorted(overlap),'anchor_on_previous_created':anchor in created,'source_on_previous_predecessors':sorted(source&pred),'distance_from_previous_created':found,'shared_qubits':sorted(qprev&qcur)}

out={'metadata':{'context_xfers':ctx.num_xfers,'ecc_file':'nam_ecc.json','method':'native binding trace with persistent GUIDs, each action replayed and native successor hash checked against saved QASM','anchor_recovery':'prefer saved ID if valid; otherwise find anchor reproducing saved successor; alternative anchors are not exhaustively checked','normalization':'eliminate_rotation=True; declared and surviving destination gates separately recorded','relation':'match-source overlap with surviving created GUIDs; graph adjacency and shared qubits shown separately'},'datasets':[],'errors':[]}
start=time.time()
for circuit in ['barenco_tof_3','gf2^6_mult']:
 paths=[]
 # Best-terminal segments first, so a useful preview is available early.
 dirs=sorted((BASE/'trajectories'/circuit).iterdir(),key=lambda p:tuple(map(int,p.name.split('_'))))
 for directory in dirs:
  if not directory.is_dir():continue
  files=sorted(directory.glob('*.qasm'),key=lambda p:int(p.stem.split('_')[0]));fields=[[int(part) for part in f.stem.split('_')] for f in files]
  g=quartz.PyGraph.from_qasm(context=ctx,filename=str(files[0]));states=[snapshot(g)];actions=[];relations=[];error=None
  for i in range(len(files)-1):
   step,cost,reward,savedid,xferid=fields[i];target=quartz.PyGraph.from_qasm(context=ctx,filename=str(files[i+1]));targethash=int(target.hash());xfer=ctx.get_xfer_from_id(id=xferid)
   order=list(range(len(g.nodes)))
   if savedid in order:order.remove(savedid);order.insert(0,savedid)
   found=None
   for nodeid in order:
    node=g.get_node_from_id(id=nodeid)
    if not g.xfer_appliable(xfer=xfer,node=node):continue
    result=g.apply_xfer_with_binding_trace(xfer=xfer,node=node,eliminate_rotation=True,predecessor_layers=0)
    if result[0] is not None and int(result[0].hash())==targethash:
     found=(nodeid,int(node.guid),result);break
   if found is None:
    error={'circuit':circuit,'path':directory.name,'step':i,'saved_node_id':savedid,'xfer':xferid,'error':'No matching replay anchor in this ECC'};out['errors'].append(error);print('TRACE_FAILED',json.dumps(error),flush=True);break
   nodeid,anchor,(nextgraph,_,sourceguids,destguids)=found;before=states[-1];after=snapshot(nextgraph);old={o['guid'] for o in before['ops']};live={o['guid'] for o in after['ops']};src=set(map(int,sourceguids));dst=set(map(int,destguids));pred={e['a'] for e in before['edges'] if e['b'] in src and e['a'] not in src};q={q for o in before['ops'] if o['guid'] in src for q in o['q']}
   action={'step':i,'xfer':xferid,'saved_node_id':savedid,'resolved_node_id':nodeid,'anchor_guid':anchor,'source':sorted(src),'declared_destination':sorted(dst),'created_surviving':sorted(dst&live),'normalized_away_destination':sorted(dst-live),'effective_created':sorted(live-old),'effective_removed':sorted(old-live),'normalization_removed_outside_source':sorted((old-live)-src),'predecessors_surviving':sorted(pred&live),'qubits':sorted(q),'before_gates':g.gate_count,'after_gates':nextgraph.gate_count,'reward':reward,'successor_hash_matches':True}
   assert len(src)==xfer.src_gate_count,(circuit,directory.name,i,len(src),xfer.src_gate_count)
   if actions:relations.append(classify(actions[-1],action,before))
   actions.append(action);states.append(after);g=nextgraph
  if error is None:
   p={'id':directory.name,'states':states,'actions':actions,'relations':relations};paths.append(p)
   print('TRACE_OK',circuit,directory.name,len(actions),dict(collections.Counter(r['category'] for r in relations)),flush=True)
  if len(paths)%5==0:
   with gzip.open(BASE/'preview.json.gz','wt') as f:json.dump({**out,'current':{'name':circuit,'paths':paths}},f,separators=(',',':'))
 out['datasets'].append({'name':circuit,'paths':paths})
 with gzip.open(BASE/'action_traces.json.gz','wt') as f:json.dump(out,f,separators=(',',':'))
 print('CIRCUIT_DONE',circuit,len(paths),'seconds',time.time()-start,flush=True)
summary={'seconds':time.time()-start,'metadata':out['metadata'],'errors':out['errors'],'datasets':[]}
for d in out['datasets']:
 counts=collections.Counter(r['category'] for p in d['paths'] for r in p['relations']);pairs=sum(counts.values());summary['datasets'].append({'name':d['name'],'paths':len(d['paths']),'actions':sum(len(p['actions']) for p in d['paths']),'pairs':pairs,'categories':dict(counts),'direct_output_anchor_pairs':sum(r['anchor_on_previous_created'] for p in d['paths'] for r in p['relations']),'path_counts':[{'id':p['id'],'actions':len(p['actions']),'categories':dict(collections.Counter(r['category'] for r in p['relations']))} for p in d['paths']]})
(BASE/'action_summary.json').write_text(json.dumps(summary,indent=2));print('EXPORT_DONE',json.dumps(summary),flush=True)
