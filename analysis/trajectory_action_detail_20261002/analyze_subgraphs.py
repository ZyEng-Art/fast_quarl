from pathlib import Path
import gzip,json,csv
import numpy as np
out=Path(__file__).resolve().parent
data=json.loads(gzip.decompress((out/'action_traces.json.gz').read_bytes()))
rows=[];summaries=[]
def neighborhood(s,seed,hops):
 adj={o['guid']:set() for o in s['ops']}
 for e in s['edges']:adj[e['a']].add(e['b']);adj[e['b']].add(e['a'])
 seen=set(seed)&adj.keys();front=set(seen)
 for _ in range(hops):
  front={v for u in front for v in adj[u]}-seen;seen|=front
 return seen
for d in data['datasets']:
 dr=[]
 for p in d['paths']:
  for i,a in enumerate(p['actions']):
   s,t=p['states'][i:i+2];before={o['guid']:o for o in s['ops']};after={o['guid']:o for o in t['ops']}
   source=set(a['source']);removed=set(a['effective_removed']);created=set(a['effective_created'])
   modified={g for g in before.keys()&after.keys() if (before[g]['g'],before[g]['p'],before[g]['q'])!=(after[g]['g'],after[g]['p'],after[g]['q'])}
   old={(e['a'],e['b'],e['ap'],e['bp']) for e in s['edges']};new={(e['a'],e['b'],e['ap'],e['bp']) for e in t['edges']};delta=old^new;ends={g for e in delta for g in e[:2]}
   affected_before=(source|removed|modified|ends)&before.keys();affected_after=(created|modified|ends)&after.keys()
   q={q for g in affected_before for q in before[g]['q']}|{q for g in affected_after for q in after[g]['q']}
   r={'dataset':d['name'],'path':p['id'],'step':a['step'],'xfer':a['xfer'],'circuit_gates':s['gates'],'source_gates':len(source),'declared_destination_gates':len(a['declared_destination']),'removed_gates':len(removed),'created_gates':len(created),'modified_surviving_gates':len(modified),'changed_nodes_total':len(removed)+len(created)+len(modified),'outside_source_removed':len(a['normalization_removed_outside_source']),'source_qubits':len({q for g in source for q in before[g]['q']}),'affected_qubits':len(q),'affected_before_gates':len(affected_before),'affected_after_gates':len(affected_after),'boundary_surviving_gates':len((ends&before.keys()&after.keys())-modified),'removed_edges':len(old-new),'added_edges':len(new-old),'changed_edges':len(delta),'one_hop_gates':len(neighborhood(s,source,1)),'two_hop_gates':len(neighborhood(s,source,2)),'source_fraction':len(source)/s['gates'],'affected_before_fraction':len(affected_before)/s['gates']}
   rows.append(r);dr.append(r)
 metrics={}
 for key in dr[0]:
  if key in ['dataset','path','step','xfer']:continue
  v=np.array([r[key] for r in dr]);u,c=np.unique(v,return_counts=True)
  metrics[key]={'min':float(v.min()),'median':float(np.median(v)),'mean':float(v.mean()),'p90':float(np.percentile(v,90)),'p95':float(np.percentile(v,95)),'max':float(v.max()),'histogram':[[float(x),int(n)] for x,n in zip(u,c)]}
 summaries.append({'dataset':d['name'],'paths':len(d['paths']),'actions':len(dr),'metrics':metrics,'actions_with_outside_source_removal':sum(r['outside_source_removed']>0 for r in dr)})
result={'definitions':{'source_gates':'规则原生 binding 中的匹配旧门数。','changed_nodes_total':'删除旧门＋新增存活门＋属性改变的存活门；前后不同状态的节点合计，不是单个子图的门数。','affected_before_gates':'重写前：匹配源门、额外删除/属性改变的门、变化依赖边端点的并集。包含接线边界，不是受影响的所有下游门。','affected_after_gates':'重写后：新增/属性改变的门与变化依赖边端点的并集。','changed_edges':'前后 (源GUID,目标GUID,源端口,目标端口) 的对称差；并行量子位边分别计数，不包含电路输入输出边。','one_hop_gates':'重写前，以源门为种子沿无向门依赖边扩展一跳，包含源门。','two_hop_gates':'重写前，以源门为种子沿无向门依赖边扩展两跳，包含源门。'},'limitations':['按保存成功轨迹的 action 加权，不代表完整搜索中候选动作的分布。','使用能重现后继图的一种匹配，原始 binding 未完整保存，存在匹配歧义。','接线影响范围不等于 GNN 缓存失效范围；后者还取决于网络层数和归一化。'],'datasets':summaries,'rows':rows}
(out/'subgraph_sizes.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')))
with (out/'subgraph_sizes.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for d in summaries:
 print(d['dataset'],d['actions'],'outside',d['actions_with_outside_source_removal'])
 for k in ['source_gates','source_qubits','changed_nodes_total','affected_before_gates','affected_before_fraction','two_hop_gates']:
  print(k,{q:round(v,4) for q,v in d['metrics'][k].items() if q!='histogram'})
