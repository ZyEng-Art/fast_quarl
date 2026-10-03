from pathlib import Path
import json,gzip,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,patches
OUT=Path(__file__).resolve().parent
font_manager.fontManager.addfont('/System/Library/Fonts/STHeiti Medium.ttc');plt.rcParams['font.family']=font_manager.FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc').get_name();plt.rcParams['svg.fonttype']='none'
summary=json.loads((OUT/'action_summary.json').read_text());data=json.loads(gzip.decompress((OUT/'action_traces.json.gz').read_bytes()))
categories=['uses_previous_output','adjacent_region','same_qubits_elsewhere','disjoint_qubits'];names=['使用前一步产物','产物一跳 / 上游边界','共享量子位','量子位不相交'];colors=['#7c3aed','#0d9488','#f59e0b','#64748b']
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for d,ax in zip(summary['datasets'],axes):
 vals=[d['categories'].get(k,0) for k in categories];bars=ax.bar(names,vals,color=colors)
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+max(vals)*.02,f'{v}\n{v/d["pairs"]:.1%}',ha='center')
 ax.set(title=f'{d["name"]}：{d["pairs"]} 对相邻动作',ylabel='动作对数',ylim=(0,max(vals)*1.25));ax.tick_params(axis='x',labelsize=9)
fig.suptitle('下一动作不一定接着改写上一动作的产物\n按重放一致的匹配恢复；每一步原生后继哈希与保存状态一致',fontsize=15)
for ext in ['png','svg','pdf']:fig.savefig(OUT/('action_relations.'+ext),dpi=180)
plt.close(fig)
def draw(ax,state,colormap,title,labels):
 nodes={o['guid']:o for o in state['ops']};adj=collections.defaultdict(set);pred=collections.defaultdict(list)
 for e in state['edges']:adj[e['a']].add(e['b']);adj[e['b']].add(e['a']);pred[e['b']].append(e['a'])
 selected=set(colormap)
 for g in list(selected):selected|=adj[g]
 layer={}
 for o in state['ops']:layer[o['guid']]=1+max([layer[p] for p in pred[o['guid']]] or [0])
 levels=sorted(set(layer[g] for g in selected));positions={};taken=collections.defaultdict(list)
 for o in state['ops']:
  g=o['guid']
  if g not in selected:continue
  x=levels.index(layer[g]);y=-sum(o['q'])/len(o['q'])
  while any(abs(y-v)<1.4 for v in taken[x]):y-=1.5
  taken[x].append(y);positions[g]=(x,y)
 for e in state['edges']:
  a,b=e['a'],e['b']
  if a in positions and b in positions:
   ax.add_patch(patches.FancyArrowPatch(positions[a],positions[b],arrowstyle='-|>',mutation_scale=9,shrinkA=23,shrinkB=23,color='#94a3b8',lw=.8,zorder=1))
 for g,(x,y) in positions.items():
  o=nodes[g];c=colormap.get(g,'#94a3b8');ax.text(x,y,f'v{labels[g]} {o["g"].upper()}\n'+','.join('q'+str(q) for q in o['q']),ha='center',va='center',fontsize=7,bbox={'boxstyle':'round,pad=.25','facecolor':'white','edgecolor':c,'linewidth':2 if g in colormap else .8},zorder=3,color=c)
 if positions:ax.set_xlim(-.6,len(levels)-.4);ax.set_ylim(min(y for x,y in positions.values())-1,max(y for x,y in positions.values())+1)
 ax.set_title(title,fontsize=11);ax.axis('off')
fig,axes=plt.subplots(2,3,figsize=(18,10),layout='constrained');examples=[]
for row,d in enumerate(data['datasets']):
 desired='38_4' if row==0 else '370_2';p=next(p for p in d['paths'] if p['id']==desired)
 productive=[i for i,r in enumerate(p['relations']) if r['category']=='uses_previous_output' and p['actions'][i+1]['after_gates']<p['actions'][i]['before_gates']];pair=productive[-1] if productive else next(i for i,r in enumerate(p['relations']) if r['category']=='uses_previous_output')
 labels={}
 for s in p['states']:
  for o in s['ops']:
   if o['guid'] not in labels:labels[o['guid']]=len(labels)
 A=p['actions'][pair];B=p['actions'][pair+1];maps=[{g:'#dc2626' for g in A['source']},{g:'#059669' for g in A['created_surviving']},{g:'#2563eb' for g in B['created_surviving']}]
 for g in B['source']:maps[1][g]='#7c3aed' if g in maps[1] else '#f59e0b'
 for j in range(3):
  s=p['states'][pair+j];draw(axes[row,j],s,maps[j],f'{d["name"]} / {p["id"]}：状态 {pair+j}，{s["gates"]} 门\n'+['A 匹配的旧子图（红）','A 产物（绿） + B 匹配（橙） = 重叠（紫）','B 生成并保留的新子图（蓝）'][j],labels)
 examples.append({'circuit':d['name'],'path':p['id'],'action_A':pair,'action_B':pair+1,'xfer_A':A['xfer'],'xfer_B':B['xfer'],'shared_output_nodes':[labels[g] for g in p['relations'][pair]['consumed_previous_created']]})
fig.suptitle('门级 DAG 上的连续改写实例：紫色节点说明 B 在继续改写 A 的产物\n只显示匹配区域及一跳邻域；v 编号追踪持续存活的门，箭头表示门依赖',fontsize=15)
for ext in ['png','svg','pdf']:fig.savefig(OUT/('dag_examples.'+ext),dpi=180)
(OUT/'examples.json').write_text(json.dumps(examples,indent=2));print(json.dumps(examples))
