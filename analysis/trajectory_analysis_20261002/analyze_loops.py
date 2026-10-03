from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
out=Path(__file__).resolve().parent;data=json.loads((out/'data.json').read_text());summary=json.loads((out/'summary.json').read_text())
font_manager.fontManager.addfont('/System/Library/Fonts/STHeiti Medium.ttc');plt.rcParams['font.family']=font_manager.FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc').get_name();plt.rcParams['svg.fonttype']='none'
fig,axs=plt.subplots(2,2,figsize=(14,8),layout='constrained')
for d,ds,axes in zip(data,summary['datasets'],axs):
 for p in d['paths']:
  stack=[];positions={}
  for i,r in enumerate(p['states']):
   h=r['hash']
   if h in positions:
    j=positions[h]
    for old in stack[j+1:]:positions.pop(p['states'][old]['hash'],None)
    stack=stack[:j+1];stack[-1]=i
   else:positions[h]=len(stack);stack.append(i)
  p['summary']['loop_erased_indices']=stack;p['summary']['loop_erased_steps']=len(stack)-1
  next(s for s in ds['path_summaries'] if s['id']==p['summary']['id']).update({'loop_erased_steps':len(stack)-1,'loop_erased_indices':stack})
 p=next(p for p in d['paths'] if p['summary']['id']==d['representative']);s=p['summary'];r=p['states'];indices=s['loop_erased_indices'];axes[0].plot([x['gates'] for x in r],color='#64748b',label=f'原保存路径：{s["steps"]} 步');axes[0].scatter(indices,[r[i]['gates'] for i in indices],color='#f97316',s=12,label='删去结构闭环后保留的状态');axes[0].set(title=f'{d["name"]} / {s["id"]}',xlabel='原始步号',ylabel='门数');axes[0].legend()
 axes[1].plot([r[i]['gates'] for i in indices],marker='o',markersize=3,color='#2563eb');axes[1].set(title=f'结构去环示意：{s["steps"]} → {s["loop_erased_steps"]} 步',xlabel='去环后状态序号',ylabel='门数')
 print(d['name'],s['steps'],s['loop_erased_steps'],[r[i]['gates'] for i in indices])
fig.suptitle('成功轨迹也包含可删去的结构闭环\n按 DAG 指纹去环的示意；尚未重放规则，不能直接作为可执行压缩轨迹',fontsize=15)
for ext in ['png','svg','pdf']:fig.savefig(out/('loops.'+ext),dpi=170)
summary['structural_loop_erasure_note']='Illustrative DAG-fingerprint loop removal, not executable rule replay; node identifiers may need remapping.'
(out/'data.json').write_text(json.dumps(data,separators=(',',':')));(out/'summary.json').write_text(json.dumps(summary,indent=2))
