# coding: utf-8
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
out=Path(__file__).resolve().parent
s=json.loads((out/'subgraph_sizes.json').read_text());font=FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')
fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for ax,(key,title) in zip(axes.flat,[('source_gates','规则匹配源门数'),('affected_before_gates','改写前影响范围：包含接线边界'),('source_qubits','规则涉及量子位数'),('two_hop_gates','源门及两跳邻域：展示范围')]):
 for d,c in zip(s['datasets'],['#7c3aed','#0d9488']):
  h=d['metrics'][key]['histogram'];ax.plot([x for x,n in h],[100*n/d['actions'] for x,n in h],marker='o',color=c,label=d['dataset'])
 ax.set_title(title,fontproperties=font);ax.set_xlabel('门数' if key!='source_qubits' else '量子位数',fontproperties=font);ax.set_ylabel('动作占比（%）',fontproperties=font);ax.grid(alpha=.2);ax.legend()
fig.suptitle('3513 个已保存动作的局部子图大小分布',fontproperties=font,fontsize=20)
for ext in ['png','svg','pdf']:fig.savefig(out/('subgraph_sizes.'+ext),dpi=180)
