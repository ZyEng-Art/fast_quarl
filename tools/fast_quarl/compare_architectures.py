"""Read-only comparison of independent Quarl arms; writes JSON/HTML snapshots."""
import argparse,pathlib,json,time,re,statistics,html
ROOT=pathlib.Path(__file__).resolve().parents[2]
def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(value);temp.replace(path)
def report(registry,output):
    spec=read(registry);rows=[]
    for arm in spec['runs']:
        base=pathlib.Path(arm['path']);status=read(base/'status.json',{});result=read(base/'result.json',{});protocol=read(base/'protocol.json',{})
        run=base/f"{protocol.get('seed',98766)}_{protocol.get('mode','update')}"
        timing=[]
        try:
            for line in (run/'iteration_timing.jsonl').read_text().splitlines():
                try:timing.append(json.loads(line))
                except json.JSONDecodeError:pass
        except FileNotFoundError:pass
        try:
            pairs=re.findall(r'Data for iter (\d+) collected in ([0-9.eE+-]+) s',(run/'train.log').read_text(errors='replace'))
            collection={int(i):float(t) for i,t in pairs}
        except FileNotFoundError:collection={}
        matched=[(r['seconds'],collection[r['iteration']]) for r in timing if r['iteration'] in collection]
        error=read(base/'error.json');target=result.get('first_target_seconds')
        state='completed' if target is not None else 'error' if error else 'running' if status else 'not started'
        curve=read(base/'milestones.json',status.get('milestones',[]));best=100000;points=[]
        for event in curve:
            best=min(best,event['rm_gates']);points.append([event['seconds']/60,best])
        encoder=read(run/'initial.json',{})
        rows.append(dict(arm,phase=state,error=error,best_gates=status.get('best_rm'),elapsed_minutes=status.get('elapsed_seconds',0)/60,first_35_minutes=target/60 if target is not None else None,completed_iterations=len(timing),completed_transitions=timing[-1].get('transitions',0) if timing else 0,mean_iteration_seconds=statistics.mean(r['seconds'] for r in timing) if timing else None,last5_iteration_seconds=statistics.mean(r['seconds'] for r in timing[-5:]) if timing else None,mean_collection_seconds=statistics.mean(x[1] for x in matched) if matched else None,mean_ppo_and_other_seconds=statistics.mean(x[0]-x[1] for x in matched) if matched else None,parameters=encoder.get('parameters'),curve=points,code_commit=protocol.get('code_commit',protocol.get('commit')),last_encoder_stats=timing[-1].get('train_encoder_stats') if timing else None))
    bench=read(ROOT/'results/graphormer/encoder_benchmark.json')
    summary=dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),seed=spec['seed'],target=spec['target_gates'],arms=rows,encoder_benchmark=bench,limitations=['One seed; differing start times and host load. Observed wall times are not a causal architecture ranking.','Model parameter counts differ. All full-graph arms use identical PPO/search budgets and no pretraining.','Iteration time excludes outer checkpoint saving, initialization and concurrent search; first-target time includes warmup and search.','Completed transitions exclude any in-progress rollout. Baseline with subgraphs is a historical completed run.','Graphormer path edge encoding averages all shortest paths rather than picking one; no molecular pretraining or fairseq dependency.'])
    output.mkdir(parents=True,exist_ok=True);atomic(output/'summary.json',json.dumps(summary,indent=2))
    def fmt(x):return '-' if x is None else f'{x:.2f}'
    table=''.join('<tr><td>'+html.escape(r['name'])+'</td><td>'+r['phase']+'</td><td>'+str(r['best_gates'])+'</td><td>'+fmt(r['elapsed_minutes'])+'</td><td>'+fmt(r['first_35_minutes'])+'</td><td>'+str(r['completed_iterations'])+'</td><td>'+f"{r['completed_transitions']:,}"+'</td><td>'+fmt(r['mean_iteration_seconds'])+'</td><td>'+fmt(r['mean_collection_seconds'])+'</td><td>'+fmt(r['mean_ppo_and_other_seconds'])+'</td></tr>' for r in rows)
    payload=json.dumps(summary).replace('</','<\/')
    document="""<!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Quarl architecture comparison</title><style>body{font:15px system-ui;max-width:1250px;margin:30px auto;padding:16px;background:#f7f9fc;color:#183048}table{width:100%;border-collapse:collapse;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}canvas{width:100%;height:380px;background:white;border-radius:10px;margin:20px 0}li{margin:9px 0}</style><h1>Quarl architecture comparison</h1><p>UPDATE_TIME · seed 98766 · verified target 35 gates</p><canvas id="chart" width="1200" height="380"></canvas><table><tr><th>Arm</th><th>Status</th><th>Best gates</th><th>Elapsed min</th><th>First 35 min</th><th>Iterations</th><th>Completed transitions</th><th>Mean iter s</th><th>Collect s</th><th>PPO + other s</th></tr>TABLE_ROWS</table><h2>Interpretation</h2><ul>NOTES</ul><p><a href="summary.json">Raw comparison JSON</a></p><script>const data=PAYLOAD;const c=document.getElementById('chart'),ctx=c.getContext('2d');const colors=['#888','#2879ca','#bd6830','#258a59'];const maxT=Math.max(1,...data.arms.map(r=>r.elapsed_minutes));const px=t=>65+t/maxT*1100,py=g=>320-(g-34)/13*265;ctx.strokeStyle='#aaa';ctx.beginPath();ctx.moveTo(65,35);ctx.lineTo(65,320);ctx.lineTo(1165,320);ctx.stroke();ctx.font='13px system-ui';for(let g=35;g<=46;g++){ctx.fillStyle='#444';ctx.fillText(g,35,py(g)+4);ctx.strokeStyle='#eef1f5';ctx.beginPath();ctx.moveTo(65,py(g));ctx.lineTo(1165,py(g));ctx.stroke()}for(let i=0;i<=6;i++)ctx.fillText((maxT*i/6).toFixed(0),px(maxT*i/6)-8,344);ctx.fillText('Elapsed minutes',540,370);data.arms.forEach((r,i)=>{ctx.strokeStyle=colors[i];ctx.lineWidth=2;ctx.beginPath();let last=null;r.curve.forEach(([t,g])=>{if(last)ctx.lineTo(px(t),py(last[1]));else ctx.moveTo(px(t),py(g));ctx.lineTo(px(t),py(g));last=[t,g]});if(last)ctx.lineTo(px(r.elapsed_minutes),py(last[1]));ctx.stroke();ctx.fillStyle=colors[i];ctx.fillText(r.name,85+i*270,20)});</script></html>"""
    document=document.replace('UPDATE_TIME',summary['updated_utc']).replace('TABLE_ROWS',table).replace('NOTES',''.join('<li>'+html.escape(n)+'</li>' for n in summary['limitations'])).replace('PAYLOAD',payload)
    atomic(output/'index.html',document);print('COMPARISON_UPDATED',summary['updated_utc'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registry',type=pathlib.Path,default=ROOT/'results/graphormer/comparison_runs.json');p.add_argument('--output',type=pathlib.Path,default=ROOT/'runs/architecture_comparison');p.add_argument('--watch',type=int,default=0);a=p.parse_args()
    while True:
        report(a.registry,a.output)
        if not a.watch:break
        time.sleep(a.watch)
