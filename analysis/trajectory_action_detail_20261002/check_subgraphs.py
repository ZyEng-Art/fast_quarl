# coding: utf-8
from pathlib import Path
import json,gzip
from playwright.sync_api import sync_playwright
out=Path(__file__).resolve().parent
stats=json.loads((out/'subgraph_sizes.json').read_text());data=json.loads(gzip.decompress((out/'viewer_data.json.gz').read_bytes()))
assert len(stats['rows'])==3513
for d in data['datasets']:
 for p in d['paths']:
  for a in p['actions']:
   f=a['footprint'];assert len(a['footprint_before'])==f['affected_before_gates'];assert len(a['footprint_after'])==f['affected_after_gates'];assert set(a['source'])<=set(a['footprint_before']);assert len(a['source'])==f['source_gates']
with sync_playwright() as pw:
 b=pw.chromium.launch(channel='chrome',headless=True);page=b.new_page(viewport={'width':1500,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((out/'action_viewer.html').as_uri());page.wait_for_selector('#sizeHistogram rect',timeout=60000)
 assert '935' in page.locator('#sizeStats').inner_text()
 for ds,total in [('0','935'),('1','2578')]:
  page.locator('#dataset').select_option(ds);page.locator('#sizeScope').select_option('dataset');assert total in page.locator('#sizeStats').inner_text()
  for value in page.locator('#sizeMetric option').evaluate_all('(es)=>es.map(e=>e.value)'):
   page.locator('#sizeMetric').select_option(value);assert page.locator('#sizeHistogram rect').count()>0
  page.locator('#sizeScope').select_option('path');n=page.evaluate('P.actions.length');assert str(n) in page.locator('#sizeStats').inner_text()
  page.locator('#sizeTrace circle[data-size-action="0"]').first.click();assert page.evaluate('pair')==0
  assert '影响范围' in page.locator('#bindings').inner_text()
  page.locator('#footprint').check();assert page.locator('#state0 svg rect[stroke="#0891b2"]').count()>0
  page.screenshot(path=str(out/('viewer_subgraphs_'+ds+'.png')),full_page=True)
  page.locator('#footprint').uncheck()
 assert not errors,errors;b.close()
print('3513 footprints consistent; both datasets, all 9 metrics, scope, jump and DAG boundary passed')
