from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path(__file__).resolve().parent
with sync_playwright() as p:
 browser=p.chromium.launch(channel='chrome',headless=True)
 page=browser.new_page(viewport={'width':1650,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((out/'action_viewer.html').as_uri());page.wait_for_selector('#state1 svg',timeout=60000)
 print('Barenco path count',page.locator('#path option').count());assert page.locator('#path option').count()==52
 page.locator('#path').select_option('38_4');page.locator('#pair').evaluate("e=>{e.value=2;e.dispatchEvent(new Event('input'))}")
 print('Barenco relation',page.locator('#relation').inner_text());assert 'B 使用了 A 生成的门' in page.locator('#relation').inner_text()
 assert page.locator('#state0 svg').count()==1 and page.locator('#state2 svg').count()==1
 page.screenshot(path=str(out/'viewer_barenco_dag.png'),full_page=True)
 for state in ['state0','state1','state2','dag','actiondag']:(out/('barenco_'+state+'.svg')).write_text(page.locator('#'+state+' svg').evaluate('e=>e.outerHTML'))
 page.locator('#dataset').select_option('1');assert page.locator('#path option').count()==81
 page.locator('#direct').click();print('GF direct relation',page.locator('#relation').inner_text())
 page.locator('#jump').click();print('GF jump relation',page.locator('#relation').inner_text());assert '集合不相交' in page.locator('#relation').inner_text()
 page.locator('#local').uncheck();assert page.locator('#state1 svg g').count()>=370
 page.locator('#local').check();page.screenshot(path=str(out/'viewer_gf_dag.png'),full_page=True)
 assert not errors,errors;print('DAG controls passed; browser errors',errors);browser.close()
