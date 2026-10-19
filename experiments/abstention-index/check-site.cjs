const assert=require('assert'),fs=require('fs'),path=require('path');
const pptr=require('/home/rka/codex-browser-automation/node_modules/puppeteer-core');
const root=path.resolve(__dirname,'../..');
const base=process.env.ABSTENTION_INDEX_BASE||'http://127.0.0.1:8788/';
const qa=process.env.ABSTENTION_INDEX_QA||path.join(root,'artifacts/abstention-index-qa');
const catalog=JSON.parse(fs.readFileSync(path.join(root,'experiments/abstention-index/catalog.json')));
fs.mkdirSync(qa,{recursive:true});
(async()=>{let browser;try{
 browser=await pptr.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-first-run']});
 const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.setViewport({width:1440,height:1100,deviceScaleFactor:1});
 const res=await page.goto(new URL('results/abstention/index.html',base).href,{waitUntil:'networkidle0'});assert.equal(res.status(),200);
 const rows=await page.$$eval('tbody tr',xs=>xs.map(r=>({title:r.querySelector('th').textContent,description:r.querySelectorAll('td')[1].textContent,links:[...r.querySelectorAll('a')].map(a=>a.href)})));
 assert.equal(rows.length,catalog.length);
 for(let i=0;i<rows.length;i++){assert.equal(rows[i].title,catalog[i].title);assert.equal(rows[i].description,catalog[i].description)}
 assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
 await page.screenshot({path:path.join(qa,'desktop.png'),fullPage:true});
 await page.type('#search','vampire');assert.equal(await page.$$eval('tbody tr',rs=>rs.filter(r=>!r.hidden).length),catalog.filter(r=>Object.values(r).join(' ').toLowerCase().includes('vampire')).length);
 await page.$eval('#search',x=>{x.value='';x.dispatchEvent(new Event('input'))});assert.equal(await page.$$eval('tbody tr',rs=>rs.filter(r=>!r.hidden).length),21);
 await page.type('#search','no-such-experiment');assert.equal(await page.$$eval('tbody tr',rs=>rs.filter(r=>!r.hidden).length),0);
 await page.$eval('#search',x=>{x.value='';x.dispatchEvent(new Event('input'))});
 await page.setViewport({width:390,height:844,deviceScaleFactor:1});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
 await page.screenshot({path:path.join(qa,'mobile.png')});
 const reportUrls=catalog.filter(r=>r.report).map(r=>new URL('results/'+r.report,base).href);let checked=0,localAssets=0;
 for(const url of reportUrls){const r=await fetch(url);assert.equal(r.status,200,url);checked++}
 const newReports=['2026-10-19-confidence-gate-activation','2026-10-19-uncertainty-gate-activation','2026-10-19-answerable-abstention','2026-10-19-answerable-all-token','2026-10-19-answerable-input-only'];
 for(const name of newReports){
  await page.setViewport({width:1440,height:1100,deviceScaleFactor:1});
  const r=await page.goto(new URL('results/'+name+'/index.html',base).href,{waitUntil:'networkidle0'});assert.equal(r.status(),200,name);
  const links=await page.$$eval('a[href],img[src],link[href],script[src]',es=>es.map(e=>e.href||e.src).filter(x=>x&&!x.includes('#')));
  for(const url of new Set(links.filter(x=>x.startsWith(base)))){const r=await fetch(url);assert.equal(r.status,200,url);localAssets++}
  await page.setViewport({width:390,height:844,deviceScaleFactor:1});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)),name);
 }
 await page.goto(new URL('index.html',base).href,{waitUntil:'networkidle0'});assert(await page.$('a[href="results/abstention/index.html"]'));
 assert.equal(errors.length,0);
 const receipt={passed:true,base,index_entries:rows.length,descriptions_verified:rows.length,report_links_ok:checked,new_report_assets_checked:localAssets,new_reports_rendered:newReports.length,search_and_zero_results:true,desktop_overflow:false,mobile_overflow:false,browser_errors:errors};
 fs.writeFileSync(path.join(qa,'verification.json'),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
}finally{if(browser)await browser.close()}})().catch(e=>{console.error(e);process.exitCode=1});
