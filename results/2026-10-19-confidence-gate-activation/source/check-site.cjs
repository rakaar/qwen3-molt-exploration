const pptr=require('/home/rka/codex-browser-automation/node_modules/puppeteer-core');
const fs=require('fs'),path=require('path'),assert=require('assert');
const W=path.resolve(__dirname,'../..'),R=path.join(W,'results/2026-10-19-confidence-gate-activation'),Q=path.join(W,'artifacts/confidence-gate-activation/qa');fs.mkdirSync(Q,{recursive:true});
const d=JSON.parse(fs.readFileSync(path.join(R,'artifacts/measurements.json'))),keys=d.protocol.features.map(f=>`L${f.layer}-T${f.transform}`);
(async()=>{let browser;try{
 browser=await pptr.launch({executablePath:'/usr/bin/google-chrome',headless:true,userDataDir:'/home/rka/.cache/codex-confidence-activation-browser',args:['--no-first-run']});const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));await page.setViewport({width:1440,height:1000,deviceScaleFactor:1});
 const res=await page.goto('http://127.0.0.1:8787/2026-10-19-confidence-gate-activation/index.html',{waitUntil:'networkidle0'});assert.equal(res.status(),200);
 const actual=await page.$$eval('.case',cs=>cs.map(c=>({id:c.dataset.id,group:c.dataset.group,panels:[...c.querySelectorAll('.answer-panel')].map(p=>({feature:p.dataset.feature,tokens:[...p.querySelectorAll('.token')].map(t=>({token:t.textContent,position:Number(t.dataset.pos),gate:Number(t.dataset.gate),raw:Number(t.dataset.raw),bg:getComputedStyle(t).backgroundColor}))}))})));
 assert.equal(actual.length,13);let tokens=0,active=0;
 for(const c of actual){const r=d.rows.find(r=>r.id===c.id);assert(r);assert.equal(c.group,r.group);assert.equal(c.panels.length,3);for(const p of c.panels){assert(keys.includes(p.feature));const ts=r.answer_tokens.filter(t=>!t.special);assert.equal(p.tokens.length,ts.length);assert.equal(p.tokens.map(t=>t.token).join(''),r.answer);for(let i=0;i<ts.length;i++){let a=p.tokens[i],b=ts[i],x=b.features[p.feature];assert.equal(a.token,b.token);assert.equal(a.position,b.position);assert(Math.abs(a.gate-x.gate)<1e-9);assert(Math.abs(a.raw-x.raw)<1e-9);if(a.gate===0)assert.equal(a.bg,'rgb(255, 255, 255)');else active++;tokens++}}}
 assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
 await page.screenshot({path:path.join(Q,'desktop-top.png')});
 for(const g of ['correct','made_up','abstention','positive_control']){const el=await page.$('#'+g);await el.screenshot({path:path.join(R,g+'-token-heatmaps.png')});}
 await page.$eval('#positive_control',x=>x.scrollIntoView());await page.screenshot({path:path.join(Q,'desktop-control.png')});
 await page.setViewport({width:390,height:844,deviceScaleFactor:1});await page.evaluate(()=>scrollTo(0,0));assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));await page.screenshot({path:path.join(Q,'mobile-top.png')});
 const links=await page.$$eval('a',as=>as.map(a=>a.getAttribute('href'))),missing=links.filter(x=>!/^https?:|#/.test(x)).filter(x=>!fs.existsSync(path.resolve(R,x)));assert.equal(missing.length,0);assert.equal(errors.length,0);
 const receipt={passed:true,case_rows:13,qa_cases:12,separate_control:1,feature_columns:3,rendered_token_values_verified:tokens,rendered_active_tokens:active,desktop_overflow:false,mobile_overflow:false,missing_links:missing,browser_errors:errors};fs.writeFileSync(path.join(Q,'render-verification.json'),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
}finally{if(browser)await browser.close()}})().catch(e=>{console.error(e);process.exitCode=1});
