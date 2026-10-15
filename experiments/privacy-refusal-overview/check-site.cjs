// Verify rendered data and representative interactions locally or on GitHub Pages.
const pptr = require('/home/rka/codex-browser-automation/node_modules/puppeteer-core');
const fs = require('fs'), path = require('path'), assert = require('assert');
const root=path.resolve(__dirname,'../..'), out=path.join(root,'results/2026-10-15-privacy-refusal-overview');
const qa=path.join(root,'artifacts/privacy-refusal-overview-qa');fs.mkdirSync(qa,{recursive:true});
const url=process.argv[2]||'http://127.0.0.1:8790/results/2026-10-15-privacy-refusal-overview/index.html';
const live=url.startsWith('https:');const evidence=JSON.parse(fs.readFileSync(path.join(out,'evidence.json')));
(async()=>{let browser;try{
 browser=await pptr.launch({executablePath:'/usr/bin/google-chrome',headless:true,userDataDir:'/home/rka/.cache/codex-privacy-refusal-overview-browser',args:['--no-first-run']});
 const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.setCacheEnabled(false);
 await page.setViewport({width:1440,height:1050,deviceScaleFactor:1});
 const response=await page.goto(url,{waitUntil:'networkidle0'});assert.equal(response.status(),200);
 await page.addStyleTag({content:'html{scroll-behavior:auto!important}'});
 const state=await page.evaluate(()=>({title:document.title,overflow:document.documentElement.scrollWidth>innerWidth,
  natural:[...document.querySelectorAll('.natural-card')].map(x=>({id:x.dataset.id,answer:x.querySelector('.response').textContent,
    tokens:[...x.querySelectorAll('.token')].map(t=>({position:Number(t.dataset.position),gate:Number(t.dataset.gate)}))})),
  causal:[...document.querySelectorAll('.steered')].map(x=>({id:x.dataset.trial,answer:x.querySelector('.response').textContent})),
  pairColumns:[...document.querySelectorAll('#natural .pair')].map(x=>getComputedStyle(x).gridTemplateColumns.split(' ').length),
  images:[...document.images].map(i=>({src:i.getAttribute('src'),loaded:i.complete&&i.naturalWidth>0,width:i.naturalWidth,height:i.naturalHeight})),
  links:[...document.querySelectorAll('a')].map(x=>x.getAttribute('href'))}));
 assert(!state.overflow);assert.equal(state.natural.length,6);assert.deepEqual(state.pairColumns,[2,2,2]);assert.equal(state.causal.length,3);assert(state.images.every(x=>x.loaded));
 for(const row of state.natural){const expected=evidence.natural.find(x=>x.id===row.id);assert.equal(row.answer,expected.answer);assert.equal(row.tokens.length,expected.tokens.length);row.tokens.forEach((t,i)=>{assert.equal(t.position,expected.tokens[i].position);assert(Math.abs(t.gate-expected.tokens[i].gate)<1e-9)})}
 for(const row of state.causal)assert.equal(row.answer,evidence.causal.find(x=>x.trial_id===row.id).steered_answer);
 const prefix=live?'public-':'local-';
 await page.screenshot({path:path.join(qa,prefix+'desktop-top.png')});
 await page.$eval('#natural',x=>x.scrollIntoView());await page.screenshot({path:path.join(qa,prefix+'desktop-natural.png')});
 await page.click('#show-values');assert(await page.evaluate(()=>document.body.classList.contains('hide-values')));await page.click('#show-values');
 await page.$eval('#causal',x=>x.scrollIntoView());await page.screenshot({path:path.join(qa,prefix+'desktop-causal.png')});
 await page.$eval('#prompts details',x=>x.open=true);assert(await page.$eval('#prompts details',x=>x.open));
 await page.setViewport({width:390,height:844,deviceScaleFactor:1});await page.evaluate(()=>scrollTo(0,0));
 await page.screenshot({path:path.join(qa,prefix+'mobile-top.png')});
 state.mobile=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth,pairColumns:getComputedStyle(document.querySelector('#natural .pair')).gridTemplateColumns.split(' ').length}));assert(!state.mobile.overflow);assert.equal(state.mobile.pairColumns,1);
 await page.$eval('#natural .unauthorized',x=>x.scrollIntoView());await page.screenshot({path:path.join(qa,prefix+'mobile-natural.png')});
 await page.$eval('#causal .steered',x=>x.scrollIntoView());await page.screenshot({path:path.join(qa,prefix+'mobile-causal.png')});
 state.errors=errors;assert.equal(errors.length,0);state.url=url;state.passed=true;
 fs.writeFileSync(path.join(qa,prefix+'render-verification.json'),JSON.stringify(state,null,2)+'\n');
 console.log(JSON.stringify({url,natural:state.natural.length,causal:state.causal.length,desktopOverflow:state.overflow,mobile:state.mobile,images:state.images,errors,passed:true}));
 }finally{if(browser)await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1});
