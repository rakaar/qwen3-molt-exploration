const pptr=require('/home/rka/codex-browser-automation/node_modules/puppeteer-core');
const fs=require('fs'),path=require('path'),assert=require('assert');
const root=path.resolve(__dirname,'../../..'),R=path.join(root,'results/2026-10-19-answerable-abstention'),QA=path.join(root,'artifacts/answerable-abstention-qa');fs.mkdirSync(QA,{recursive:true});
const raw=JSON.parse(fs.readFileSync(path.join(R,'artifacts/all-results.json'))),summary=JSON.parse(fs.readFileSync(path.join(R,'summary.json'))),reviews=JSON.parse(fs.readFileSync(path.join(R,'reviewed-judgments.json'))).rows;
(async()=>{let browser;try{
 browser=await pptr.launch({executablePath:'/usr/bin/google-chrome',headless:true,userDataDir:'/home/rka/.cache/codex-answerable-abstention-browser',args:['--no-first-run']});
 const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));await page.setCacheEnabled(false);await page.setViewport({width:1440,height:1000,deviceScaleFactor:1});
 const response=await page.goto('http://127.0.0.1:8787/2026-10-19-answerable-abstention/index.html',{waitUntil:'networkidle0'});assert.equal(response.status(),200);
 const state=await page.evaluate(()=>({title:document.title,caseCount:document.querySelectorAll('.example').length,trials:[...document.querySelectorAll('.trial')].map(x=>({id:x.id,answer:x.querySelector('blockquote').textContent})),cards:[...document.querySelectorAll('.example')].map(x=>({id:x.id.replace('case-',''),answers:[...x.querySelectorAll('blockquote')].map(e=>e.textContent)})),successCards:document.querySelectorAll('.answer.success').length,overflow:document.documentElement.scrollWidth>innerWidth,images:[...document.images].map(i=>({src:i.getAttribute('src'),loaded:i.complete&&i.naturalWidth>0})),links:[...document.querySelectorAll('a')].map(x=>x.getAttribute('href'))}));
 assert.equal(state.caseCount,10);assert.equal(state.successCards,3);assert.equal(state.trials.length,245);assert(!state.overflow);assert(state.images.every(x=>x.loaded));
 for(const x of state.trials)assert.equal(x.answer,raw.rows.find(r=>r.trial_id===x.id).answer);
 for(const x of state.cards){const c=summary.per_case.find(r=>r.id===x.id);assert.equal(x.answers[0],c.baseline);assert.equal(x.answers[1],c.selected_answer)}
 await page.screenshot({path:path.join(QA,'desktop-top.png')});await page.$eval('#case-P2-A1',e=>e.scrollIntoView());await page.screenshot({path:path.join(QA,'desktop-success.png')});
 await page.click('[data-filter="abstention"]');const visible=await page.$$eval('.trial',xs=>xs.filter(x=>!x.hidden).map(x=>x.id));assert.deepEqual(visible,reviews.filter(x=>x.category==='abstention').map(x=>x.trial_id));state.abstentionTrialsVisible=visible.length;
 await page.click('[data-filter="all"]');await page.$eval('.trial',x=>x.open=true);assert(await page.$eval('.trial',x=>x.open));
 await page.setViewport({width:390,height:844,deviceScaleFactor:1});await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(QA,'mobile-top.png')});
 state.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);assert(!state.mobileOverflow);await page.$eval('#case-P2-A3',e=>e.scrollIntoView());await page.screenshot({path:path.join(QA,'mobile-success.png')});
 state.missingLinks=state.links.filter(x=>!/^https?:|#/.test(x)).filter(x=>!fs.existsSync(path.resolve(R,x)));assert.equal(state.missingLinks.length,0);assert.equal(errors.length,0);state.errors=errors;state.passed=true;
 fs.writeFileSync(path.join(QA,'render-verification.json'),JSON.stringify(state,null,2)+'\n');console.log(JSON.stringify({cases:state.caseCount,trials:state.trials.length,successfulCases:state.successCards,abstentionTrials:state.abstentionTrialsVisible,desktopOverflow:state.overflow,mobileOverflow:state.mobileOverflow,missingLinks:state.missingLinks,errors,passed:true}));
 }finally{if(browser)await browser.close()}})().catch(e=>{console.error(e);process.exitCode=1});
