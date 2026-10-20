const fs=require('fs'),path=require('path'),assert=require('assert');
const pptr=require('/home/rka/codex-browser-automation/node_modules/puppeteer-core');
const root=path.resolve(__dirname,'../..'),R=path.join(root,'results/2026-10-20-privacy-negative-steering');
const base=process.env.PRIVACY_NEGATIVE_BASE||'http://127.0.0.1:8788/';
const Q=process.env.PRIVACY_NEGATIVE_QA||path.join(root,'artifacts/privacy-negative-qa');fs.mkdirSync(Q,{recursive:true});
const d=JSON.parse(fs.readFileSync(path.join(R,'artifacts/results.json'))),j=JSON.parse(fs.readFileSync(path.join(R,'reviews.json')));
(async()=>{let browser;try{
 browser=await pptr.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-first-run']});
 const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.setViewport({width:1500,height:1100,deviceScaleFactor:1});
 const res=await page.goto(new URL('results/2026-10-20-privacy-negative-steering/index.html',base).href,{waitUntil:'networkidle0'});assert.equal(res.status(),200);
 const cards=await page.$$eval('article',xs=>xs.map(a=>({id:a.id,answer:a.querySelector('blockquote').textContent,category:a.querySelector('.category').textContent})));
 assert.equal(cards.length,29);for(const c of cards){const r=d.rows.find(r=>r.trial.id===c.id);assert(r);assert.equal(c.answer,r.answer);assert.equal(c.category,j[c.id].category)}
 const cells=await page.$$eval('tbody td a',xs=>xs.map(a=>({id:a.getAttribute('href').slice(1),category:a.textContent})));assert.equal(cells.length,24);
 for(const c of cells){assert.equal(j[c.id].category,c.category);assert(d.rows.find(r=>r.trial.id===c.id).trial.mode==='additive')}
 assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
 await page.screenshot({path:path.join(Q,'desktop.png')});
 const success=d.rows.find(r=>j[r.trial.id].clean_coherent_disclosure);if(success){await page.$eval('#'+success.trial.id,x=>x.scrollIntoView());await page.screenshot({path:path.join(Q,'disclosure-example.png')})}
 await page.setViewport({width:390,height:844,deviceScaleFactor:1});await page.evaluate(()=>scrollTo(0,0));assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
 await page.screenshot({path:path.join(Q,'mobile.png')});
 let assets=0;const hrefs=await page.$$eval('a[href]',xs=>xs.map(a=>a.href));for(const url of new Set(hrefs.filter(x=>x.startsWith(base)&&!x.includes('#')))){const r=await fetch(url);assert.equal(r.status,200,url);if(r.body)await r.body.cancel();assets++}
 assert.equal(errors.length,0);const proof={passed:true,base,complete_answers_verified:29,review_categories_verified:29,matrix_cells_verified:24,local_links_checked:assets,desktop_and_mobile_overflow:false,browser_errors:errors};
 fs.writeFileSync(path.join(Q,'verification.json'),JSON.stringify(proof,null,2)+'\n');console.log(JSON.stringify(proof));
}finally{if(browser)await browser.close()}})().catch(e=>{console.error(e);process.exitCode=1});
