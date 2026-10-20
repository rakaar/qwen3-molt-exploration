const fs=require('fs'),path=require('path'),assert=require('assert');
const pptr=require('/home/rka/codex-browser-automation/node_modules/puppeteer-core');
const root=path.resolve(__dirname,'../..'),base=process.env.PRIVACY_REPORT_BASE||'http://127.0.0.1:8788/';
const out=process.env.PRIVACY_QA_DIR||path.join(root,'artifacts/privacy-zero-publication-qa');fs.mkdirSync(out,{recursive:true});
(async()=>{let browser;try{
 browser=await pptr.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-first-run']});
 const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));let assets=0;
 for(const name of ['2026-10-15-privacy-zero-clamp','2026-10-15-privacy-query-clamp','2026-10-15-privacy-refusal-overview']){
  await page.setViewport({width:1440,height:1000,deviceScaleFactor:1});
  const res=await page.goto(new URL('results/'+name+'/index.html',base).href,{waitUntil:'networkidle0'});assert.equal(res.status(),200);
  if(name!=='2026-10-15-privacy-refusal-overview'){
   const data=JSON.parse(fs.readFileSync(path.join(root,'results',name,'artifacts/results.json')));
   const text=await page.$eval('body',x=>x.textContent);for(const r of data.rows)assert(text.includes(r.answer));
   assert(await page.$('nav[aria-label="Related reports"]'));
  }else{assert(await page.$('#zero-clamp'));assert((await page.$eval('#zero-clamp',x=>x.textContent)).includes('Same complete refusal'))}
  const hrefs=await page.$$eval('a[href],img[src],link[href]',es=>es.map(e=>e.href||e.src).filter(x=>x&&!x.includes('#')));
  for(const url of new Set(hrefs.filter(x=>x.startsWith(base)))){const r=await fetch(url);assert.equal(r.status,200,url);if(r.body)await r.body.cancel();assets++}
  assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)),name+' desktop overflow');
  await page.screenshot({path:path.join(out,name+'-desktop.png')});
  await page.setViewport({width:390,height:844,deviceScaleFactor:1});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)),name+' mobile overflow');
  await page.screenshot({path:path.join(out,name+'-mobile.png')});
 }
 assert.equal(errors.length,0);const proof={passed:true,base,reports:3,old_full_answers_checked:true,overview_reverse_direction_section:true,local_assets_checked:assets,desktop_and_mobile_overflow:false,browser_errors:errors};
 fs.writeFileSync(path.join(out,'verification.json'),JSON.stringify(proof,null,2)+'\n');console.log(JSON.stringify(proof));
}finally{if(browser)await browser.close()}})().catch(e=>{console.error(e);process.exitCode=1});
