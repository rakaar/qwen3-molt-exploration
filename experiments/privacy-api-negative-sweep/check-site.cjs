const fs = require('fs'), path = require('path'), assert = require('assert');
const puppeteer = require('/home/rka/codex-browser-automation/node_modules/puppeteer-core');
const root = path.resolve(__dirname, '../..');
const result = path.join(root, 'results/2026-10-20-privacy-api-negative-sweep');
const qa = path.join(root, 'artifacts/privacy-api-negative-sweep-ops/qa');

(async () => {
  fs.mkdirSync(qa, {recursive: true});
  const rows = fs.readdirSync(result).filter(n => n.startsWith('privacy-api-negative') && fs.statSync(path.join(result,n)).isDirectory()).flatMap(stage => JSON.parse(fs.readFileSync(path.join(result,stage,'artifacts/results.json'))).rows).filter(r => r.trial.mode === 'additive');
  assert.equal(rows.length, 21);
  const browser = await puppeteer.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
  const page = await browser.newPage(), errors = [];
  page.on('pageerror',error=>errors.push(String(error)));
  await page.setViewport({width:1280,height:980});
  await page.goto('file://'+path.join(result,'index.html'), {waitUntil:'load'});
  const state = await page.evaluate(()=>({
    answers:[...document.querySelectorAll('table.responses tr[id]')].map(r=>({id:r.id,text:r.querySelector('.answer').textContent})),
    rungs:document.querySelectorAll('table.matrix tbody tr').length,
    desktopOverflow:document.documentElement.scrollWidth>innerWidth,
    links:[...document.querySelectorAll('a[href]')].map(a=>a.getAttribute('href')),
    text:document.body.textContent,
  }));
  assert.equal(state.answers.length,21);assert.equal(state.rungs,7);assert(!state.desktopOverflow);
  for(const row of rows){
    const id=row.trial.id.split('__')[0]+'-'+Math.abs(row.trial.offset);
    assert.equal(state.answers.find(r=>r.id===id).text,row.answer);
  }
  assert(state.text.includes('Northstar Weather'));assert(!state.text.includes('MAPLE-482'));
  for(const href of state.links) if(!href.startsWith('#')) assert(fs.existsSync(path.resolve(result,href)),href);
  await page.screenshot({path:path.join(qa,'desktop.png')});
  await page.setViewport({width:390,height:844});
  const mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);assert(!mobileOverflow);
  await page.screenshot({path:path.join(qa,'mobile.png')});
  await page.goto('file://'+path.join(root,'results/2026-10-20-privacy-input-output-negative-sweep/index.html'));
  await page.waitForFunction(()=>location.pathname.includes('/2026-10-20-privacy-api-negative-sweep/'));
  const redirectVerified=await page.evaluate(()=>document.title.includes('Matched API-key'));assert(redirectVerified);
  await page.goto('file://'+path.join(root,'results/2026-10-20-privacy-input-output-negative-sweep/bob-alice-original.html'));
  assert((await page.evaluate(()=>document.body.textContent)).includes('MAPLE-482'));
  await page.goto('file://'+path.join(root,'results/2026-10-15-privacy-refusal-overview/index.html'));
  assert.equal(await page.$$eval('.natural-card',x=>x.length),6);
  assert.equal(await page.$$eval('.steered',x=>x.length),3);
  assert((await page.$eval('#matched-reverse',x=>x.textContent)).includes('L13–T921 at −32'));
  assert.deepEqual(errors,[]);
  const verification={passed:true,completeAnswersVerified:21,strengthRows:7,desktopOverflow:state.desktopOverflow,mobileOverflow,linkedArtifactsChecked:state.links.filter(h=>!h.startsWith('#')).length,oldLinkRedirectVerified:true,bobAliceArchivePresent:true,originalPositiveCardsPreserved:3,originalNaturalCardsPreserved:6,errors};
  fs.writeFileSync(path.join(result,'render-verification.json'),JSON.stringify(verification,null,2)+'\n');
  console.log(JSON.stringify(verification,null,2));
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
