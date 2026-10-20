const fs = require('fs');
const path = require('path');
const assert = require('assert');
const puppeteer = require('/home/rka/codex-browser-automation/node_modules/puppeteer-core');

(async () => {
  const root = path.resolve(__dirname, '../../results/2026-10-20-privacy-input-output-negative-sweep');
  const reviews = JSON.parse(fs.readFileSync(path.join(root, 'reviews.json')));
  const output = path.resolve(__dirname, '../../artifacts/privacy-input-output-negative-sweep-ops/qa');
  fs.mkdirSync(output, {recursive: true});
  const browser = await puppeteer.launch({executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox']});
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.setViewport({width: 1280, height: 980});
  await page.goto('file://' + path.join(root, 'index.html'), {waitUntil: 'load'});
  const observed = await page.evaluate(() => ({
    answers: [...document.querySelectorAll('table.responses tr[id]')].map(row => ({id: row.id, answer: row.querySelector('td.answer').textContent})),
    matrixRows: document.querySelectorAll('table.matrix tbody tr').length,
    baselineRows: [...document.querySelectorAll('table.responses tbody tr')].filter(row => !row.id).length,
    overflow: document.documentElement.scrollWidth > innerWidth,
    links: [...document.querySelectorAll('a[href]')].map(a => a.getAttribute('href')),
  }));
  assert.equal(observed.answers.length, 19);
  assert.equal(observed.matrixRows, 7);
  assert.equal(observed.baselineRows, 4);
  for (const row of reviews.rows) {
    assert.equal(observed.answers.find(x => x.id === `${row.group}-${Math.abs(row.offset)}`).answer, row.answer);
  }
  for (const href of observed.links) {
    if (!href.startsWith('#')) assert(fs.existsSync(path.resolve(root, href)), href);
  }
  assert.equal(observed.overflow, false);
  await page.screenshot({path: path.join(output, 'desktop.png')});
  await page.setViewport({width: 390, height: 844});
  const mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  assert.equal(mobileOverflow, false);
  await page.screenshot({path: path.join(output, 'mobile.png')});
  assert.deepEqual(errors, []);
  const verification = {passed: true, all19CompleteInterventionAnswersMatched: true, matrixRows: observed.matrixRows, baselineRows: observed.baselineRows, linkedArtifactsChecked: observed.links.filter(x => !x.startsWith('#')).length, desktopOverflow: observed.overflow, mobileOverflow, errors};
  fs.writeFileSync(path.join(root, 'render-verification.json'), JSON.stringify(verification, null, 2) + '\n');
  console.log(JSON.stringify(verification, null, 2));
  await browser.close();
})().catch(error => {console.error(error);process.exit(1);});
