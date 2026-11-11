'use strict';
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ordinal = n => ({5:'5th',11:'11th',12:'12th'})[n];
const labels = {original:'Original · with the cue', revised:'Revised · cue removed'};
const modal = document.getElementById('modal');
let data, list = 'list-1', format = 'numbered', position = 11;
function color(g) {
  if (g === 0) return 'background:#fffefa;color:#4e5d55';
  const t = Math.min(1, g / data.scale_max), a = [255,242,214], b = [196,61,22];
  const rgb = a.map((x,i) => Math.round(x + (b[i]-x)*t));
  return `background:rgb(${rgb});color:${t>.58?'white':'#293a30'}`;
}
function tip(t) {
  return `Token ${t.position} · ${t.phase}/${t.region} · ID ${t.token_id} · gate ${t.gate.toFixed(6)} · raw ${t.raw.toFixed(6)} · followed by ${JSON.stringify(t.followed_by)}${t.final_input?' · final input: predicts first answer token':''}${t.trace_only_terminal?' · terminal trace only; no extra generated token':''}`;
}
function stream(side, c, rows) {
  return rows.map(t => `<span class="token ${t.gate>0?'positive':''} ${t.region==='question'?'question':''} ${t.trace_only_terminal?'terminal':''}" tabindex="0" style="${color(t.gate)}" data-side="${side}" data-case="${c.id}" data-position="${t.position}" title="${esc(tip(t))}">${esc(t.token.replace(/ /g,'\u200b·').replace(/\t/g,'⇥').replace(/\n/g,'↵\n'))}</span>`).join('');
}
function selected(side) { return data[side].find(c => c.list_id===list && c.format===format && c.ordinal===position); }
function mark(c) { return c.assessment.strict_correct ? '✓' : c.assessment.first_word_correct ? '≈' : '×'; }
function badge(c) { return c.assessment.strict_correct ? 'Correct' : c.assessment.first_word_correct ? 'Right word, extra text' : 'Wrong word'; }
function showFull(side, c) {
  document.getElementById('modal-title').textContent = `${labels[side]} · ${c.id}`;
  document.getElementById('modal-body').innerHTML = `<h3>Exact model input</h3><pre>${esc(c.formatted_prompt)}</pre><h3>Native response</h3><pre>${esc(c.response)}</pre><p>Expected: <strong>${esc(c.expected)}</strong> · Native/observer output and all logits: exact.</p>`;
  modal.showModal();
}
function render() {
  document.getElementById('list-select').value = list;
  document.querySelectorAll('button[data-format]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.format===format)));
  document.querySelectorAll('button[data-ordinal]').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.ordinal)===position)));
  document.querySelectorAll('#answers tbody tr').forEach(tr => tr.classList.toggle('selected-row', tr.dataset.list===list && tr.dataset.format===format));
  document.getElementById('cards').innerHTML = ['original','revised'].map(side => {
    const c=selected(side), s=c.stats, hidden=c.trace.filter(t=>t.region==='template_before');
    const input=c.trace.filter(t=>t.phase==='input' && !['template_before','answer_prefix'].includes(t.region));
    return `<article class="case ${side}" data-side="${side}" data-id="${c.id}"><div class="case-head"><h3>${labels[side]}</h3><span class="badge ${c.assessment.strict_correct?'':'fail'}">${badge(c)}</span></div><p class="expected">${ordinal(position)} word · expected <strong>${esc(c.expected)}</strong></p><div class="response">${esc(c.response)}</div><div class="stats">Max gate: question ${s.question.max_gate.toFixed(3)} · prefix ${s.answer_prefix.max_gate.toFixed(3)} · response ${s.response.max_gate.toFixed(3)}<br><span class="highlight">Final input gate: ${s.final_input_gate.toFixed(3)}</span></div><p class="trace-label">User prompt · question underlined</p><div class="stream input-stream">${stream(side,c,input)}</div><p class="trace-label">Chat answer prefix · still input</p><div class="stream prefix-stream">${stream(side,c,c.trace.filter(t=>t.region==='answer_prefix'))}</div><p class="trace-label">Generated response tokens · including end token</p><div class="stream output-stream">${stream(side,c,c.trace.filter(t=>t.phase==='response'))}</div><details><summary>Leading input wrapper · ${hidden.filter(t=>t.gate>0).length} active tokens</summary><div class="stream">${stream(side,c,hidden)}</div></details><button class="full-button" data-full="${side}">Exact full input and response ↗</button></article>`;
  }).join('');
  document.querySelectorAll('[data-full]').forEach(b => b.addEventListener('click', () => showFull(b.dataset.full, selected(b.dataset.full))));
}
function total(side, n, region) { return data.summary.summaries[side].groups.filter(g=>g.ordinal===n).reduce((a,g)=>a+g.active_cases[region],0); }
function correct(side, n) { return data.summary.summaries[side].groups.filter(g=>g.ordinal===n).reduce((a,g)=>a+g.strict_correct,0); }
function init() {
  const s=data.summary, a=s.summaries.original, b=s.summaries.revised;
  document.getElementById('headline').innerHTML = [[`${s.paired_cases}`,'matched prompt pairs'],[`${a.strict_correct} → ${b.strict_correct}`,'word-only correct / 30'],[`${s.native_observer_exact_runs}/62`,'prompt-comparison controls exact'],['0','gate edits in prompt comparison']].map(([v,t])=>`<div class="metric"><strong>${v}</strong><span>${t}</span></div>`).join('');
  document.getElementById('findings').innerHTML = `<p><strong>User question:</strong> all ${a.question_active_events} original active question tokens disappear. Revised questions have ${b.question_active_events.length} active tokens across all 30 cases.</p><p><strong>The eleventh-request signal persists in the answer prefix:</strong> ${total('revised',11,'answer_prefix')}/10 cases activate there, including ${10-correct('revised',11)} wrong answers. Twelfth requests activate there in ${total('revised',12,'answer_prefix')}/10; fifth requests in ${total('revised',5,'answer_prefix')}/10.</p><p><strong>While reading the list:</strong> numbered lists still activate on the newline before printed 11.; bullet lists have no list-region activation. Generated response words remain inactive. This is observation only; an indexing function is not established.</p>`;
  document.querySelector('#summary tbody').innerHTML = [5,11,12].map(n=>`<tr><td>${ordinal(n)}</td>${['question','answer_prefix','response'].map(region=>`<td>${total('original',n,region)}/10 → ${total('revised',n,region)}/10</td>`).join('')}<td>${correct('original',n)}/10 → ${correct('revised',n)}/10</td></tr>`).join('');
  document.getElementById('paired-changes').textContent = `${s.response_changed}/30 response texts changed. Word-only accuracy: ${s.strict_improved} improved, ${s.strict_worsened} worsened, ${30-s.strict_improved-s.strict_worsened} unchanged. First-listed-word accuracy: ${a.first_word_correct}/30 → ${b.first_word_correct}/30. The numbered “12. hammer” answer became “hammer”; a correct “ticket” became “kite”; one already-wrong answer changed from “ring” to “bell”.`;
  document.getElementById('list-select').innerHTML = data.lists.map((l,i)=>`<option value="${l.id}">List ${i+1}</option>`).join('');
  document.getElementById('list-select').addEventListener('change',e=>{list=e.target.value;render();});
  document.querySelectorAll('button[data-format]').forEach(b=>b.addEventListener('click',()=>{format=b.dataset.format;render();}));
  document.querySelectorAll('button[data-ordinal]').forEach(b=>b.addEventListener('click',()=>{position=Number(b.dataset.ordinal);render();}));
  document.getElementById('scale-max').textContent = data.scale_max.toFixed(2)+' · gate';
  document.querySelector('#answers tbody').innerHTML = data.lists.flatMap((l,i)=>['numbered','bullets'].map(fmt=>`<tr data-list="${l.id}" data-format="${fmt}"><td>List ${i+1} · ${fmt}</td>${[5,11,12].map(n=>{const old=data.original.find(c=>c.list_id===l.id&&c.format===fmt&&c.ordinal===n),now=data.revised.find(c=>c.id===old.id);return `<td><button class="answer-button" data-open="${old.id}" title="Expected: ${esc(old.expected)}"><span class="pair-response"><span>${mark(old)} ${esc(old.response)}</span><span class="arrow">↓ cue removed</span><span>${mark(now)} ${esc(now.response)}</span><small>expected ${esc(old.expected)}</small></span></button></td>`;}).join('')}</tr>`)).join('');
  document.querySelectorAll('[data-open]').forEach(b=>b.addEventListener('click',()=>{const c=data.original.find(c=>c.id===b.dataset.open);list=c.list_id;format=c.format;position=c.ordinal;render();document.getElementById('comparison').scrollIntoView({behavior:'auto',block:'start'});}));
  document.querySelector('#activity tbody').innerHTML = s.groups.map(g=>`<tr><td>${g.format} · ${ordinal(g.ordinal)}</td>${['list','question','answer_prefix','response'].map(region=>`<td>${g.original.active_cases[region]}/5 → ${g.revised.active_cases[region]}/5</td>`).join('')}</tr>`).join('');
  document.getElementById('verification').textContent = 'All 60 pilot captures and both month calibrations exactly match their unhooked native repeats: generated IDs, selected probabilities, every decoding-logit hash, and terminal-unused-logit hash. All cases are complete; no truncation occurred.';
  document.getElementById('prefix-check').textContent = `Unequal prefill lengths can slightly change BF16 raw scores on the unchanged list prefix. A separate right-padded check gave all six original/revised queries per word order and format the same prefill length. Shared-prefix raw-score differences were exactly ${s.paired_prefix_numerical_check.maximum_raw_difference} in all ten groups. It generated no answers and did not replace these original traces. Earlier list activations cannot be caused by a later question edit.`;
  render();
}
function inspect(e) {
  const el=e.target.closest('.token'); if(!el)return;
  const c=data[el.dataset.side].find(c=>c.id===el.dataset.case), t=c.trace.find(t=>t.position===Number(el.dataset.position));
  document.getElementById('token-readout').textContent = `${labels[el.dataset.side]} · ${JSON.stringify(t.token)} · ${tip(t)}`;
}
['pointerover','focusin','click'].forEach(event=>document.addEventListener(event,inspect));
document.getElementById('close-modal').addEventListener('click',()=>modal.close());
modal.addEventListener('click',e=>{if(e.target===modal)modal.close();});
fetch('comparison-data.json?v=09f4475669be').then(r=>{if(!r.ok)throw Error('Data unavailable');return r.json();}).then(x=>{data=x;init();}).catch(e=>{document.getElementById('headline').textContent='Could not load the results. Please reload.';console.error(e);});
