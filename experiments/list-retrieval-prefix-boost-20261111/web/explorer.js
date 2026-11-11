'use strict';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=p=>p*100>=.001?(p*100).toFixed(4):(p*100).toExponential(2);
const formatName=f=>f==='numbered'?'numbered':'bullets';
let data,gate=40,request='all',format='all';
const modal=document.getElementById('boost-modal');
function classification(r){
 if(r.native_already_eleventh&&r.strict_eleventh)return 'Already item 11 natively';
 if(r.new_strict_eleventh)return 'New item-11 word-only answer';
 if(r.new_first_listed_eleventh)return 'New item-11 answer with extra text';
 if(['eleven','11','eleventh','11th'].includes(r.response.trim().toLowerCase()))return 'Literal number word; not list retrieval';
 if(r.response===r.native_response)return 'Unchanged response';
 if(r.retains_requested)return 'Still the requested word';
 return 'Other word / format change';
}
function inspect(id){
 const c=data.cases.find(c=>c.id===id),r=data.rows.find(r=>r.id===id&&r.target_gate===gate);
 document.getElementById('boost-modal-title').textContent=`${id} · gate ${gate}`;
 document.getElementById('boost-modal-body').innerHTML=`<h3>Exact user prompt</h3><pre>${esc(c.prompt)}</pre><h3>Native response</h3><pre>${esc(r.native_response)}</pre><h3>Steered response · complete</h3><pre>${esc(r.response)}</pre><p>${esc(classification(r))} · Stop: ${esc(r.stop_reason)} · Exact generated-sequence probability including selected EOS: ${pct(r.generated_sequence_probability)}%.</p><p>Requested word: <strong>${esc(c.expected)}</strong>; item 11: <strong>${esc(c.eleventh_word)}</strong>. Saved native eleventh-word control answered <strong>${esc(c.native_eleventh_question_control.response)}</strong> (${c.native_eleventh_question_control.strict_correct?'correct':'wrong'}).</p><h3>Edited input tokens</h3><table><thead><tr><th>Position</th><th>Location</th><th>Token</th><th>Scalar</th></tr></thead><tbody>${c.prefix_positions.map(t=>`<tr><td>${t.position}</td><td>${t.location}</td><td><code>${esc(t.token.replace(/\n/g,'↵'))}</code></td><td>${t.natural_gate.toFixed(6)} → ${gate}</td></tr>`).join('')}</tbody></table><p>All other input and generated-token outputs were unedited. The recipient operator was kept local to this prompt.</p><h3>Direct word probabilities</h3><table><thead><tr><th>Word position</th><th>Native (%)</th><th>Steered (%)</th></tr></thead><tbody>${[5,11,12].map(n=>`<tr><td>${n}</td><td>${pct(r['native_p_word_'+n])}</td><td>${pct(r['steered_p_word_'+n])}</td></tr>`).join('')}</tbody></table><p>These score the exact lowercase word at response start and exclude leading sentences and EOS.</p>`;
 modal.showModal();
}
function render(){
 document.getElementById('boost-gate').value=String(gate);
 const rows=data.rows.filter(r=>r.target_gate===gate&&(request==='all'||r.ordinal===Number(request))&&(format==='all'||r.format===format));
 document.getElementById('response-count').textContent=`${rows.length} cases shown at fixed scalar ${gate}. Click “Full prompt + response” for complete text and exact edited tokens.`;
 document.querySelector('#boost-responses tbody').innerHTML=rows.map(r=>{
  const c=data.cases.find(c=>c.id===r.id),label=classification(r),hit=r.new_first_listed_eleventh,literal=label.startsWith('Literal');
  return `<tr data-id="${r.id}"><td>List ${r.list_id.split('-')[1]} · ${formatName(r.format)} · ask ${r.ordinal}<div class="boost-flags">Asked control: ${r.native_requested_correct?'correct':'wrong'}<br>11th control: ${r.native_eleventh_control_correct?'correct':'wrong'}</div></td><td>${esc(r.expected)} / ${esc(r.eleventh_word)}</td><td class="boost-text native-response">${esc(r.native_response)}</td><td><div class="boost-text steered-response">${esc(r.response.length>100?r.response.slice(0,100)+'…':r.response)}</div><div class="boost-result ${hit?'hit':literal?'literal':''}">${esc(label)}</div></td><td class="boost-prob">${pct(r.native_p_word_11)} → ${pct(r.steered_p_word_11)}</td><td class="boost-prob">${pct(r['native_p_word_'+r.ordinal])} → ${pct(r['steered_p_word_'+r.ordinal])}</td><td><button class="boost-inspect" data-inspect="${r.id}">Full prompt + response</button></td></tr>`;
 }).join('');
 document.querySelectorAll('[data-inspect]').forEach(b=>b.addEventListener('click',()=>inspect(b.dataset.inspect)));
}
document.getElementById('boost-gate').addEventListener('change',e=>{gate=Number(e.target.value);render();});
document.getElementById('boost-request').addEventListener('change',e=>{request=e.target.value;render();});
document.getElementById('boost-format').addEventListener('change',e=>{format=e.target.value;render();});
document.getElementById('boost-close').addEventListener('click',()=>modal.close());
modal.addEventListener('click',e=>{if(e.target===modal)modal.close();});
fetch('explorer-data.json?v=__VERSION__').then(r=>{if(!r.ok)throw Error('Results unavailable');return r.json();}).then(x=>{data=x;document.getElementById('boost-gate').innerHTML=data.strengths.map(g=>`<option value="${g}">${g}</option>`).join('');render();}).catch(e=>{document.getElementById('response-count').textContent='Could not load the results. Please reload.';console.error(e);});
