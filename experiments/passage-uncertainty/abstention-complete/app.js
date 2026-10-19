const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={correct:'Answer retained',answer:'Answer given',mixed:'Answer / uncertainty / false rationale',abstention:'Clear coherent abstention',broken:'Incoherent / repetitive',question_drift:'Question drift',wrong:'Wrong / off-topic',other:'Other'};
function revealHash(){let id=decodeURIComponent(location.hash.slice(1));let el=document.getElementById(id);if(el){let parent=el.closest('details.case');if(parent)parent.open=true;(parent||el).scrollIntoView({block:"start"});}}
addEventListener('hashchange',revealHash);revealHash();
(async()=>{const data=await fetch('viewer-data.json').then(r=>r.json());window.studyData=data;const q=document.getElementById('question'),scope=document.getElementById('scope'),view=document.getElementById('sweep'),status=document.getElementById('viewer-status');
 q.innerHTML=data.cases.map(c=>`<option value="${esc(c.id)}">${c.cohort==='correct'?'Answer present':'Made-up baseline'} · ${esc(c.id)} · ${esc(c.question)}</option>`).join('');
 function render(){const c=data.cases.find(c=>c.id===q.value),rs=data.rows.filter(r=>r.case_id===c.id&&r.scope===scope.value).sort((a,b)=>a.strength-b.strength);let count=rs.length;
 const base=`<tr><th>0<br><small>Native</small></th><td><span class="badge">${c.cohort==='correct'?'Correct answer':'Model made up an answer'}</span></td><td><blockquote>${esc(c.prior_native_answer)}</blockquote></td></tr>`;
 view.innerHTML=`<table><thead><tr><th>Addition</th><th>Review</th><th>Complete response</th></tr></thead><tbody>${base}${rs.map(r=>`<tr data-id="${esc(r.id)}"><th>+${r.strength}</th><td><span class="badge ${esc(r.review.category)}">${esc(labels[r.review.category]||r.review.category)}</span></td><td><blockquote>${esc(r.answer)}</blockquote><p class="fine">${esc(r.review.notes)}</p></td></tr>`).join('')}</tbody></table>`;
 status.textContent=`${count} tested positive strengths plus native baseline. Every response is shown in full.`;view.dataset.case=c.id;view.dataset.scope=scope.value;
 }
 q.addEventListener('change',render);scope.addEventListener('change',render);render();window.ready=true;
})().catch(e=>{document.getElementById('viewer-status').textContent='Could not load response data: '+e.message;throw e});
