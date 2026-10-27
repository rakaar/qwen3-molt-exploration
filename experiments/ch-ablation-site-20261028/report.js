(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('report-data').textContent);
  const el = id => document.getElementById(id);
  const escape = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const percent = value => (value * 100).toFixed(6);
  function render() {
    const gate = el('gate').value, letterCase = el('letter-case').value;
    const word = el('word').value, metric = el('metric').value;
    const activeOnly = el('active-only').checked;
    const rows = data.rows.filter(row => (gate === 'all' || row.feature === gate)
      && (letterCase === 'all' || row.letter_case === letterCase)
      && (word === 'all' || row.word === word) && (!activeOnly || row.active));
    el('row-count').textContent = `${rows.length} of ${data.rows.length} comparisons`;
    el('prefix-heading').textContent = metric === 'first_output' ? 'Probability context' : 'Identical native prefix';
    el('rows').innerHTML = rows.length ? rows.map(row => {
      const native = row[`native_p_${metric}`], off = row[`gate_off_p_${metric}`];
      const delta = (off - native) * 100;
      const color = delta < 0 ? 'negative' : delta > 0 ? 'positive' : 'zero';
      const context = metric === 'first_output' ? 'prompt alone' : (row.answer_slot.prefix || 'prompt alone');
      const gates = row.natural_gates.map((value, index) => `${row.target_tokens[index]}: ${value.toFixed(4)}`).join(' · ');
      return `<tr data-feature="${escape(row.feature)}" data-case="${escape(row.case_id)}"><td><strong>${escape(row.feature)}</strong></td>`
        + `<td><code>${escape(row.word)}</code><span class="prompt-label">${row.letter_case === 'upper' ? 'Uppercase' : 'Lowercase'}</span></td>`
        + `<td><span class="badge ${row.active ? 'active' : 'off'}">${row.active ? 'active edit' : 'inactive · no-op'}</span><span class="gate-values">${escape(gates)}</span></td>`
        + `<td><code>${escape(row.native_answer)}</code></td><td><code>${escape(row.gate_off_answer)}</code></td>`
        + `<td><code>${escape(context)}</code></td><td>${percent(native)}</td><td>${percent(off)}</td>`
        + `<td class="${color}">${delta > 0 ? '+' : ''}${delta.toFixed(6)}</td></tr>`;
    }).join('') : '<tr><td class="empty" colspan="9">No comparisons match these filters. Try another word or include inactive gates.</td></tr>';
  }
  el('filters').addEventListener('change', render);
  render();
})();
