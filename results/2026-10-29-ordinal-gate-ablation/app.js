'use strict';
const $ = (selector) => document.querySelector(selector);
const make = (tag, className, text) => {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
};
const visible = (text) => text ? text.replace(/ /g, '·').replace(/\n/g, '↵\n').replace(/\t/g, '⇥') : '◌';
const formatGate = (value) => value === null ? 'not measured' : Number(value).toFixed(6);
function colorGate(value) {
  const weight = Math.min(1, Math.max(0, value / 3));
  const light = [244, 249, 247], dark = [38, 147, 118];
  return `rgb(${light.map((v, i) => Math.round(v + weight * (dark[i] - v))).join(',')})`;
}
function inspect(token, side, phase, feature, button) {
  const status = token.edited ? `Applied gate: 0 (natural value before removal: ${formatGate(token.natural_gate)}).`
    : `Natural gate: ${formatGate(token.natural_gate)}. ${side === 'Gate off' ? 'This position was untouched.' : 'No intervention.'}`;
  const text = `${feature} · ${side} · ${phase} token ${token.position} · ID ${token.token_id} · ${JSON.stringify(token.text)} · ${status}`;
  $('#inspector').replaceChildren(make('span', 'inspector-dot'), make('span', '', text));
  const local = button.closest('.example').querySelector('.token-reading');
  local.hidden = false;
  local.textContent = text;
}
function tokens(list, side, phase, feature, answer = false) {
  const wrap = make('div', `tokens${answer ? ' answer' : ''}`);
  wrap.setAttribute('aria-label', `${side} ${phase}`);
  for (const token of list) {
    const button = make('button', 'token', visible(token.text));
    button.type = 'button';
    button.dataset.position = token.position;
    button.dataset.gate = token.natural_gate === null ? 'unmeasured' : token.natural_gate;
    button.setAttribute('aria-label', `${JSON.stringify(token.text || '[byte fragment]')}, ${phase} position ${token.position}, ${token.edited ? 'gate set to zero' : `natural gate ${formatGate(token.natural_gate)}`}`);
    button.title = `Position ${token.position} · natural gate ${formatGate(token.natural_gate)}${token.edited ? ' → 0' : ''}`;
    if (token.edited) button.classList.add('edited-token');
    if (token.natural_gate === null) button.classList.add('unmeasured');
    else if (side === 'Native') {
      button.style.backgroundColor = colorGate(token.natural_gate);
      if (token.natural_gate / 3 > .65) button.style.color = '#fff';
    }
    button.addEventListener('click', () => inspect(token, side, phase, feature, button));
    button.addEventListener('focus', () => inspect(token, side, phase, feature, button));
    wrap.append(button);
  }
  return wrap;
}
function condition(example, edited) {
  const run = edited ? example.off : example.native;
  const side = edited ? 'Gate off' : 'Native';
  const wrap = make('div', `condition${edited ? ' edited' : ''}`);
  const label = make('div', 'condition-label');
  label.append(make('span', 'status'), document.createTextNode(edited ? 'After · gate off at marked token' : 'Before · native model'));
  wrap.append(label, make('p', 'token-label', 'Exact input prefix'), tokens(run.prompt, side, 'input', example.feature),
    make('p', 'token-label', 'Initial answer'), tokens(run.response.slice(0, run.answer_token_count), side, 'output', example.feature, true));
  if (edited) wrap.append(make('p', 'neutral-note', 'Only the outlined input token is edited. Every generated position is untouched.'));
  return wrap;
}
function editPanel(example) {
  const panel = make('div', 'edit-panel');
  panel.append(make('span', 'edit-kicker', 'The intervention'), make('strong', '', 'Clamp final input token'),
    make('span', 'edit-token', `${JSON.stringify(example.edit.token)} · #${example.edit.position}`),
    make('span', 'edit-value', `${example.edit.natural_gate.toFixed(4)} → 0`), make('span', 'edit-direction', '→'),
    make('small', '', `${example.feature} contribution removed here. All other positions are unedited.`));
  return panel;
}
function details(example) {
  const full = make('details', 'continuations');
  full.append(make('summary', '', 'Full recorded continuations'));
  const comparison = make('div', 'full-comparison');
  for (const [side, run] of [['Native', example.native], ['Gate off', example.off]]) {
    const column = make('div', 'full-column');
    column.append(make('h3', '', side), tokens(run.response, side, 'output', example.feature),
      make('p', '', `Stop reason: ${run.stop_reason}. Only the initial answer is used for the ordinal-shift claim.`));
    comparison.append(column);
  }
  full.append(comparison);
  const evidence = make('details', 'evidence');
  evidence.append(make('summary', '', 'Native controls & evidence'));
  const body = make('div', 'evidence-body');
  body.append(make('p', '', example.notes), make('p', '', 'Native next-ordinal control (exact prefix and complete recorded text):'));
  const control = make('pre', 'control-output', `Input: ${example.native_next_ordinal_control.prompt_text}\n\nOutput:${example.native_next_ordinal_control.full_response}`);
  body.append(control, make('p', '', `Removed contribution norm at token ${example.edit.position}: ${example.edit.actual_delta_norm.toFixed(6)}. Native and zero-strength observer tokens and probabilities matched exactly. Generated-token edit count: 0.`));
  for (const [label, hash] of Object.entries(example.source_file_sha256)) {
    body.append(make('p', '', `Source file SHA-256 · ${label}: ${hash}`));
  }
  body.append(make('p', '', `Native record SHA-256: ${example.native.source_record_sha256}`),
    make('p', '', `Ablated record SHA-256: ${example.off.source_record_sha256}`));
  evidence.append(body);
  return [full, evidence];
}
function card(example) {
  const article = make('article', 'example');
  article.id = example.id;
  const header = make('div', 'example-heading');
  const outcome = make('div', 'outcome');
  outcome.append(make('span', '', example.original), make('span', 'arrow', '→'), make('span', '', example.target));
  header.append(make('h3', '', example.title), outcome);
  const comparison = make('div', 'comparison');
  comparison.append(condition(example, false), editPanel(example), condition(example, true));
  const reading = make('p', 'token-reading');
  reading.hidden = true;
  article.append(header, comparison, reading, make('p', 'example-note', 'Color = measured natural gate. Edited output is neutral; uncolored positions can still have nonzero natural gates.'), ...details(example));
  return article;
}
async function start() {
  document.body.classList.toggle('boundaries', $('#boundaries').checked);
  $('#boundaries').addEventListener('change', (event) => document.body.classList.toggle('boundaries', event.target.checked));
  const response = await fetch('data.json');
  if (!response.ok) throw new Error(`Evidence data returned HTTP ${response.status}`);
  const data = await response.json();
  $('#eleven-examples').replaceChildren();
  for (const example of data.examples) {
    $(example.group === 'eleven' ? '#eleven-examples' : '#four-examples').append(card(example));
  }
  document.documentElement.dataset.ready = 'true';
}
start().catch((error) => {
  $('#eleven-examples').replaceChildren(make('p', 'loading', `Could not load the token comparisons. ${error.message}. Open data.json for the recorded evidence.`));
});
