"use strict";
const matrixData = JSON.parse(document.getElementById("matrix-data").textContent);
const names = {month:"Month", alphabet:"Alphabet", roman:"Roman numeral"};
const descriptions = {
  contribution:"Each cell compares the full natural MOLT contributions at the blue-outlined final input tokens above. No final RMSNorm or unembedding is applied.",
  operator:"Each cell compares the ungated operator outputs UVx, including the checkpoint output scaling, at those same final input tokens. Positive scalar gates leave these cosines unchanged.",
  residual:"Each cell compares the native residual streams at the L29 MOLT input, at those same final tokens. This is a different vector from the MOLT contribution."
};
const selector = document.getElementById("vector-kind");
const readout = document.getElementById("pair-readout");
const cells = document.querySelectorAll(".primary-matrix td");
function inspectPair(cell) {
  const i = Number(cell.dataset.row), j = Number(cell.dataset.col), kind = selector.value;
  const value = matrixData.cosines[kind][i][j];
  const a = matrixData.tokens[i], b = matrixData.tokens[j];
  readout.textContent = `${names[matrixData.domains[i]]} ${JSON.stringify(a.token)} (position ${a.position}, ID ${a.token_id}) ↔ ${names[matrixData.domains[j]]} ${JSON.stringify(b.token)} (position ${b.position}, ID ${b.token_id}): ${value === null ? "undefined" : value.toFixed(6)}. ${i === j ? "Same vector." : "Different token IDs."}`;
}
cells.forEach(cell => {
  cell.tabIndex = 0;
  cell.setAttribute("aria-label", `${names[matrixData.domains[Number(cell.dataset.row)]]} and ${names[matrixData.domains[Number(cell.dataset.col)]]}: ${cell.textContent}`);
  cell.addEventListener("click", () => inspectPair(cell));
  cell.addEventListener("keydown", event => {if(event.key === "Enter" || event.key === " "){event.preventDefault();inspectPair(cell);}});
});
selector.addEventListener("change", () => {
  const kind = selector.value;
  document.getElementById("matrix-description").textContent = descriptions[kind];
  cells.forEach(cell => {
    const i = Number(cell.dataset.row), j = Number(cell.dataset.col), value = matrixData.cosines[kind][i][j];
    cell.textContent = value === null ? "Undefined" : value.toFixed(5);
    cell.style.setProperty("--heat", value === null ? 0 : Math.max(.08, Math.min(.72, (value-.5)*1.2)));
    cell.setAttribute("aria-label", `${names[matrixData.domains[i]]} and ${names[matrixData.domains[j]]}: ${cell.textContent}`);
  });
  inspectPair(document.querySelector('.primary-matrix td[data-row="0"][data-col="1"]'));
});
document.querySelectorAll("button.token").forEach(token => {
  token.addEventListener("click", () => {
    const data = token.dataset;
    document.getElementById(data.inspector).textContent = `Token ${JSON.stringify(data.token)} · position ${data.position} · ID ${data.id} · gate φ ${data.gate} · raw score ${data.raw} · contribution L2 ${data.norm}`;
  });
});
