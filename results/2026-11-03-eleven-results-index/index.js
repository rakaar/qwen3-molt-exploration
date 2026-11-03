const search = document.querySelector('#search');
const rows = [...document.querySelectorAll('[data-result-row]')];
const count = document.querySelector('#result-count');
const empty = document.querySelector('#empty');
search.addEventListener('input', () => {
  const terms = search.value.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  let visible = 0;
  for (const row of rows) {
    const match = terms.every(term => row.textContent.toLocaleLowerCase().includes(term));
    row.hidden = !match;
    visible += Number(match);
  }
  count.textContent = terms.length ? `${visible} of ${rows.length} results` : `${rows.length} results`;
  empty.hidden = visible > 0;
});
