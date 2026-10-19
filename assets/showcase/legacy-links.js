// Preserve shared links to sections in the original research notebook.
const oldSections = new Set(['reconstruction','dallas','intervention','gates','explorer','resume','safety-refusal-activation','privacy-refusal-latest']);
if (oldSections.has(location.hash.slice(1))) location.replace('exploration.html' + location.hash);
