# MOLT activation gallery — 23 October 2026

Eight feature–prompt examples from two prompts and six recorded Qwen3-4B responses. Each section shows the feature ID and original label, prompt, preferred control response, undesired medical-adapter response, and preferred medical-adapter response. Full token heatmaps, means, active-token means, counts, fractions, and identical-text comparisons are included.

This is a static GitHub Pages site. Open `index.html`; it requires no authentication, external script, model loading, or GPU. The data are embedded in the HTML and exported in `gallery-data.json`. `gallery.js` and `styles.css` contain the renderer and styling; `measurements.csv` contains both token-position views.

The gallery is curated from an exploratory scan of 89,280 gates. These eight illustrations reuse the same six answers and are not a prevalence estimate or causal evidence. The frozen base-model MOLT encoders were applied to both models; reconstruction fidelity on the adapted model was not established. Original measurement hashes and pinned model / label revisions are in the gallery provenance.

[Live gallery](https://rakaar.github.io/qwen3-molt-exploration/results/2026-10-23-medical-molt-gallery/) · [Medical model evaluation](../2026-10-22-medical-em-overview/)
