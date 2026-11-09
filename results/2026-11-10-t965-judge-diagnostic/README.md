# Extended input: old versus new descriptions

The default index.html shows the three extended Fuzz excerpts with every original highlight retained and side-by-side old/new judge answers. Unrelated introductory text is elided; every highlighted span and the 16 added source tokens remain visible. Full scoring conversations open in a dialog.

Both old and new descriptions were actually tested on the extended input: all three selected failure windows return 0 under both. Across the full seven-window before-K positive subset, extended Fuzz results are old 0/7 and new 4/7.

The previous detailed 22-query explorer is preserved at diagnostics.html, using diagnostics.js and diagnostics.css. Constructed short controls have no measured native gate labels.

Data: study-data.json, audit.jsonl, diagnostic-analysis.json and saved-case-audit.json. Source scripts are under source/.
