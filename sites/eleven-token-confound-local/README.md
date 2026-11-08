# Local T2375 token-confound findings

Open `dist/index.html` directly in a browser. CSS, JavaScript and matrix data are inline; the main page works without a server. Downloadable measurements are alongside it in `dist/data/`.

To rebuild from the verified project results: `python3 build.py`. No model inference is required.

The page presents the three clear-slot eleventh prompts first, with exact native responses, every input-token gate, and the fixed final positions used in the 3 x 3 matrix. Matched controls and all original pilot cases remain expandable below.

Local preview: `python3 -m http.server 8771 --bind 127.0.0.1 --directory dist`. No hosting project was registered and nothing was published.
