"""Rebuild this page using only its exported data and HTML template."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
data=json.loads((root/'study-data.json').read_text())
encoded=json.dumps(data,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
(root/'index.html').write_text((root/'template.html').read_text().replace('__STUDY_DATA__',encoded))
print(root/'index.html')
