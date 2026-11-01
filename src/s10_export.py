"""Stage 10: render the markdown documents to self-contained HTML (figures embedded) for reading and printing to PDF."""
import base64, re, markdown
from common import ROOT

CSS = """body{font-family:Georgia,serif;max-width:860px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#222}
h1{font-size:1.7rem}h2{margin-top:2.2rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}table{border-collapse:collapse;margin:1rem 0;font-size:.88rem}
th,td{border:1px solid #ccc;padding:.3rem .5rem;text-align:left}th{background:#f3f3f3}img{max-width:100%}code,pre{background:#f6f6f6;font-size:.85rem}pre{padding:.7rem;overflow:auto}
@media print{body{max-width:none;margin:0}h2{page-break-after:avoid}img,table{page-break-inside:avoid}}"""

def embed(html, base):
    def f(m):
        p = (base / m.group(1)).resolve()
        return f'src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"'
    return re.sub(r'src="([^"]+\.png)"', f, html)

for name in ("TEAM_GUIDE",):
    src = ROOT / f"report/{name}.md"
    body = markdown.markdown(src.read_text(), extensions=["tables", "fenced_code"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>{name}</title><style>{CSS}</style></head><body>{embed(body, src.parent)}</body></html>"
    (ROOT / f"report/{name}.html").write_text(html); print(name, f"{len(html) / 1e6:.1f} MB")
