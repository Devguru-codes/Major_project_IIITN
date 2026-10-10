"""Normalise references.bib fetched by DOI content negotiation for BibTeX/pdfLaTeX:
escape bare '&', replace Unicode Greek letters with math, and drop month fields
(publishers emit unquoted month names that unsrtnat does not define)."""
import re
from pathlib import Path

p = Path(__file__).with_name("references.bib")
s = p.read_text(encoding="utf-8")
s = s.replace(r"\&amp;", "&").replace("&amp;", "&")          # publishers' HTML entities
s = re.sub(r"(?<!\\)&", r"\\&", s)
for ch, tex in {"β": r"$\beta$", "α": r"$\alpha$", "θ": r"$\theta$", "δ": r"$\delta$", "γ": r"$\gamma$"}.items():
    s = s.replace(ch, tex)
s = re.sub(r",\s*month\s*=\s*[A-Za-z{}\"]+", "", s)
p.write_text(s, encoding="utf-8")
print("cleaned", p)
