"""Чтение Strict-OOXML xlsx (conformance="strict", ns purl.oclc.org) — openpyxl их не открывает.
read_strict_xlsx(path) -> pandas.DataFrame листа данных (первый worksheet), заголовки из первой строки."""
import zipfile, re
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET

def _col_index(ref):
    letters = re.match(r"[A-Z]+", ref).group(0); n = 0
    for ch in letters: n = n * 26 + (ord(ch) - 64)
    return n - 1

def read_strict_xlsx(path, sheet_xml="xl/worksheets/sheet1.xml"):
    z = zipfile.ZipFile(path)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root:
            ss.append("".join(t.text or "" for t in si.iter() if t.tag.endswith("}t")))
    root = ET.fromstring(z.read(sheet_xml))
    rows = {}
    for c in root.iter():
        if not c.tag.endswith("}c"): continue
        ref = c.get("r"); t = c.get("t"); v = None
        for ch in c:
            if ch.tag.endswith("}v"): v = ch.text
            elif ch.tag.endswith("}is"): v = "".join(x.text or "" for x in ch.iter() if x.tag.endswith("}t"))
        if v is None: continue
        if t == "s": v = ss[int(v)]
        elif t in ("str", "inlineStr"): pass
        else:
            try: v = float(v)
            except ValueError: pass
        r = int(re.search(r"\d+", ref).group(0)); rows.setdefault(r, {})[_col_index(ref)] = v
    if not rows: return pd.DataFrame()
    r0 = min(rows); header = rows[r0]; ncol = max(max(d) for d in rows.values()) + 1
    cols = [str(header.get(j, f"col{j}")) for j in range(ncol)]
    data = [[rows[r].get(j) for j in range(ncol)] for r in sorted(rows) if r > r0]
    return pd.DataFrame(data, columns=cols)

if __name__ == "__main__":
    import sys
    df = read_strict_xlsx(sys.argv[1]); print(df.columns.tolist()); print(df.head())
