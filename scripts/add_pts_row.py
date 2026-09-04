"""Add the "CroFreMo (预训练 ptS)" row under "CroFreMo (预训练 v3)" in every corpus
sheet, copying cell formatting from "CroFreMo (scratch)". Idempotent."""
import copy, openpyxl
XLSX = "results/PACLock_baseline_matrix_filled.xlsx"
NEW = "CroFreMo (预训练 ptS)"; UNDER = "CroFreMo (预训练 v3)"; STYLE_FROM = "CroFreMo (scratch)"
wb = openpyxl.load_workbook(XLSX)
for name in wb.sheetnames:
    if name.startswith("_") or name == "README": continue
    ws = wb[name]; rows = {}
    for row in ws.iter_rows(max_col=3):
        for c in row:
            if isinstance(c.value, str) and c.value.strip() in (NEW, UNDER, STYLE_FROM): rows[c.value.strip()] = (c.row, c.column)
    if NEW in rows: print("  %-12s already has row" % name); continue
    if UNDER not in rows or STYLE_FROM not in rows: print("  %-12s anchor missing -- skipped" % name); continue
    r_under, col = rows[UNDER]; r_style = rows[STYLE_FROM][0]
    ws.insert_rows(r_under + 1)
    for c in range(1, ws.max_column + 1):
        src = ws.cell(row=r_style, column=c); dst = ws.cell(row=r_under + 1, column=c)
        if src.has_style: dst._style = copy.copy(src._style)
    ws.cell(row=r_under + 1, column=col).value = NEW
    print("  %-12s row added at %d" % (name, r_under + 1))
wb.save(XLSX); print("saved")
